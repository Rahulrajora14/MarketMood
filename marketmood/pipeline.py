"""Stage functions that connect the six layers. Each stage reads the previous
stage's Parquet output, so any stage can be re-run on its own."""
from __future__ import annotations

import io
import json
from pathlib import Path
import logging
import time

import joblib
import numpy as np
import pandas as pd
import requests

from . import panel as P
from .calendar import TradingCalendar
from .config import PROJECT_ROOT, Config, save_universe
from .features import aggregate as A
from .features import price as PF
from .features import sentiment as S
from .models import param_grid, tune_and_fit

log = logging.getLogger("marketmood")

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
SP500_HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                     "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def calendar_for(cfg: Config) -> TradingCalendar:
    st = cfg["study"]
    start = (pd.Timestamp(st["start"]) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    return TradingCalendar(start, st["end"], st["exchange"], st["timezone"])


# ====================================================================== universe
def sp500_members_on(date: str, url: str = SP500_HISTORY_URL) -> list[str]:
    """Index members on a given date, from the public fja05680/sp500 history file."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    hist = pd.read_csv(io.StringIO(r.text))
    hist["date"] = pd.to_datetime(hist["date"])
    row = hist[hist["date"] <= pd.Timestamp(date)].iloc[-1]
    return sorted(t.strip() for t in row["tickers"].split(","))


def stage_fnspid(cfg: Config, members: list[str]) -> pd.DataFrame:
    """Filter both FNSPID files to the candidate firms and the study window."""
    from .data.fnspid_ranges import read_nasdaq_parallel
    from .data.news import audit_timestamps, combine_parts, read_fnspid

    raw, st, d = cfg.dir("raw"), cfg["study"], cfg["data"]
    parts = raw / "fnspid_parts"
    for f in d["fnspid_news_files"]:
        tag = "nasdaq" if "nasdaq" in f else "external"
        if list(parts.glob(f"{tag}_*.parquet")) and (parts / f"{tag}.done").exists():
            continue
        local = raw / "fnspid" / Path(f).name
        src = local if local.exists() else HF.format(repo=d["fnspid_repo"], file=f)
        log.info("reading %s from %s", tag, src)
        if tag == "nasdaq" and not local.exists():
            # 23 GB: fetched and parsed in parallel byte ranges
            read_nasdaq_parallel(str(src), st["start"], st["end"], set(members), parts)
        else:
            read_fnspid(src, st["start"], st["end"], set(members), parts, tag)
        (parts / f"{tag}.done").write_text("ok")
    news, stats = combine_parts(parts)
    pd.Series(stats).to_csv(cfg.dir("reports") / "tables" / "fnspid_cleaning.csv", header=["rows"])
    # the audit uses every raw row: the recurring stories it relies on are
    # exactly the multi-firm round-ups that the cleaning step removes
    import pyarrow.parquet as pq
    sample = pd.concat([pd.read_parquet(f, columns=["stated", "title"])
                        for f in sorted(parts.glob("*.parquet"))
                        if "stated" in pq.read_schema(f).names])
    audit = audit_timestamps(sample)
    audit.to_csv(cfg.dir("reports") / "tables" / "fnspid_timestamp_audit.csv", index=False)
    return news


def price_history_available(tickers: list[str], start: str, end: str,
                            need: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Yahoo prices for candidates (in the given order) and keep those
    whose history spans the whole window.

    Free price sources drop delisted firms, so a firm acquired or delisted
    during the sample cannot be followed to its end. Rather than hide this,
    the returned table lists every candidate checked and why it was kept or not.
    Stops once `need` firms qualify.
    """
    from .data.prices import fetch_yahoo

    rows, frames = [], []
    pad_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    for t in tickers:
        px = fetch_yahoo([t], pad_start, end)
        if px.empty:
            rows.append({"ticker": t, "available": False, "reason": "no Yahoo history (delisted or renamed)"})
            continue
        lo, hi = px.date.min(), px.date.max()
        ok = lo <= pd.Timestamp(start) - pd.DateOffset(months=6) and hi >= pd.Timestamp(end) - pd.DateOffset(days=10)
        rows.append({"ticker": t, "available": ok, "first": lo.date(), "last": hi.date(),
                     "reason": "ok" if ok else "history does not span the sample"})
        if ok:
            frames.append(px)
            if need and len(frames) >= need:
                break
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame(rows), prices


def stage_universe(cfg: Config) -> list[str]:
    raw = cfg.dir("raw")
    st = cfg["study"]
    members = sp500_members_on(st["start"])
    pd.Series(members, name="ticker").to_csv(raw / "sp500_members_start.csv", index=False)
    log.info("%d S&P 500 members on %s", len(members), st["start"])

    cand_path = raw / "news_candidates.parquet"
    if not cand_path.exists():
        stage_fnspid(cfg, members).to_parquet(cand_path, index=False)
    news = pd.read_parquet(cand_path)

    cal = calendar_for(cfg)
    news = A.assign_sessions(news, cal)
    first_test = cfg["validation"]["first_test_year"]
    # coverage is measured on the first training window only, so the choice of
    # firms does not use any information from the test years
    window = cal.between(st["start"], f"{first_test - 1}-12-31")
    w = news[news.session.isin(window)]
    cov = (w.groupby("ticker")["session"].nunique() / len(window)).rename("news_day_share")
    full = (news.groupby("ticker")["session"].nunique() /
            len(cal.between(st["start"], st["end"]))).rename("news_day_share_full")
    cov = pd.concat([cov, full, news.groupby("ticker").size().rename("items")], axis=1)
    cov = cov.fillna(0).sort_values("news_day_share", ascending=False)
    cov.to_csv(cfg.dir("reports") / "tables" / "universe_coverage.csv")

    u = cfg["universe"]
    eligible = cov[cov.news_day_share >= u["min_news_days_share"]]
    avail, prices = price_history_available(list(eligible.index), st["start"], st["end"],
                                            need=u["candidates_top_n"])
    avail.to_csv(cfg.dir("reports") / "tables" / "universe_price_availability.csv", index=False)
    tickers = sorted(avail.loc[avail.available, "ticker"])
    prices.to_parquet(raw / "prices.parquet", index=False)
    save_universe(cfg, tickers)
    news[news.ticker.isin(tickers)].drop(columns="session").to_parquet(raw / "news.parquet", index=False)
    log.info("universe frozen: %d tickers", len(tickers))
    return tickers


# ====================================================================== download
def stage_download(cfg: Config) -> None:
    from .data.epu import fetch_epu
    from .data.prices import cross_check, fetch_yahoo
    from .data.reddit import fetch_reddit

    raw, st, tickers = cfg.dir("raw"), cfg["study"], cfg.tickers
    start = (pd.Timestamp(st["start"]) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    if not (raw / "prices.parquet").exists():
        fetch_yahoo(tickers, start, st["end"]).to_parquet(raw / "prices.parquet", index=False)
    cc_path = cfg.dir("reports") / "tables" / "price_crosscheck.csv"
    if not cc_path.exists():
        z = raw / "fnspid" / "full_history.zip"
        cross_check(pd.read_parquet(raw / "prices.parquet"), tickers,
                    z if z.exists() else None).to_csv(cc_path, index=False)
    if not (raw / "epu.parquet").exists():
        fetch_epu(cfg["data"]["epu_fred_series"]).to_parquet(raw / "epu.parquet", index=False)
    rd = cfg["data"]["reddit"]
    log.info("prices, cross-check and EPU ready")
    if rd["enabled"] and not (raw / "reddit.parquet").exists():
        fetch_reddit(tickers, rd["subreddits"], st["start"], st["end"],
                     max_pages=rd.get("max_pages", 15)).to_parquet(
            raw / "reddit.parquet", index=False)


# ====================================================================== sentiment
def stage_sentiment(cfg: Config) -> None:
    raw, interim = cfg.dir("raw"), cfg.dir("interim")
    sc = cfg["sentiment"]
    lm = S.LMScorer()

    def factory():
        return S.FinBERTScorer(sc["finbert_model"], sc["max_length"], sc["batch_size"],
                               quantize=sc.get("quantize", False))

    cache = interim / "finbert_cache.parquet"
    news = pd.read_parquet(raw / "news.parquet")
    news = news[news.ticker.isin(cfg.tickers)].reset_index(drop=True)
    text = S.compose_text(news["title"], news.get("lead"), sc["use_lead_paragraph"])
    fb = S.score_with_cache(text, cache, factory)
    out = pd.concat([news, lm.score(text), fb[["fb_pos", "fb_neg", "fb_neu", "fb_polarity"]]], axis=1)
    out.to_parquet(interim / "news_scored.parquet", index=False)

    rp = raw / "reddit.parquet"
    if rp.exists():
        red = pd.read_parquet(rp)
        if len(red):
            red = red[red.ticker.isin(cfg.tickers)].reset_index(drop=True)
            fb = S.score_with_cache(red["title"], cache, factory)
            out = pd.concat([red, lm.score(red["title"]),
                             fb[["fb_pos", "fb_neg", "fb_neu", "fb_polarity"]]], axis=1)
            out.to_parquet(interim / "reddit_scored.parquet", index=False)


# ====================================================================== panel
def stage_build(cfg: Config) -> pd.DataFrame:
    raw, interim, proc = cfg.dir("raw"), cfg.dir("interim"), cfg.dir("processed")
    st, fc, tc = cfg["study"], cfg["features"], cfg["targets"]
    cal = calendar_for(cfg)

    prices = pd.read_parquet(raw / "prices.parquet")
    prices = prices[prices.ticker.isin(cfg.tickers)]
    # keep exchange sessions only (drops the odd vendor row on a holiday)
    prices = prices[prices.date.isin(cal.sessions)]

    feats = PF.price_features(prices, fc["return_lags"], fc["har_windows"], fc["turnover_window"])
    targets = P.make_targets(prices, cal, tc["direction_horizon"], tc["vol_horizon"])
    panel = feats.merge(targets, on=["ticker", "date"], how="left")
    panel["decision_time"] = pd.Series(cal.decision_time(panel["date"]), index=panel.index)

    w = fc["sentiment_smoothing_days"]
    news = A.assign_sessions(pd.read_parquet(interim / "news_scored.parquet"), cal)
    panel = A.sentiment_features(panel, A.aggregate_items(news, "news"), "news", w)

    rp = interim / "reddit_scored.parquet"
    if rp.exists():
        red = A.assign_sessions(pd.read_parquet(rp), cal)
        panel = A.sentiment_features(panel, A.aggregate_items(red, "reddit"), "reddit", w)

    panel = P.attach_epu(panel, pd.read_parquet(raw / "epu.parquet"))
    panel = panel[(panel.date >= pd.Timestamp(st["start"])) & (panel.date <= pd.Timestamp(st["end"]))]
    panel = panel.dropna(subset=["har_monthly", "ret_20d"]).reset_index(drop=True)
    panel.to_parquet(proc / "panel.parquet", index=False)
    log.info("panel: %d rows, %d tickers, %s to %s", len(panel), panel.ticker.nunique(),
             panel.date.min().date(), panel.date.max().date())
    return panel


def stage_leakdemo(cfg: Config) -> None:
    """Rerun the same models with two common but leaky ways of joining news to prices.

    after_close : news dated D is used at the close of D. Stories released after
                  the 16:00 close (earnings, guidance, downgrades) then "forecast"
                  the move they caused on D+1.
    same_day    : news dated D is used to forecast the return *of day D* itself,
                  i.e. the label is the move the story was written about.
    Everything else (features, folds, learners, grid) is identical to the honest run.
    """
    interim, proc = cfg.dir("interim"), cfg.dir("processed")
    cal = calendar_for(cfg)
    w = cfg["features"]["sentiment_smoothing_days"]
    panel = pd.read_parquet(proc / "panel.parquet")
    base = pd.read_parquet(interim / "news_scored.parquet")
    drop = [c for c in panel.columns if c.startswith("news_")]
    models = ["logistic:finbert", "lightgbm:price", "lightgbm:finbert", "lightgbm:lm"]

    for variant in ("after_close", "same_day"):
        news = base.copy()
        if variant == "after_close":
            news["published_utc"] = news["published_utc"] - pd.Timedelta(days=1)  # -> session D
            news = A.assign_sessions(news, cal)
        else:
            start_of_day = news["published_utc"] - pd.Timedelta(hours=23, minutes=59)
            sess = cal.assign_session(start_of_day)                              # session D
            news["session"] = cal.shift(sess, -1).values                         # its label = return of D
            news = news.dropna(subset=["session"])
        naive = A.sentiment_features(panel.drop(columns=drop), A.aggregate_items(news, "news"), "news", w)
        naive = naive[panel.columns]
        stage_train(cfg, only=models, panel=naive, tag=f"_leak_{variant}", tasks=("clf",))


# ====================================================================== feature sets
def feature_sets(cfg: Config, panel: pd.DataFrame) -> dict[str, list[str]]:
    w = cfg["features"]["sentiment_smoothing_days"]
    price = PF.price_feature_names(cfg["features"]["return_lags"])
    attention = ["news_logn", "news_has", "news_attn_abn"]
    sets = {
        "price": price,
        "attention": price + attention,
        "lm": price + A.lm_feature_names("news", w),
        "finbert": price + A.finbert_feature_names("news", w),
    }
    reddit_cols = A.finbert_feature_names("reddit", w)
    # the plan allows dropping social media if it is too sparse to be informative
    min_cov = cfg["data"]["reddit"].get("min_coverage", 0.10)
    if all(c in panel.columns for c in reddit_cols) and panel["reddit_has"].mean() >= min_cov:
        sets["all_text"] = sets["finbert"] + reddit_cols
    return sets


def _run_specs(task: str, sets: dict) -> list[tuple[str, str, list[str]]]:
    """(learner, feature-set name, columns) for every model in one task."""
    specs = []
    if task == "clf":
        specs += [("majority", "price", sets["price"]), ("persistence", "price", sets["price"])]
        for learner in ("logistic", "lightgbm"):
            specs += [(learner, k, v) for k, v in sets.items()]
    else:
        specs.append(("last_vol", "price", sets["price"]))
        for k, v in sets.items():
            extra = [c for c in v if c not in sets["price"]]
            specs.append(("har_ols", k, PF.HAR_FEATURES + extra))
        specs += [("lightgbm", k, v) for k, v in sets.items()]
    return specs


# ====================================================================== training
def stage_train(cfg: Config, shuffle_labels: bool = False, only: list[str] | None = None,
                panel: pd.DataFrame | None = None, tag: str | None = None,
                tasks: tuple[str, ...] = ("clf", "reg")) -> None:
    proc = cfg.dir("processed")
    if panel is None:
        panel = pd.read_parquet(proc / "panel.parquet")
    sets = feature_sets(cfg, panel)
    v = cfg["validation"]
    folds = P.walk_forward_folds(panel["date"], v["first_test_year"], v["last_test_year"],
                                 v["embargo_days"])
    seed, budget = cfg["seed"], cfg["models"]["tuning_budget"]
    rng = np.random.default_rng(seed)

    for task, ycol in (("clf", "y_dir"), ("reg", "y_logvol")):
        if task not in tasks:
            continue
        data = panel.dropna(subset=[ycol]).reset_index(drop=True)
        specs = _run_specs(task, sets)
        if only:
            specs = [s for s in specs if f"{s[0]}:{s[1]}" in only]
        out, chosen = [], []
        for fold in folds:
            t0 = time.time()
            tr = data[data.date.isin(fold.train_sessions)]
            te = data[data.date.isin(fold.test_sessions)].copy()
            fit_s, val_s = P.inner_split(fold.train_sessions, v["inner_val_share"], v["embargo_days"])
            fit, val = tr[tr.date.isin(fit_s)], tr[tr.date.isin(val_s)]
            y_tr, y_fit = tr[ycol].values, fit[ycol].values
            if shuffle_labels:
                y_tr, y_fit = rng.permutation(y_tr), rng.permutation(y_fit)
            for learner, fset, cols in specs:
                grid = param_grid(cfg["models"], learner, budget, seed)
                res = tune_and_fit(learner, task, grid, fit[cols], y_fit, val[cols],
                                   val[ycol].values, tr[cols], y_tr, seed)
                name = f"{learner}:{fset}"
                if task == "clf":
                    te[f"p_{name}"] = res.estimator.predict_proba(te[cols])[:, 1]
                    te[f"thr_{name}"] = res.threshold
                else:
                    te[f"v_{name}"] = res.estimator.predict(te[cols])
                chosen.append({"fold": fold.name, "model": name, "params": json.dumps(res.params)})
            te["fold"] = fold.name
            te["train_end"] = fold.train_end
            out.append(te)
            log.info("%s fold %s: %d train / %d test rows, %.0fs", task, fold.name,
                     len(tr), len(te), time.time() - t0)
        suffix = tag if tag is not None else ("_shuffled" if shuffle_labels else "")
        preds = pd.concat(out, ignore_index=True)
        preds.to_parquet(proc / f"predictions_{task}{suffix}.parquet", index=False)
        if not shuffle_labels and not suffix:
            pd.DataFrame(chosen).to_csv(cfg.dir("reports") / "tables" / f"chosen_params_{task}.csv",
                                        index=False)


def stage_final_models(cfg: Config) -> None:
    """Refit the headline models on every available row, for the API."""
    proc = cfg.dir("processed")
    art = PROJECT_ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    panel = pd.read_parquet(proc / "panel.parquet")
    sets = feature_sets(cfg, panel)
    params = {t: pd.read_csv(cfg.dir("reports") / "tables" / f"chosen_params_{t}.csv")
              for t in ("clf", "reg")}
    meta = {}
    for task, ycol, learner in (("clf", "y_dir", "lightgbm"), ("reg", "y_logvol", "lightgbm")):
        data = panel.dropna(subset=[ycol])
        for fset in ("price", "finbert"):
            cols = sets[fset]
            name = f"{learner}:{fset}"
            p = params[task]
            last = json.loads(p[p.model == name].iloc[-1]["params"])
            res = tune_and_fit(learner, task, [last], data[cols], data[ycol].values,
                               data[cols].head(100), data[ycol].values[:100],
                               data[cols], data[ycol].values, cfg["seed"])
            fn = f"{task}_{learner}_{fset}.joblib"
            joblib.dump({"model": res.estimator, "features": cols, "threshold": res.threshold}, art / fn)
            meta[name + f"|{task}"] = {"file": fn, "params": last, "trained_through": str(data.date.max().date())}
    (art / "models.json").write_text(json.dumps(meta, indent=2))
