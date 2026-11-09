"""Turns panel and predictions into the data-inspection report, result tables and figures."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as viz
from .config import PROJECT_ROOT, Config
from .evaluate import backtest as BT
from .evaluate import regime as RG
from .evaluate import stats as ST
from .evaluate.metrics import direction_metrics, qlike, summarise_folds, vol_metrics

log = logging.getLogger("marketmood")


def _models(preds: pd.DataFrame, prefix: str) -> list[str]:
    return [c[len(prefix):] for c in preds.columns if c.startswith(prefix)]


def _label(model: str) -> str:
    learner, fset = model.split(":")
    names = {"majority": "Majority class", "persistence": "Persistence (random walk)",
             "last_vol": "Last-week vol (random walk)", "logistic": "Logistic",
             "lightgbm": "LightGBM", "har_ols": "HAR"}
    if learner in ("majority", "persistence", "last_vol"):
        return names[learner]
    return f"{names[learner]} | {viz.LABELS[fset]}"


# ====================================================================== inspection
def data_inspection(cfg: Config) -> None:
    viz.apply_style()
    proc, rep = cfg.dir("processed"), cfg.dir("reports")
    fig_dir, tab_dir = rep / "figures", rep / "tables"
    fig_dir.mkdir(exist_ok=True); tab_dir.mkdir(exist_ok=True)
    panel = pd.read_parquet(proc / "panel.parquet")

    per_firm = panel.groupby("ticker").agg(
        firm_days=("date", "size"), news_days=("news_has", "sum"), items=("news_n", "sum"))
    per_firm["news_day_share"] = per_firm.news_days / per_firm.firm_days
    if "reddit_has" in panel:
        per_firm["reddit_day_share"] = panel.groupby("ticker")["reddit_has"].mean()
    per_firm = per_firm.sort_values("news_day_share")
    per_firm.to_csv(tab_dir / "coverage_per_firm.csv")

    by_year = panel.groupby(panel.date.dt.year).agg(
        firm_days=("date", "size"), share_with_news=("news_has", "mean"),
        items=("news_n", "sum"), up_share=("y_dir", "mean"), mean_epu=("epu", "mean"))
    if "reddit_has" in panel:
        by_year["share_with_reddit"] = panel.groupby(panel.date.dt.year)["reddit_has"].mean()
    by_year.to_csv(tab_dir / "coverage_per_year.csv")

    # coverage per firm
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(per_firm) + 1.2))
    ax.barh(per_firm.index, per_firm.news_day_share, color=viz.COLORS["finbert"], height=0.65)
    ax.set_xlabel("Share of trading days with at least one news item")
    ax.set_title("News coverage by firm, 2012-2023")
    ax.grid(axis="y", visible=False)
    fig.savefig(fig_dir / "coverage_per_firm.png"); plt.close(fig)

    # items over time
    monthly = panel.set_index("date")["news_n"].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(monthly.index, monthly.values, color=viz.COLORS["finbert"], alpha=0.25, lw=0)
    ax.plot(monthly.index, monthly.values, color=viz.COLORS["finbert"], lw=1.2, label="News (FNSPID)")
    if "reddit_n" in panel:
        rm = panel.set_index("date")["reddit_n"].resample("ME").sum()
        ax.plot(rm.index, rm.values, color=viz.COLORS["all_text"], lw=1.2, label="Reddit posts")
        ax.legend(loc="upper left")
    ax.set_title("Text items per month across the universe")
    ax.set_ylabel("items")
    fig.savefig(fig_dir / "items_per_month.png"); plt.close(fig)

    # EPU
    e = panel.drop_duplicates("date").set_index("date")["epu"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(e.index, e.values, color=viz.MUTED, lw=0.5, alpha=0.6, label="daily")
    ax.plot(e.index, e.rolling(30).mean(), color=viz.INK, lw=1.4, label="30-day mean")
    ax.set_yscale("log")
    ax.set_yticks([25, 50, 100, 200, 400, 800], ["25", "50", "100", "200", "400", "800"])
    ax.minorticks_off()
    ax.set_ylim(15, 1000)
    ax.set_title("US daily Economic Policy Uncertainty index")
    viz.subtitle(ax, "Baker, Bloom & Davis, news-based; log scale")
    ax.legend(loc="upper left")
    fig.savefig(fig_dir / "epu_timeline.png"); plt.close(fig)

    # sentiment distribution
    d = panel[panel.news_has == 1]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(d["news_fb_mean"], bins=60, color=viz.COLORS["finbert"], alpha=0.85)
    ax.set_title("Daily FinBERT tone on firm-days with news")
    ax.set_xlabel("mean P(positive) - P(negative)")
    fig.savefig(fig_dir / "finbert_tone_hist.png"); plt.close(fig)

    lines = ["# Data inspection report", "",
             f"Panel: {len(panel):,} firm-days, {panel.ticker.nunique()} firms, "
             f"{panel.date.min().date()} to {panel.date.max().date()}.", "",
             f"* Firm-days with at least one news item: {panel.news_has.mean():.1%}",
             f"* Firm-days with no text at all: "
             f"{(1 - panel[[c for c in ['news_has', 'reddit_has'] if c in panel]].max(axis=1).mean()):.1%}",
             f"* News items mapped to sessions: {int(panel.news_n.sum()):,}",
             f"* Next-day up share (class balance): {panel.y_dir.mean():.3f}",
             f"* EPU median / 90th pct: {e.median():.0f} / {e.quantile(0.9):.0f}", "",
             "## Coverage per year", "", by_year.round(3).to_markdown(), "",
             "## Coverage per firm", "", per_firm.round(3).to_markdown()]
    (rep / "data_inspection.md").write_text("\n".join(lines), encoding="utf-8")


# ====================================================================== results
def evaluate_all(cfg: Config) -> None:
    viz.apply_style()
    proc, rep = cfg.dir("processed"), cfg.dir("reports")
    tab_dir = rep / "tables"
    ev, bt = cfg["evaluation"], cfg["backtest"]
    panel = pd.read_parquet(proc / "panel.parquet")

    clf = pd.read_parquet(proc / "predictions_clf.parquet")
    reg = pd.read_parquet(proc / "predictions_reg.parquet")
    clf_models, reg_models = _models(clf, "p_"), _models(reg, "v_")

    # EPU regime per test day, cut points from that fold's training window
    for preds in (clf, reg):
        preds["epu_regime"] = None
        for fold, d in preds.groupby("fold"):
            tr = panel[panel.date <= d["train_end"].iloc[0]].drop_duplicates("date")["epu"]
            preds.loc[d.index, "epu_regime"] = RG.epu_terciles(tr, d["epu"]).astype(str).values

    # ---------- per-fold metrics
    rows = []
    for fold, d in clf.groupby("fold"):
        for m in clf_models:
            rows.append({"fold": fold, "model": m,
                         **direction_metrics(d.y_dir.values, d[f"p_{m}"].values, d[f"thr_{m}"].iloc[0])})
    fold_clf = pd.DataFrame(rows)
    fold_clf.to_csv(tab_dir / "fold_metrics_direction.csv", index=False)
    sum_clf = summarise_folds(fold_clf, ["model"], ["bal_acc", "macro_f1", "auc"])
    sum_clf["label"] = sum_clf.model.map(_label)
    sum_clf.to_csv(tab_dir / "summary_direction.csv", index=False)

    rows = []
    for fold, d in reg.groupby("fold"):
        for m in reg_models:
            rows.append({"fold": fold, "model": m, **vol_metrics(d.y_logvol.values, d[f"v_{m}"].values)})
    fold_reg = pd.DataFrame(rows)
    fold_reg.to_csv(tab_dir / "fold_metrics_volatility.csv", index=False)
    sum_reg = summarise_folds(fold_reg, ["model"], ["rmse_logvol", "qlike", "r2_logvol"])
    sum_reg["label"] = sum_reg.model.map(_label)
    sum_reg.to_csv(tab_dir / "summary_volatility.csv", index=False)

    # ---------- significance: each text-augmented model vs its own price-only twin
    sets = sorted({m.split(":")[1] for m in clf_models} - {"price"})
    sig = []
    for learner in ("logistic", "lightgbm"):
        base = f"{learner}:price"
        y = clf.y_dir.values
        clf[f"loss_{base}"] = (y - clf[f"p_{base}"]) ** 2
        clf[f"hat_{base}"] = (clf[f"p_{base}"] > clf[f"thr_{base}"]).astype(int)
        for s in sets:
            m = f"{learner}:{s}"
            clf[f"loss_{m}"] = (y - clf[f"p_{m}"]) ** 2
            clf[f"hat_{m}"] = (clf[f"p_{m}"] > clf[f"thr_{m}"]).astype(int)
            dm = ST.diebold_mariano(ST.daily_loss_diff(clf, f"loss_{m}", f"loss_{base}").values)
            bs = ST.bootstrap_balacc_diff(clf, f"hat_{m}", f"hat_{base}", reps=ev["bootstrap_reps"],
                                          block=ev["bootstrap_block"], seed=cfg["seed"])
            sig.append({"task": "direction", "model": m, "vs": base, "loss": "Brier",
                        **{f"dm_{k}": v for k, v in dm.items()},
                        "balacc_diff": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
    a_vol = np.exp(2 * reg.y_logvol.values)
    for learner in ("har_ols", "lightgbm"):
        base = f"{learner}:price"
        reg[f"ql_{base}"] = _qlike_from_log(a_vol, reg[f"v_{base}"].values)
        for s in sets:
            m = f"{learner}:{s}"
            reg[f"ql_{m}"] = _qlike_from_log(a_vol, reg[f"v_{m}"].values)
            dm = ST.diebold_mariano(ST.daily_loss_diff(reg, f"ql_{m}", f"ql_{base}").values, h=5)
            sig.append({"task": "volatility", "model": m, "vs": base, "loss": "QLIKE",
                        **{f"dm_{k}": v for k, v in dm.items()}})
    sig = pd.DataFrame(sig)
    # Holm adjustment across every comparison we ran
    sig["dm_p_holm"] = _holm(sig["dm_p_value"].values)
    sig.to_csv(tab_dir / "significance.csv", index=False)

    # ---------- regimes
    rc = RG.regime_table(clf, clf_models, "clf")
    rr = RG.regime_table(reg, reg_models, "reg")
    rc.to_csv(tab_dir / "regime_direction.csv", index=False)
    rr.to_csv(tab_dir / "regime_volatility.csv", index=False)

    controls_v = ["har_daily", "har_weekly", "har_monthly"]
    inter = pd.concat([
        RG.interaction_test(panel, "y_logvol", "news_fb_mean", controls_v, "reg").assign(target="log vol"),
        RG.interaction_test(panel, "y_logvol", "news_attn_abn", controls_v, "reg").assign(target="log vol (attention)"),
        RG.interaction_test(panel, "y_dir", "news_fb_mean", ["ret_1d", "mkt_ret"], "clf").assign(target="direction"),
    ])
    inter.to_csv(tab_dir / "epu_interaction.csv", index=False)

    # ---------- backtest
    perf, curves = [], {}
    for m in ("lightgbm:price", "lightgbm:finbert", "logistic:price", "logistic:finbert"):
        if f"p_{m}" not in clf:
            continue
        daily = BT.long_short(clf, f"p_{m}", bt["quantile"], bt["cost_bps"], bt["lag_days"])
        curves[m] = daily
        perf.append({"model": m, **BT.performance(daily["net"]),
                     "gross_sharpe": BT.performance(daily["gross"])["sharpe"],
                     "avg_turnover": daily["turnover"].mean()})
    pd.DataFrame(perf).to_csv(tab_dir / "backtest.csv", index=False)

    # ---------- shuffled-label control
    shuf = proc / "predictions_clf_shuffled.parquet"
    if shuf.exists():
        s = pd.read_parquet(shuf)
        rows = []
        for fold, d in s.groupby("fold"):
            for m in _models(s, "p_"):
                rows.append({"fold": fold, "model": m,
                             **direction_metrics(d.y_dir.values, d[f"p_{m}"].values, d[f"thr_{m}"].iloc[0])})
        sc = summarise_folds(pd.DataFrame(rows), ["model"], ["bal_acc", "auc"])
        sc.to_csv(tab_dir / "shuffled_label_control.csv", index=False)

    # ---------- naive-alignment comparison
    variants = {"honest (next session)": clf}
    for tag, label in (("after_close", "leaky: used at close of D"),
                       ("same_day", "leaky: predicts day D itself")):
        f = proc / f"predictions_clf_leak_{tag}.parquet"
        if f.exists():
            variants[label] = pd.read_parquet(f)
    if len(variants) > 1:
        rows = []
        models_leak = _models(variants[list(variants)[1]], "p_")
        for m in models_leak:
            for label, src in variants.items():
                f = [direction_metrics(d.y_dir.values, d[f"p_{m}"].values, d[f"thr_{m}"].iloc[0])
                     for _, d in src.groupby("fold")]
                f = pd.DataFrame(f)
                rows.append({"model": m, "alignment": label, "bal_acc_mean": f.bal_acc.mean(),
                             "bal_acc_std": f.bal_acc.std(), "auc_mean": f.auc.mean()})
        ld = pd.DataFrame(rows)
        # would the leaky gain have looked "significant"? compare with the honest price-only twin
        tests = []
        for label, src in variants.items():
            for m in ("lightgbm:finbert", "logistic:finbert"):
                if f"p_{m}" not in src:
                    continue
                base = m.split(":")[0] + ":price"
                d = src[["date", "y_dir", f"p_{m}", f"thr_{m}"]].merge(
                    clf[["date", "ticker", "y_dir", f"p_{base}", f"thr_{base}"]].assign(_i=1),
                    left_index=True, right_index=True, suffixes=("", "_b"))
                d["hat_a"] = (d[f"p_{m}"] > d[f"thr_{m}"]).astype(int)
                d["hat_b"] = (d[f"p_{base}"] > d[f"thr_{base}"]).astype(int)
                bs = ST.bootstrap_balacc_diff(d, "hat_a", "hat_b", reps=ev["bootstrap_reps"],
                                              block=ev["bootstrap_block"], seed=cfg["seed"])
                tests.append({"model": m, "alignment": label, "gain_vs_price": bs["diff"],
                              "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
        ld = ld.merge(pd.DataFrame(tests), on=["model", "alignment"], how="left")
        ld.to_csv(tab_dir / "leak_demo.csv", index=False)

    _figures(cfg, fold_clf, sum_clf, sum_reg, fold_reg, rr, rc, curves)
    _leak_figure(cfg)
    _reaction_profile(cfg, panel)
    results_markdown(cfg)
    log.info("evaluation tables and figures written to %s", rep)


def _qlike_from_log(actual_var, pred_logvol):
    return qlike(actual_var, np.exp(2 * pred_logvol))


def _holm(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ====================================================================== figures
def _figures(cfg, fold_clf, sum_clf, sum_reg, fold_reg, rr, rc, curves) -> None:
    fig_dir = cfg.dir("reports") / "figures"

    def color_of(model):
        learner, fset = model.split(":")
        return viz.COLORS["baseline"] if learner in ("majority", "persistence", "last_vol") \
            else viz.COLORS[fset]

    # direction: dot + whisker per model
    d = sum_clf.sort_values("bal_acc_mean")
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.errorbar(d.bal_acc_mean, y, xerr=d.bal_acc_std, fmt="none", ecolor=viz.GRID, elinewidth=3)
    ax.scatter(d.bal_acc_mean, y, c=[color_of(m) for m in d.model], s=46, zorder=3)
    ax.axvline(0.5, color=viz.INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y, d.label)
    ax.grid(axis="y", visible=False)
    n_folds = fold_clf["fold"].nunique()
    ax.set_xlabel(f"Balanced accuracy (mean across {n_folds} yearly test folds, bar = 1 s.d.)")
    ax.set_title("Next-day direction: balanced accuracy by model")
    fig.savefig(fig_dir / "direction_balacc.png"); plt.close(fig)

    # volatility: QLIKE (lower is better), dot + whisker like the direction chart
    d = sum_reg.sort_values("qlike_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.errorbar(d.qlike_mean, y, xerr=d.qlike_std, fmt="none", ecolor=viz.GRID, elinewidth=3)
    ax.scatter(d.qlike_mean, y, c=[color_of(m) for m in d.model], s=46, zorder=3)
    ax.set_yticks(y, d.label)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("QLIKE loss, lower is better (mean across folds, bar = 1 s.d.)")
    ax.set_title("5-day volatility forecasts")
    fig.savefig(fig_dir / "volatility_qlike.png"); plt.close(fig)

    # fold-by-fold path
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for m in ("lightgbm:price", "lightgbm:finbert", "lightgbm:lm"):
        s = fold_clf[fold_clf.model == m]
        if len(s):
            ax.plot(s.fold.astype(int), s.bal_acc, marker="o", ms=4, lw=1.6,
                    color=color_of(m), label=viz.LABELS[m.split(":")[1]])
    ax.axhline(0.5, color=viz.INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_title("Direction accuracy by test year (LightGBM)")
    ax.set_ylabel("balanced accuracy")
    ax.legend(loc="best")
    fig.savefig(fig_dir / "direction_by_year.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for m in ("har_ols:price", "lightgbm:price", "lightgbm:finbert", "lightgbm:attention"):
        s = fold_reg[fold_reg.model == m]
        if len(s):
            ls = "--" if m.startswith("har") else "-"
            ax.plot(s.fold.astype(int), s.r2_logvol, marker="o", ms=4, lw=1.6, ls=ls,
                    color=viz.INK if m.startswith("har") else color_of(m), label=_label(m))
    ax.set_title("Volatility forecast R² by test year")
    ax.set_ylabel("out-of-sample R² (log vol)")
    ax.legend(loc="best")
    fig.savefig(fig_dir / "volatility_by_year.png"); plt.close(fig)

    # regimes: gain from text inside each EPU tercile
    def gain(table, metric, better_low):
        t = table.pivot(index="regime", columns="model", values=f"{metric}_mean").reindex(RG.LABELS)
        out = {}
        for s in ("attention", "lm", "finbert", "all_text"):
            m, b = f"lightgbm:{s}", "lightgbm:price"
            if m in t:
                out[s] = (t[b] - t[m]) if better_low else (t[m] - t[b])
        return pd.DataFrame(out)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (tab, metric, low, title, unit) in zip(axes, [
            (rc, "bal_acc", False, "Direction", "Δ balanced accuracy"),
            (rr, "qlike", True, "Volatility", "QLIKE reduction")]):
        g = gain(tab, metric, low)
        w = 0.8 / max(len(g.columns), 1)
        x = np.arange(len(g))
        for i, s in enumerate(g.columns):
            ax.bar(x + i * w - 0.4 + w / 2, g[s], width=w, color=viz.COLORS[s], label=viz.LABELS[s])
        ax.axhline(0, color=viz.INK, lw=0.8)
        ax.set_xticks(x, [f"{r} EPU" for r in g.index])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Gain over price-only LightGBM, by policy-uncertainty tercile", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "regime_gain.png"); plt.close(fig)

    # backtest
    if curves:
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        for m, s in curves.items():
            if m.startswith("lightgbm"):
                lab = viz.LABELS[m.split(":")[1]]
                ax.plot(s.index, (1 + s["net"]).cumprod(), lw=1.5, color=color_of(m), label=f"{lab}, after costs")
                ax.plot(s.index, (1 + s["gross"]).cumprod(), lw=1.0, ls="--", color=color_of(m),
                        label=f"{lab}, before costs")
        ax.axhline(1, color=viz.INK, lw=0.8)
        ax.set_title("Long-short backtest with a one-day lag")
        viz.subtitle(ax, "top vs bottom 20% of firms by predicted P(up); costs 5 bp per unit turnover")
        ax.set_ylabel("growth of 1")
        ax.legend(loc="best")
        fig.savefig(fig_dir / "backtest_equity.png"); plt.close(fig)


# ====================================================================== method diagrams
def method_figures(cfg: Config) -> None:
    """Two explanatory diagrams: news-to-session alignment and walk-forward folds."""
    from matplotlib.patches import FancyBboxPatch

    viz.apply_style()
    fig_dir = cfg.dir("reports") / "figures"

    # --- alignment timeline
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.set_xlim(0, 72); ax.set_ylim(-1.6, 2.4)
    ax.axis("off")
    days = [("Mon", 0), ("Tue", 24), ("Wed", 48)]
    for name, x0 in days:
        ax.add_patch(FancyBboxPatch((x0 + 9.5, -0.18), 6.5, 0.36, boxstyle="round,pad=0.02",
                                    fc="#E4DED3", ec="none"))
        ax.text(x0 + 12.75, -0.55, f"{name} session", ha="center", fontsize=8, color=viz.MUTED)
        ax.plot([x0 + 16, x0 + 16], [-0.35, 0.35], color=viz.INK, lw=1.6)
        ax.text(x0 + 16, 0.5, "16:00\nclose", ha="center", va="bottom", fontsize=7.5, color=viz.INK)
    ax.plot([0, 72], [0, 0], color=viz.INK, lw=0.8)
    # an item dated Monday, any clock time
    ax.add_patch(FancyBboxPatch((1, 1.3), 22, 0.6, boxstyle="round,pad=0.05",
                                fc=viz.COLORS["finbert"], ec="none", alpha=0.18))
    ax.text(12, 1.6, 'headline dated Monday (time unknown)', ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(40, 0.25), xytext=(20, 1.3),
                arrowprops=dict(arrowstyle="-|>", color=viz.COLORS["finbert"], lw=1.6))
    ax.text(31, 1.05, "used at Tuesday's close", fontsize=8.5, color=viz.COLORS["finbert"])
    ax.annotate("", xy=(64, -0.25), xytext=(40, -0.25),
                arrowprops=dict(arrowstyle="-|>", color=viz.INK, lw=1.2))
    ax.text(52, -1.0, "forecast: Tue close → Wed close", ha="center", fontsize=8.5)
    ax.annotate("", xy=(16, 0.25), xytext=(10, 1.3),
                arrowprops=dict(arrowstyle="-|>", color=viz.MUTED, lw=1, ls="--"))
    ax.text(0.5, -1.0, "dashed: naive join to the same day, which leaks after-close news",
            fontsize=7.5, color=viz.MUTED)
    ax.set_title("How a news item reaches a forecast")
    fig.savefig(fig_dir / "method_alignment.png"); plt.close(fig)

    # --- walk-forward folds
    v = cfg["validation"]
    years = list(range(int(cfg["study"]["start"][:4]), v["last_test_year"] + 1))
    tests = list(range(v["first_test_year"], v["last_test_year"] + 1))
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for i, ty in enumerate(tests):
        y = len(tests) - i
        ax.barh(y, ty - years[0] - 0.04, left=years[0], height=0.62, color=viz.COLORS["price"], alpha=0.55)
        ax.barh(y, 0.04, left=ty - 0.04, height=0.62, color=viz.PAPER)
        ax.barh(y, 1, left=ty, height=0.62, color=viz.COLORS["finbert"])
        ax.text(years[0] - 0.15, y, f"fold {ty}", ha="right", va="center", fontsize=8)
    ax.set_yticks([]); ax.set_xticks(years + [years[-1] + 1])
    ax.set_xticklabels([str(y) for y in years] + [""], fontsize=8)
    ax.set_xlim(years[0] - 1.3, years[-1] + 1.1)
    ax.grid(False); ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.COLORS["price"], alpha=0.55, label="training window (expanding)"),
                       Patch(color=viz.COLORS["finbert"], label="test year")],
              loc="lower right", ncol=2, fontsize=8, bbox_to_anchor=(1.0, -0.28))
    ax.set_title("Walk-forward evaluation with a 5-day embargo before every test year")
    fig.savefig(fig_dir / "method_folds.png"); plt.close(fig)


# ====================================================================== results tables as markdown
def results_markdown(cfg: Config) -> str:
    """All headline tables in one markdown file (docs/RESULTS_TABLES.md)."""
    tab = cfg.dir("reports") / "tables"

    def md(name, cols=None, rename=None, digits=4):
        p = tab / f"{name}.csv"
        if not p.exists():
            return "_not produced_"
        d = pd.read_csv(p)
        if cols:
            d = d[[c for c in cols if c in d]]
        if rename:
            d = d.rename(columns=rename)
        return d.round(digits).to_markdown(index=False)

    parts = [
        "# Result tables", "",
        "Generated by `python -m marketmood evaluate`. Means and standard deviations are across",
        "the yearly walk-forward test folds.", "",
        "## RQ1: next-day direction", "",
        md("summary_direction", ["label", "bal_acc_mean", "bal_acc_std", "macro_f1_mean", "auc_mean", "auc_std"]),
        "", "## RQ2: 5-day volatility", "",
        md("summary_volatility", ["label", "qlike_mean", "qlike_std", "rmse_logvol_mean", "r2_logvol_mean"]),
        "", "## Significance (text model vs its own price-only twin)", "",
        md("significance", ["task", "model", "vs", "loss", "dm_mean_diff", "dm_stat", "dm_p_value",
                            "dm_p_holm", "balacc_diff", "ci_low", "ci_high"]),
        "", "## RQ3: by policy-uncertainty tercile", "", "### Direction", "",
        md("regime_direction", ["regime", "model", "bal_acc_mean", "auc_mean"]),
        "", "### Volatility", "",
        md("regime_volatility", ["regime", "model", "qlike_mean", "r2_logvol_mean"]),
        "", "### Interaction regressions (sentiment x EPU, SE clustered by date)", "",
        md("epu_interaction"),
        "", "## Long-short backtest (5 bp costs, one-day lag)", "",
        md("backtest", digits=3),
        "", "## Honesty check: shuffled labels", "",
        md("shuffled_label_control"),
        "", "## Leak demonstration: naive same-day alignment", "",
        md("leak_demo"),
    ]
    text = "\n".join(parts) + "\n"
    out = PROJECT_ROOT / "docs" / "RESULTS_TABLES.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def _leak_figure(cfg: Config) -> None:
    p = cfg.dir("reports") / "tables" / "leak_demo.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    models = list(dict.fromkeys(d.model))
    labels = list(dict.fromkeys(d.alignment))
    palette = [viz.COLORS["price"], viz.COLORS["lm"], viz.COLORS["finbert"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(models))
    h = 0.8 / len(labels)
    for i, label in enumerate(labels):
        v = d[d.alignment == label].set_index("model").reindex(models)
        ax.barh(y - 0.4 + h * (i + 0.5), v.bal_acc_mean - 0.5, left=0.5, height=h * 0.92,
                color=palette[i % 3], label=label)
    ax.axvline(0.5, color=viz.INK, lw=0.8)
    ax.set_yticks(y, [_label(m) for m in models])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("balanced accuracy")
    ax.set_title("Same models, same folds: only the news alignment differs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=8)
    fig.savefig(cfg.dir("reports") / "figures" / "leak_demo.png"); plt.close(fig)


def _reaction_profile(cfg: Config, panel: pd.DataFrame) -> None:
    """Correlation between the tone of news dated D and the stock's return on D+k.

    A peak at k <= 0 means news mostly describes moves that already happened.
    """
    news = pd.read_parquet(cfg.dir("interim") / "news_scored.parquet", columns=["ticker", "stated", "fb_polarity"])
    news["date"] = pd.to_datetime(news["stated"].str.slice(0, 10))
    tone = news.groupby(["ticker", "date"])["fb_polarity"].mean().rename("tone").reset_index()
    p = panel[["ticker", "date", "ret_1d"]].sort_values(["ticker", "date"]).copy()
    rows = []
    for k in range(-3, 4):
        p[f"r{k}"] = p.groupby("ticker")["ret_1d"].shift(-k)
    d = p.merge(tone, on=["ticker", "date"], how="inner")
    for k in range(-3, 4):
        x = d[["tone", f"r{k}"]].dropna()
        r = x.corr().iloc[0, 1]
        se = 1 / np.sqrt(len(x))
        rows.append({"lag_k": k, "corr": r, "ci95": 1.96 * se, "n": len(x)})
    t = pd.DataFrame(rows)
    t.to_csv(cfg.dir("reports") / "tables" / "news_reaction_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cols = [viz.COLORS["finbert"] if k <= 0 else viz.COLORS["price"] for k in t.lag_k]
    ax.bar(t.lag_k, t["corr"], color=cols, width=0.62)
    ax.errorbar(t.lag_k, t["corr"], yerr=t.ci95, fmt="none", ecolor=viz.INK, elinewidth=1, capsize=3)
    ax.axhline(0, color=viz.INK, lw=0.8)
    ax.set_xticks(t.lag_k, [f"D{k:+d}" if k else "D" for k in t.lag_k])
    ax.set_xlabel("return on day ...")
    ax.set_ylabel("correlation with tone")
    ax.set_title("News tone follows prices more than it leads them")
    viz.subtitle(ax, "FinBERT tone of headlines dated D vs the same firm's daily return; bars = 95% CI")
    ax.grid(axis="x", visible=False)
    fig.savefig(cfg.dir("reports") / "figures" / "news_reaction_profile.png"); plt.close(fig)
