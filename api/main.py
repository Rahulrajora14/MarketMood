"""Read-only API over the trained artefacts.

    uvicorn api.main:app --port 8000

The dashboard talks only to this service, so what is shown in the demo is
exactly what the experiment pipeline produced.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from marketmood.calendar import TradingCalendar
from marketmood.config import PROJECT_ROOT, load_config

app = FastAPI(title="MarketMood API", version="0.1.0")

CFG = load_config(os.environ.get("MARKETMOOD_CONFIG"))
RAW, INTERIM, PROC = CFG.dir("raw"), CFG.dir("interim"), CFG.dir("processed")
TABLES = CFG.dir("reports") / "tables"
ART = PROJECT_ROOT / "artifacts"

SERIES_COLS = ["date", "ticker", "ret_1d", "har_weekly", "news_n", "news_fb_mean", "news_fb_3d",
               "news_lm_mean", "news_attn_abn", "reddit_n", "reddit_fb_mean", "epu", "y_dir",
               "y_logvol", "fwd_ret_1"]


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(df.to_json(orient="records", date_format="iso"))


@lru_cache(maxsize=1)
def panel() -> pd.DataFrame:
    p = pd.read_parquet(PROC / "panel.parquet")
    return p


@lru_cache(maxsize=2)
def predictions(task: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"predictions_{task}.parquet")


@lru_cache(maxsize=1)
def prices() -> pd.DataFrame:
    return pd.read_parquet(RAW / "prices.parquet")


@lru_cache(maxsize=1)
def headlines() -> pd.DataFrame:
    st = CFG["study"]
    cal = TradingCalendar(st["start"], st["end"], st["exchange"], st["timezone"])
    h = pd.read_parquet(INTERIM / "news_scored.parquet")
    keep = ["ticker", "stated", "title", "published_utc", "publisher", "fb_pos", "fb_neg", "fb_neu",
            "fb_polarity", "lm_tone"]
    h = h[[c for c in keep if c in h.columns]]
    h["session"] = cal.assign_session(h["published_utc"]).values
    return h


@lru_cache(maxsize=1)
def models() -> dict:
    meta = json.loads((ART / "models.json").read_text())
    return {k: {**v, **joblib.load(ART / v["file"])} for k, v in meta.items()}


def _table(name: str) -> list[dict]:
    path = TABLES / f"{name}.csv"
    if not path.exists():
        raise HTTPException(404, f"{name} has not been produced yet")
    return _records(pd.read_csv(path))


@app.get("/health")
def health():
    return {"status": "ok", "panel_rows": int(len(panel())),
            "tickers": int(panel().ticker.nunique()),
            "from": str(panel().date.min().date()), "to": str(panel().date.max().date())}


@app.get("/tickers")
def tickers():
    return sorted(panel().ticker.unique().tolist())


@app.get("/tables/{name}")
def table(name: str):
    allowed = {"summary_direction", "summary_volatility", "significance", "regime_direction",
               "regime_volatility", "epu_interaction", "backtest", "shuffled_label_control",
               "coverage_per_firm", "coverage_per_year", "fold_metrics_direction",
               "fold_metrics_volatility", "price_crosscheck", "universe_coverage",
               "leak_demo", "news_reaction_profile", "fnspid_timestamp_audit", "fnspid_cleaning",
               "universe_price_availability"}
    if name not in allowed:
        raise HTTPException(404, "unknown table")
    return _table(name)


@app.get("/series/{ticker}")
def series(ticker: str, start: str = "2016-01-01", end: str = "2023-12-31"):
    p = panel()
    cols = [c for c in SERIES_COLS if c in p.columns]
    d = p.loc[(p.ticker == ticker) & (p.date >= start) & (p.date <= end), cols]
    if d.empty:
        raise HTTPException(404, "no data for that ticker / range")
    px = prices()
    d = d.merge(px.loc[px.ticker == ticker, ["date", "close"]], on="date", how="left")
    clf = predictions("clf")
    keep = ["date", "ticker", "p_lightgbm:price", "p_lightgbm:finbert", "fold"]
    d = d.merge(clf[[c for c in keep if c in clf]], on=["date", "ticker"], how="left")
    reg = predictions("reg")
    keep = ["date", "ticker", "v_har_ols:price", "v_lightgbm:price", "v_lightgbm:finbert"]
    d = d.merge(reg[[c for c in keep if c in reg]], on=["date", "ticker"], how="left")
    return _records(d)


@app.get("/headlines/{ticker}")
def ticker_headlines(ticker: str, date: str, limit: int = Query(30, le=200)):
    h = headlines()
    d = h[(h.ticker == ticker) & (h.session == pd.Timestamp(date))]
    d = d.sort_values("published_utc").head(limit)
    return _records(d.drop(columns="session"))


@app.get("/predict")
def predict(ticker: str, date: str):
    """Score one firm-day with the exported models (trained on all data)."""
    p = panel()
    row = p[(p.ticker == ticker) & (p.date == pd.Timestamp(date))]
    if row.empty:
        raise HTTPException(404, "no panel row for that ticker and date")
    out = {"ticker": ticker, "date": date}
    for name, m in models().items():
        X = row[m["features"]]
        if name.endswith("|clf"):
            out[name] = float(m["model"].predict_proba(X)[:, 1][0])
        else:
            out[name] = float(np.exp(m["model"].predict(X)[0]))
    out["note"] = "in-sample scores from models refit on the full sample; use /series for out-of-sample"
    return out
