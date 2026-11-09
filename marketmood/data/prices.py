"""Daily OHLCV from Yahoo Finance (primary) with Stooq as an independent cross-check."""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")          # BRK.B -> BRK-B


def fetch_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Split- and dividend-adjusted daily bars for every ticker, long format."""
    import yfinance as yf

    frames = []
    for t in tickers:
        for attempt in range(3):
            try:
                df = yf.download(_yahoo_symbol(t), start=start,
                                 end=(pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                                 auto_adjust=True, progress=False, threads=False)
                break
            except Exception as exc:  # network hiccups
                log.warning("yahoo %s attempt %d failed: %s", t, attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
                df = pd.DataFrame()
        if df.empty:
            log.warning("no Yahoo data for %s", t)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        df["ticker"] = t
        frames.append(df[COLS])
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_stooq(ticker: str) -> pd.DataFrame:
    sym = ticker.lower().replace(".", "-") + ".us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=COLS)
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    return df[COLS]


def fnspid_archive_prices(zip_path, ticker: str) -> pd.DataFrame:
    """Prices for one ticker from the FNSPID price archive (Stock_price/full_history.zip)."""
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        name = f"full_history/{ticker.replace('.', '-')}.csv"
        if name not in z.namelist():
            return pd.DataFrame(columns=COLS)
        df = pd.read_csv(z.open(name))
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["adj close"]
    df["ticker"] = ticker
    return df[COLS].sort_values("date")      # the archive is stored newest first


def cross_check(yahoo: pd.DataFrame, tickers: list[str], fallback_zip=None) -> pd.DataFrame:
    """Compare daily log returns from a second source, ticker by ticker.

    Stooq is tried first; if it cannot be reached, the FNSPID price archive is
    used. Returns are compared rather than levels because sources adjust for
    dividends differently; what matters is that the day-to-day moves agree.
    """
    rows = []
    for t in tickers:
        source = "stooq"
        try:
            s = fetch_stooq(t)
        except Exception:
            s = pd.DataFrame()
        if s.empty and fallback_zip is not None:
            source, s = "fnspid_archive", fnspid_archive_prices(fallback_zip, t)
        if s.empty:
            rows.append({"ticker": t, "status": "no second source"})
            continue
        y = yahoo[yahoo.ticker == t].sort_values("date").set_index("date")["close"]
        s = s.sort_values("date").set_index("date")["close"]
        both = pd.concat([np.log(y).diff(), np.log(s).diff()], axis=1, keys=["y", "s"]).dropna()
        if both.empty:
            rows.append({"ticker": t, "status": "no overlap"})
            continue
        gap = (both["y"] - both["s"]).abs()
        rows.append({"ticker": t, "status": "ok", "source": source, "days": len(both),
                     "corr": both.corr().iloc[0, 1],
                     "share_gt_50bp": float((gap > 0.005).mean())})
    return pd.DataFrame(rows)
