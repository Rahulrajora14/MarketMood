"""Price-only features. Every value on row (ticker, t) uses data up to the close of t."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


def add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s).diff())
    return df


def price_features(prices: pd.DataFrame, return_lags=(1, 2, 3, 5, 10, 20),
                   har=None, turnover_window: int = 20) -> pd.DataFrame:
    har = har or {"daily": 1, "weekly": 5, "monthly": 22}
    df = add_returns(prices)
    g = df.groupby("ticker", group_keys=False)

    feats = pd.DataFrame({"ticker": df["ticker"], "date": df["date"]}, index=df.index)

    # past returns over several horizons (cumulative log return ending at t)
    for k in return_lags:
        feats[f"ret_{k}d"] = g["ret"].transform(lambda s, k=k: s.rolling(k, min_periods=k).sum())

    # HAR-style volatility components, in log volatility units
    sq = df["ret"] ** 2
    for name, w in har.items():
        mv = sq.groupby(df["ticker"]).transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        feats[f"har_{name}"] = 0.5 * np.log(mv + EPS)

    # intraday range (Parkinson) is a less noisy daily volatility proxy than |r|
    feats["range_vol"] = np.log(np.log(df["high"] / df["low"]).clip(lower=0) + 1e-4)

    # trading activity relative to its own recent history
    vol_ma = g["volume"].transform(lambda s: s.rolling(turnover_window, min_periods=10).mean())
    feats["turnover_ratio"] = np.log((df["volume"] + 1) / (vol_ma + 1))

    # equal-weight universe return as a market factor (cross-section on the same day)
    mkt = df.groupby("date")["ret"].mean().rename("mkt_ret")
    feats = feats.merge(mkt, left_on="date", right_index=True, how="left")
    feats = feats.sort_values(["ticker", "date"])
    feats["mkt_ret_5d"] = feats.groupby("ticker")["mkt_ret"].transform(
        lambda s: s.rolling(5, min_periods=5).sum())

    # calendar indicators
    dow = feats["date"].dt.dayofweek
    for d in range(1, 5):
        feats[f"dow_{d}"] = (dow == d).astype(np.int8)
    ym = feats["date"].dt.to_period("M")
    last = feats.groupby([feats["ticker"], ym])["date"].transform("max")
    feats["month_end"] = (feats["date"] == last).astype(np.int8)
    return feats.reset_index(drop=True)


PRICE_FEATURES_BASE = ["har_daily", "har_weekly", "har_monthly", "range_vol",
                       "turnover_ratio", "mkt_ret", "mkt_ret_5d",
                       "dow_1", "dow_2", "dow_3", "dow_4", "month_end"]


def price_feature_names(return_lags=(1, 2, 3, 5, 10, 20)) -> list[str]:
    return [f"ret_{k}d" for k in return_lags] + PRICE_FEATURES_BASE


HAR_FEATURES = ["har_daily", "har_weekly", "har_monthly"]
