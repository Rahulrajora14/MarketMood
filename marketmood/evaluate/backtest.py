"""A deliberately plain long-short backtest with costs and an implementation lag.

Each day, firms are ranked by predicted P(up). The top quantile is bought and
the bottom quantile sold, equal weight, dollar neutral. A signal formed at the
close of t is traded at the close of t+1 (one-day lag) and so earns the return
from t+1 to t+2. Costs are charged on every unit of one-way turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def long_short(preds: pd.DataFrame, score: str, quantile: float = 0.2,
               cost_bps: float = 5.0, lag_days: int = 1) -> pd.DataFrame:
    ret_col = {0: "fwd_ret_1", 1: "fwd_ret_2"}[lag_days]
    d = preds[["date", "ticker", score, ret_col]].dropna().copy()
    d["rank"] = d.groupby("date")[score].rank(pct=True, method="first")
    d["w"] = 0.0
    n_long = d.groupby("date")["rank"].transform(lambda r: (r > 1 - quantile).sum())
    n_short = d.groupby("date")["rank"].transform(lambda r: (r <= quantile).sum())
    d.loc[d["rank"] > 1 - quantile, "w"] = 0.5 / n_long
    d.loc[d["rank"] <= quantile, "w"] = -0.5 / n_short

    W = d.pivot(index="date", columns="ticker", values="w").fillna(0.0)
    R = d.pivot(index="date", columns="ticker", values=ret_col).reindex_like(W).fillna(0.0)
    gross = (W * (np.exp(R) - 1)).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def performance(daily: pd.Series) -> dict:
    ann = 252
    eq = (1 + daily).cumprod()
    dd = eq / eq.cummax() - 1
    vol = daily.std() * np.sqrt(ann)
    return {
        "ann_return": float((1 + daily.mean()) ** ann - 1),
        "ann_vol": float(vol),
        "sharpe": float(daily.mean() * ann / vol) if vol > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "hit_rate": float((daily > 0).mean()),
    }
