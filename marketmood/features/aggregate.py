"""Firm-day aggregation of item-level sentiment onto the trading-session grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import TradingCalendar


def assign_sessions(items: pd.DataFrame, cal: TradingCalendar) -> pd.DataFrame:
    out = items.copy()
    out["session"] = cal.assign_session(out["published_utc"]).values
    return out.dropna(subset=["session"])


def aggregate_items(items: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """One row per (ticker, session) with mean tone, disagreement and volume."""
    g = items.groupby(["ticker", "session"])
    agg = pd.DataFrame({
        f"{prefix}_n": g.size(),
        f"{prefix}_fb_mean": g["fb_polarity"].mean(),
        f"{prefix}_fb_disp": g["fb_polarity"].std(ddof=0),
        f"{prefix}_fb_negshare": g["fb_neg"].apply(lambda s: float((s > 0.5).mean())),
        f"{prefix}_lm_mean": g["lm_tone"].mean(),
        f"{prefix}_lm_unc": g["lm_unc"].mean(),
        f"{prefix}_text_max_time": g["published_utc"].max(),
    })
    return agg.reset_index().rename(columns={"session": "date"})


def sentiment_features(grid: pd.DataFrame, agg: pd.DataFrame, prefix: str,
                       smoothing_days: int = 3) -> pd.DataFrame:
    """Place aggregates on the full ticker x session grid and derive features.

    Firm-days with no text get count 0 and neutral tone (0). The `has` flag lets
    a model tell "no news" apart from "neutral news".
    """
    df = grid.merge(agg, on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    n = df[f"{prefix}_n"].fillna(0)
    df[f"{prefix}_n"] = n
    for c in ["fb_mean", "fb_disp", "fb_negshare", "lm_mean", "lm_unc"]:
        df[f"{prefix}_{c}"] = df[f"{prefix}_{c}"].fillna(0.0)
    df[f"{prefix}_has"] = (n > 0).astype(np.int8)
    df[f"{prefix}_logn"] = np.log1p(n)

    g = df.groupby("ticker", group_keys=False)
    w = smoothing_days
    # count-weighted tone over the last w sessions (today included)
    for src in ["fb", "lm"]:
        num = (df[f"{prefix}_{src}_mean"] * n).groupby(df["ticker"]).transform(
            lambda s: s.rolling(w, min_periods=1).sum())
        den = n.groupby(df["ticker"]).transform(lambda s: s.rolling(w, min_periods=1).sum())
        df[f"{prefix}_{src}_{w}d"] = (num / den.replace(0, np.nan)).fillna(0.0)
    # abnormal attention: today's volume of coverage vs. its own trailing month
    df[f"{prefix}_attn_abn"] = df[f"{prefix}_logn"] - g[f"{prefix}_logn"].transform(
        lambda s: s.rolling(22, min_periods=5).mean())
    return df


def finbert_feature_names(prefix: str, w: int = 3) -> list[str]:
    return [f"{prefix}_fb_mean", f"{prefix}_fb_disp", f"{prefix}_fb_negshare",
            f"{prefix}_fb_{w}d", f"{prefix}_logn", f"{prefix}_has", f"{prefix}_attn_abn"]


def lm_feature_names(prefix: str, w: int = 3) -> list[str]:
    return [f"{prefix}_lm_mean", f"{prefix}_lm_unc", f"{prefix}_lm_{w}d",
            f"{prefix}_logn", f"{prefix}_has", f"{prefix}_attn_abn"]
