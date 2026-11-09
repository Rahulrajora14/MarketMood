"""Significance testing for forecast comparisons.

Panel forecasts are collapsed to one loss differential per trading day
(averaged across firms) so the tests run on a single time series and the
cross-sectional correlation within a day is respected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(d: np.ndarray, h: int = 1) -> dict:
    """DM test on a loss differential series d_t = L(model A) - L(model B).

    Negative mean => model A has lower loss. Uses a Newey-West long-run
    variance and the Harvey-Leybourne-Newbold small-sample correction.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    mean = d.mean()
    lag = max(h - 1, int(np.floor(T ** (1 / 3))))
    dc = d - mean
    gamma0 = np.dot(dc, dc) / T
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * np.dot(dc[k:], dc[:-k]) / T
    dm = mean / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return {"mean_diff": mean, "stat": stat, "p_value": p, "n_days": T}


def daily_loss_diff(df: pd.DataFrame, loss_a: str, loss_b: str) -> pd.Series:
    return df.groupby("date")[loss_a].mean() - df.groupby("date")[loss_b].mean()


def _stationary_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, wrap-around."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / block
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_balacc_diff(df: pd.DataFrame, pred_a: str, pred_b: str, y: str = "y_dir",
                          reps: int = 1000, block: float = 20, seed: int = 0) -> dict:
    """CI for balanced accuracy(A) - balanced accuracy(B), resampling whole days."""
    def counts(col):
        c = pd.DataFrame({"date": df["date"], "y": df[y], "h": df[col]})
        c["tp"] = (c.y == 1) & (c.h == 1)
        c["tn"] = (c.y == 0) & (c.h == 0)
        c["pos"] = c.y == 1
        c["neg"] = c.y == 0
        return c.groupby("date")[["tp", "tn", "pos", "neg"]].sum().values.astype(float)

    A, B = counts(pred_a), counts(pred_b)

    def ba(m):
        s = m.sum(0)
        return 0.5 * (s[0] / s[2] + s[1] / s[3])

    point = ba(A) - ba(B)
    rng = np.random.default_rng(seed)
    n = len(A)
    draws = np.empty(reps)
    for r in range(reps):
        i = _stationary_indices(n, block, rng)
        draws[r] = ba(A[i]) - ba(B[i])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": point, "ci_low": lo, "ci_high": hi,
            "p_le_0": float(np.mean(draws <= 0))}
