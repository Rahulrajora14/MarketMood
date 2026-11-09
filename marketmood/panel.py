"""Panel layer: targets, the joined firm-day panel, and walk-forward folds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import TradingCalendar

EPS = 1e-8


# ------------------------------------------------------------------ targets
def make_targets(prices: pd.DataFrame, cal: TradingCalendar,
                 direction_horizon: int = 1, vol_horizon: int = 5) -> pd.DataFrame:
    """All targets in one function, defined on changes rather than price levels.

    For the row (ticker, t):
      y_dir     = 1 if log(C[t+1] / C[t]) > 0 else 0
      y_logvol  = log( sqrt( mean(r[t+1]^2, ..., r[t+h]^2) ) ),  h = vol_horizon
      fwd_ret_1 = log(C[t+1] / C[t])           (kept for the backtest)
      fwd_ret_2 = log(C[t+2] / C[t+1])         (return earned with a one-day lag)
    target_time is the close at which the last ingredient of the target is known.
    """
    df = prices.sort_values(["ticker", "date"])[["ticker", "date", "close"]].copy()
    df["ret"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s).diff())
    g = df.groupby("ticker")["ret"]
    h = direction_horizon
    df["fwd_ret_1"] = g.shift(-1)
    df["fwd_ret_2"] = g.shift(-2)
    df["y_dir"] = (g.transform(lambda s: s.rolling(h).sum().shift(-h)) > 0).astype(float)
    df.loc[df["fwd_ret_1"].isna(), "y_dir"] = np.nan

    sq = df["ret"] ** 2
    fwd_mean = sq.groupby(df["ticker"]).transform(
        lambda s: s.rolling(vol_horizon, min_periods=vol_horizon).mean().shift(-vol_horizon))
    df["fwd_var"] = fwd_mean
    df["y_logvol"] = 0.5 * np.log(fwd_mean + EPS)

    # the latest moment any target value depends on
    pos = cal.position(df["date"])
    last = np.minimum(pos + max(h, vol_horizon), len(cal.sessions) - 1).astype(int)
    df["target_time"] = pd.Series(cal.closes_utc[last], index=df.index)
    return df.drop(columns=["close", "ret"])


# ------------------------------------------------------------------ EPU
def attach_epu(panel: pd.DataFrame, epu: pd.DataFrame) -> pd.DataFrame:
    """EPU for session t is the most recent daily value dated strictly before t.

    The daily index for date d is built from that day's newspapers, so it is
    not available at the close of d. Using d < t avoids that look-ahead.
    """
    e = epu.sort_values("date").rename(columns={"date": "epu_date"})
    e["epu_date"] = e["epu_date"].astype("datetime64[ns]")
    sessions = pd.DataFrame({"date": np.sort(panel["date"].unique())})
    sessions["date"] = sessions["date"].astype("datetime64[ns]")
    sessions["lookup"] = sessions["date"] - pd.Timedelta(days=1)
    m = pd.merge_asof(sessions.sort_values("lookup"), e, left_on="lookup",
                      right_on="epu_date", direction="backward")
    m["log_epu"] = np.log(m["epu"].clip(lower=1))
    return panel.merge(m[["date", "epu", "log_epu", "epu_date"]], on="date", how="left")


# ------------------------------------------------------------------ folds
@dataclass(frozen=True)
class Fold:
    name: str
    train_sessions: pd.DatetimeIndex
    test_sessions: pd.DatetimeIndex

    @property
    def train_end(self):
        return self.train_sessions.max()

    @property
    def test_start(self):
        return self.test_sessions.min()


def walk_forward_folds(sessions: pd.DatetimeIndex, first_test_year: int,
                       last_test_year: int, embargo: int,
                       start: str | None = None) -> list[Fold]:
    """Expanding-window folds, one calendar year of test data per fold.

    The last `embargo` sessions before each test year are dropped from the
    training window so no training target overlaps the test period.
    """
    sessions = pd.DatetimeIndex(np.sort(pd.unique(sessions)))
    if start is not None:
        sessions = sessions[sessions >= pd.Timestamp(start)]
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        test = sessions[sessions.year == y]
        if len(test) == 0:
            continue
        first = sessions.get_loc(test[0])
        cut = first - embargo
        if cut <= 0:
            continue
        train = sessions[:cut]
        folds.append(Fold(name=str(y), train_sessions=train, test_sessions=test))
    return folds


def inner_split(train_sessions: pd.DatetimeIndex, val_share: float, embargo: int):
    """Split a training window into (fit, validation) for tuning, with an embargo."""
    n = len(train_sessions)
    n_val = max(int(n * val_share), 20)
    fit = train_sessions[: n - n_val - embargo]
    val = train_sessions[n - n_val:]
    return fit, val
