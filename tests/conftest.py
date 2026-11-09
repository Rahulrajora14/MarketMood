import numpy as np
import pandas as pd
import pytest

from marketmood.calendar import TradingCalendar


@pytest.fixture(scope="session")
def cal():
    return TradingCalendar("2015-01-01", "2019-12-31")


@pytest.fixture(scope="session")
def prices(cal):
    """Synthetic daily bars on real NYSE sessions for four tickers."""
    rng = np.random.default_rng(0)
    sessions = cal.between("2015-01-02", "2019-12-31")
    frames = []
    for t in ["AAA", "BBB", "CCC", "DDD"]:
        r = rng.normal(0.0003, 0.015, len(sessions))
        close = 100 * np.exp(np.cumsum(r))
        hi = close * (1 + np.abs(rng.normal(0, 0.008, len(close))))
        lo = close * (1 - np.abs(rng.normal(0, 0.008, len(close))))
        frames.append(pd.DataFrame({"date": sessions, "ticker": t, "open": close, "high": hi,
                                    "low": lo, "close": close,
                                    "volume": rng.integers(1e6, 5e6, len(close))}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def news_items(cal, prices):
    """Random news timestamps spread across the day, incl. weekends and after-hours."""
    rng = np.random.default_rng(1)
    n = 3000
    start = pd.Timestamp("2015-01-01", tz="America/New_York").value
    end = pd.Timestamp("2019-12-20", tz="America/New_York").value
    ts = pd.to_datetime(rng.integers(start, end, n), utc=True)
    df = pd.DataFrame({
        "ticker": rng.choice(["AAA", "BBB", "CCC", "DDD"], n),
        "published_utc": ts,
        "fb_pos": rng.uniform(0, 1, n),
    })
    df["fb_neg"] = (1 - df.fb_pos) * rng.uniform(0, 1, n)
    df["fb_neu"] = 1 - df.fb_pos - df.fb_neg
    df["fb_polarity"] = df.fb_pos - df.fb_neg
    df["lm_tone"] = rng.uniform(-1, 1, n)
    df["lm_unc"] = rng.integers(0, 3, n)
    return df
