"""Trading calendar and the timestamp -> decision-point mapping.

This is the one place in the project where temporal alignment is decided, so it
is also the one place it is tested (tests/test_calendar.py).

Convention
----------
* A *session* is an NYSE trading day, identified by its date.
* The *decision point* of a session is its official close (16:00 New York,
  13:00 on early-close days). Forecasts for session t are formed at that close.
* A text item belongs to the first session whose close is strictly after the
  item's publication time. An item stamped exactly at 16:00:00 is treated as
  after-close and goes to the next session.
"""
from __future__ import annotations

import exchange_calendars as xcals
import numpy as np
import pandas as pd


class TradingCalendar:
    def __init__(self, start: str, end: str, exchange: str = "XNYS",
                 tz: str = "America/New_York"):
        # pad the range so that items near the edges still find a session
        s = pd.Timestamp(start) - pd.Timedelta(days=40)
        e = pd.Timestamp(end) + pd.Timedelta(days=40)
        cal = xcals.get_calendar(exchange, start=s.strftime("%Y-%m-%d"),
                                 end=e.strftime("%Y-%m-%d"))
        self.tz = tz
        self.sessions = pd.DatetimeIndex(cal.sessions).tz_localize(None).normalize().as_unit("ns")
        closes = pd.DatetimeIndex(cal.closes.values)
        if closes.tz is None:
            closes = closes.tz_localize("UTC")
        self.closes_utc = closes.tz_convert("UTC").as_unit("ns")
        self._close_ns = self.closes_utc.asi8
        self._pos = pd.Series(np.arange(len(self.sessions)), index=self.sessions)

    # ------------------------------------------------------------------ mapping
    def assign_session(self, timestamps: pd.Series | pd.DatetimeIndex) -> pd.Series:
        """Map publication timestamps (tz-aware) to the session they inform."""
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if ts.tz is None:
            raise ValueError("timestamps must be timezone-aware; localise them first")
        ns = ts.tz_convert("UTC").as_unit("ns").asi8
        idx = np.searchsorted(self._close_ns, ns, side="right")
        out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
        ok = idx < len(self.sessions)
        out[ok] = self.sessions.values[idx[ok]]
        index = timestamps.index if isinstance(timestamps, pd.Series) else None
        return pd.Series(pd.DatetimeIndex(out), index=index, name="session")

    def decision_time(self, sessions) -> pd.DatetimeIndex:
        """UTC close (decision point) for each session date."""
        pos = self._pos.reindex(pd.DatetimeIndex(sessions).normalize()).values
        if np.isnan(pos).any():
            raise KeyError("some dates are not trading sessions")
        return self.closes_utc[pos.astype(int)]

    def shift(self, sessions, n: int) -> pd.DatetimeIndex:
        """Move each session n trading days forward (n>0) or back (n<0)."""
        pos = self._pos.reindex(pd.DatetimeIndex(sessions).normalize()).values.astype(int) + n
        if (pos < 0).any() or (pos >= len(self.sessions)).any():
            raise IndexError("shift runs past the calendar range")
        return self.sessions[pos]

    def between(self, start, end) -> pd.DatetimeIndex:
        m = (self.sessions >= pd.Timestamp(start)) & (self.sessions <= pd.Timestamp(end))
        return self.sessions[m]

    def position(self, sessions) -> np.ndarray:
        return self._pos.reindex(pd.DatetimeIndex(sessions).normalize()).values


def localise_publication_times(raw: pd.Series, source_tz: str = "UTC",
                               date_only_mask: pd.Series | None = None,
                               market_tz: str = "America/New_York") -> pd.Series:
    """Parse raw publication strings into tz-aware UTC timestamps.

    Items that carry a date but no time of day cannot be placed relative to the
    close, so they are pushed to 23:59 New York time on that date. That sends
    them to the next session, which can only delay information, never leak it.
    """
    ts = pd.to_datetime(raw, errors="coerce", utc=False, format="mixed")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
    ts = ts.dt.tz_convert("UTC")
    if date_only_mask is not None and date_only_mask.any():
        # the stated calendar date is the one written in the source, so read it
        # in the source timezone before moving to the end of that day in New York
        stated = ts[date_only_mask].dt.tz_convert(source_tz).dt.tz_localize(None).dt.normalize()
        late = (stated + pd.Timedelta(hours=23, minutes=59)).dt.tz_localize(
            market_tz, ambiguous="NaT", nonexistent="shift_forward")
        ts.loc[date_only_mask] = late.dt.tz_convert("UTC")
    return ts
