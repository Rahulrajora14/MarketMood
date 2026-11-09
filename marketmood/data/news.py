"""FNSPID financial news (Dong, Fan and Peng, 2024).

Two files are used:
  Stock_news/All_external.csv        headlines from Benzinga and similar wires
  Stock_news/nasdaq_exteral_data.csv Nasdaq.com articles with full bodies

Both are tens of gigabytes in total, so they are read in chunks and only the
study window, the candidate tickers, the headline and the opening of the
article body are kept.

Timestamps
----------
FNSPID's `Date` column cannot be trusted at intraday precision:
  * most rows carry 00:00:00, i.e. a date with no time;
  * rows that do carry a time are not in UTC despite the label. Recurring
    Benzinga stories such as "... Stocks Moving In Friday's Mid-Day Session"
    are stamped around 08:40 "UTC", which is 04:40 in New York.
    (`audit_timestamps` reproduces this check.)

We therefore keep only the stated calendar date and treat every item as
published at the end of that date (23:59 New York). An item dated D can then
inform the first session that closes after D ends, i.e. the next trading day.
This throws away same-day news, which can only weaken the sentiment signal;
it can never let after-close news leak into the same day's forecast.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"


def _lead(text: pd.Series, n_chars: int = 300) -> pd.Series:
    """Opening of the article body, cut at a sentence end when possible."""
    t = text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    t = t.str.slice(0, n_chars)

    def cut(s: str) -> str:
        i = s.rfind(". ")
        return s[: i + 1] if i >= 80 else s

    return t.map(cut)


def end_of_stated_day(stated: pd.Series, market_tz: str = "America/New_York") -> pd.Series:
    """23:59 New York time on the stated calendar date, returned in UTC."""
    day = pd.to_datetime(stated.str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
    late = (day + pd.Timedelta(hours=23, minutes=59)).dt.tz_localize(
        market_tz, ambiguous="NaT", nonexistent="shift_forward")
    return late.dt.tz_convert("UTC")


def read_fnspid(source: str | Path, start: str, end: str, tickers: set[str] | None,
                out_dir: Path, tag: str, chunksize: int = 250_000) -> list[Path]:
    """Filter one FNSPID csv (local path or URL) into Parquet parts under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if str(source).startswith("http"):
        # pandas buffers a URL completely before parsing; stream it instead
        import io
        import urllib.request
        resp = urllib.request.urlopen(str(source), timeout=120)
        handle = io.TextIOWrapper(io.BufferedReader(resp, buffer_size=1 << 20),
                                  encoding="utf-8", errors="replace")
    else:
        handle = source
    reader = pd.read_csv(handle, chunksize=chunksize, dtype=str, on_bad_lines="skip",
                         encoding_errors="replace")
    parts, seen, kept = [], 0, 0
    for i, chunk in enumerate(reader):
        seen += len(chunk)
        cols = {c.lower(): c for c in chunk.columns}
        body = cols.get("lsa_summary") if tag == "nasdaq" else None
        body = body or cols.get("article")
        df = pd.DataFrame({
            "stated": chunk[cols["date"]].fillna(""),
            "ticker": chunk[cols["stock_symbol"]].fillna("").str.upper().str.strip(),
            "title": chunk[cols["article_title"]].fillna("").str.strip(),
            "lead": _lead(chunk[body]) if body else "",
            "publisher": chunk[cols["publisher"]] if "publisher" in cols else None,
            "url": chunk[cols["url"]] if "url" in cols else None,
        })
        df = df[(df.ticker != "") & (df.title != "")]
        if tickers is not None:
            df = df[df.ticker.isin(tickers)]
        day = pd.to_datetime(df.stated.str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
        df = df[(day >= lo) & (day <= hi)]
        if df.empty:
            continue
        df["published_utc"] = end_of_stated_day(df["stated"])
        df["time_stated"] = ~df.stated.str.contains(" 00:00:00")
        df["source"] = tag
        p = out_dir / f"{tag}_{i:05d}.parquet"
        df.to_parquet(p, index=False)
        parts.append(p)
        kept += len(df)
        if i % 10 == 0:
            log.info("%s: read %s rows, kept %s", tag, f"{seen:,}", f"{kept:,}")
    log.info("%s finished: read %s rows, kept %s", tag, f"{seen:,}", f"{kept:,}")
    return parts


def combine_parts(part_dir: Path, roundup_min_tickers: int = 5) -> tuple[pd.DataFrame, dict]:
    """Merge the filtered parts, remove duplicates and multi-firm round-ups.

    A headline that appears under five or more tickers on the same day
    ("Stocks That Hit 52-Week Highs On Friday", "10 Biggest Price Target
    Changes ...") says nothing specific about any one firm, so it is dropped.
    """
    cols = ["ticker", "stated", "title", "lead", "publisher", "url", "published_utc",
            "time_stated", "source"]
    files = sorted(part_dir.glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat([f[[c for c in cols if c in f]] for f in frames if len(f)], ignore_index=True)
    stats = {"rows_read": len(df)}
    df["day"] = df["stated"].str.slice(0, 10)
    df = df.drop_duplicates(subset=["ticker", "day", "title"])
    stats["after_dedup"] = len(df)
    spread = df.groupby(["day", "title"])["ticker"].transform("nunique")
    df = df[spread < roundup_min_tickers]
    stats["after_roundup_filter"] = len(df)
    df = df.drop(columns="day")
    return df.sort_values(["ticker", "published_utc"]).reset_index(drop=True), stats


# ------------------------------------------------------------------ audit
AUDIT_PATTERNS = {
    "Pre-Market Session": (4.0, 9.5),     # published before the 09:30 open
    "Mid-Day Session": (11.0, 14.0),      # published around midday
    "After-Hours Session": (16.0, 20.0),  # published after the 16:00 close
    "52-Week Highs": (9.5, 16.0),         # published during the session
    "52-Week Lows": (9.5, 16.0),
}


def audit_timestamps(sample: pd.DataFrame) -> pd.DataFrame:
    """Compare stated clock times of recurring stories with when they must run.

    `sample` needs columns `stated` (raw FNSPID Date string) and `title`.
    For each story type the table shows the median stated hour if read as UTC,
    converted to New York time, against the plausible New York window.
    """
    s = sample[~sample.stated.str.contains(" 00:00:00")].copy()
    ts = pd.to_datetime(s.stated.str.replace(" UTC", "", regex=False), errors="coerce")
    s["utc_hour"] = ts.dt.hour + ts.dt.minute / 60
    ny = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    s["ny_hour_if_utc"] = ny.dt.hour + ny.dt.minute / 60
    rows = []
    for pat, (lo, hi) in AUDIT_PATTERNS.items():
        d = s[s.title.str.contains(pat, case=False, regex=False)]
        if len(d) < 20:
            continue
        rows.append({
            "story": pat, "n": len(d),
            "stated_hour_median": d.utc_hour.median(),
            "ny_hour_if_utc_median": d.ny_hour_if_utc.median(),
            "expected_ny_window": f"{lo:.1f}-{hi:.1f}",
            "share_inside_window_if_utc": float(d.ny_hour_if_utc.between(lo, hi).mean()),
            "share_inside_window_if_stated_is_ny_minus_4h": float(((d.utc_hour + 4) % 24).between(lo, hi).mean()),
        })
    rows.append({"story": "all rows", "n": len(sample),
                 "share_date_only": float(sample.stated.str.contains(" 00:00:00").mean())})
    return pd.DataFrame(rows)
