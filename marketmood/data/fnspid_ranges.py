"""Parallel reader for the large FNSPID Nasdaq file.

The file is ~23 GB of CSV with multi-line article bodies. Reading it as one
stream is limited by a single HTTP connection, so it is split into byte ranges
that are fetched and parsed in parallel. Each piece is cut at row boundaries,
recognised by the row prefix `<index>.0,<YYYY-MM-DD HH:MM:SS> UTC,`.
"""
from __future__ import annotations

import io
import logging
import re
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .news import _lead, end_of_stated_day

log = logging.getLogger(__name__)
ROW_START = re.compile(rb"\n\d+\.0,\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC,")
PAD = 2_000_000


def _get(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as exc:
            log.warning("range %d-%d attempt %d: %s", a, b, attempt + 1, exc)
    raise RuntimeError(f"could not fetch {a}-{b}")


def _piece(url: str, header: list[str], a: int, b: int, size: int, start: str, end: str,
           tickers: set[str], out: Path) -> int:
    if out.exists():
        return -1
    base = a - 1 if a > 0 else 0          # include the newline before a
    buf = _get(url, base, min(b + PAD, size - 1))
    # first full row at or after a (the header piece starts right after the header line)
    if a == 0:
        s = buf.index(b"\n") + 1
    else:
        m = ROW_START.search(buf, 0)
        if m is None:
            pd.DataFrame(columns=["ticker"]).to_parquet(out, index=False)
            return 0
        s = m.start() + 1
    # last row that starts before b belongs to this piece
    stop = len(buf)
    if b + 1 < size:
        m = ROW_START.search(buf, b - base)
        stop = m.start() + 1 if m else len(buf)
    text = buf[s:stop].decode("utf-8", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text), names=header, dtype=str, on_bad_lines="skip")
    except pd.errors.ParserError:
        # an unbalanced quote inside an article body; parse row by row instead
        rows = ROW_START.split(b"\n" + text.encode("utf-8"))
        starts = [m.group(0)[1:] for m in ROW_START.finditer(b"\n" + text.encode("utf-8"))]
        good = []
        for pre, body in zip(starts, rows[1:]):
            try:
                good.append(pd.read_csv(io.BytesIO(pre + body), names=header, dtype=str,
                                        on_bad_lines="skip", nrows=1))
            except Exception:
                continue
        df = pd.concat(good, ignore_index=True) if good else pd.DataFrame(columns=header)
    cols = {c.lower(): c for c in df.columns}
    d = pd.DataFrame({
        "row_id": pd.to_numeric(df[header[0]], errors="coerce"),
        "stated": df[cols["date"]].fillna(""),
        "ticker": df[cols["stock_symbol"]].fillna("").str.upper().str.strip(),
        "title": df[cols["article_title"]].fillna("").str.strip(),
        "lead": _lead(df[cols["lsa_summary"]]),
        "publisher": df[cols["publisher"]],
        "url": df[cols["url"]],
    })
    d = d[(d.ticker != "") & (d.title != "") & d.ticker.isin(tickers)]
    day = pd.to_datetime(d.stated.str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
    d = d[(day >= pd.Timestamp(start)) & (day <= pd.Timestamp(end))]
    d["published_utc"] = end_of_stated_day(d["stated"])
    d["time_stated"] = ~d.stated.str.contains(" 00:00:00")
    d["source"] = "nasdaq"
    d.to_parquet(out, index=False)
    return len(d)


def read_nasdaq_parallel(url: str, start: str, end: str, tickers: set[str], out_dir: Path,
                         piece_mb: int = 120, workers: int = 6) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    head = _get(url, 0, 4096)
    header = pd.read_csv(io.BytesIO(head[: head.index(b"\n") + 1])).columns.tolist()
    step = piece_mb * 1_000_000
    ranges = [(a, min(a + step, size) - 1) for a in range(0, size, step)]
    log.info("nasdaq file: %.1f GB in %d pieces", size / 1e9, len(ranges))
    kept = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_piece, url, header, a, b, size, start, end, tickers,
                          out_dir / f"nasdaq_r{i:05d}.parquet"): i for i, (a, b) in enumerate(ranges)}
        for n, f in enumerate(as_completed(futs), 1):
            kept += max(f.result(), 0)
            if n % 10 == 0:
                log.info("nasdaq: %d / %d pieces, kept %s rows", n, len(ranges), f"{kept:,}")
