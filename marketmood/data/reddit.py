"""Retail discussion from public Reddit archives (Arctic Shift API).

Only submission titles are used. A post is attributed to a ticker when the
ticker appears in the title as an upper-case word, or as a cashtag for symbols
of one or two letters (where the bare word is too ambiguous: T, V, MA, ...).
"""
from __future__ import annotations

import logging
import re
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://arctic-shift.photon-reddit.com/api/posts/search"


def _pattern(ticker: str) -> re.Pattern:
    t = re.escape(ticker)
    if len(ticker) <= 2:
        return re.compile(rf"\${t}\b")
    return re.compile(rf"(?<![A-Za-z]){t}(?![A-Za-z])")


def fetch_subreddit_ticker(subreddit: str, ticker: str, start: str, end: str,
                           max_pages: int = 60, pause: float = 1.0) -> pd.DataFrame:
    rows, after = [], pd.Timestamp(start).strftime("%Y-%m-%d")
    pat = _pattern(ticker)
    for _ in range(max_pages):
        params = {"subreddit": subreddit, "title": ticker, "after": after,
                  "before": pd.Timestamp(end).strftime("%Y-%m-%d"),
                  "limit": 100, "sort": "asc",
                  "fields": "id,title,created_utc,score,num_comments"}
        data = None
        for attempt in range(4):
            try:
                r = requests.get(API, params=params, timeout=90)
                payload = r.json()
                data = payload.get("data")
                if data is not None:
                    break
                # the archive answers "Timeout. Maybe slow down a bit" when busy
            except Exception as exc:
                log.debug("reddit %s/%s: %s", subreddit, ticker, exc)
            time.sleep(5 * 2 ** attempt)
        if data is None:
            log.warning("reddit %s/%s gave up after retries", subreddit, ticker)
            break
        if not data:
            break
        for d in data:
            if pat.search(d.get("title", "")):
                rows.append({**d, "ticker": ticker, "subreddit": subreddit})
        after = pd.Timestamp(int(data[-1]["created_utc"]) + 1, unit="s").strftime("%Y-%m-%dT%H:%M:%S")
        if len(data) < 100:
            break
        time.sleep(pause)
    return pd.DataFrame(rows)


def fetch_reddit(tickers: list[str], subreddits: list[str], start: str, end: str,
                 max_pages: int = 20, workers: int = 2) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(s, t) for t in tickers for s in subreddits]
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda j: fetch_subreddit_ticker(j[0], j[1], start, end, max_pages), jobs))
    frames = [r for r in results if not r.empty]
    if not frames:
        return pd.DataFrame(columns=["ticker", "title", "published_utc", "score", "subreddit"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    out["published_utc"] = pd.to_datetime(out["created_utc"].astype(int), unit="s", utc=True)
    out["title"] = out["title"].astype(str).str.strip()
    return out[["ticker", "title", "published_utc", "score", "num_comments", "subreddit"]]
