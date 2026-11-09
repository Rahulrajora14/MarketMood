"""Daily news-based US Economic Policy Uncertainty index (Baker, Bloom and Davis)."""
from __future__ import annotations

import io

import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
PU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"


def fetch_epu(series: str = "USEPUINDXD") -> pd.DataFrame:
    """Return columns [date, epu]. FRED first, policyuncertainty.com as fallback."""
    try:
        r = requests.get(FRED_URL.format(series=series), timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "epu"]
    except Exception:
        r = requests.get(PU_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.rename(columns={"daily_policy_index": "epu"})[["date", "epu"]]
    df["date"] = pd.to_datetime(df["date"])
    df["epu"] = pd.to_numeric(df["epu"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)
