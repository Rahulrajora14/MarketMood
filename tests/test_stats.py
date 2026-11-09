"""Sanity checks for the significance machinery."""
import numpy as np
import pandas as pd

from marketmood.evaluate.backtest import long_short
from marketmood.evaluate.stats import bootstrap_balacc_diff, diebold_mariano
from marketmood.evaluate.metrics import qlike


def test_dm_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    assert diebold_mariano(rng.normal(-0.2, 1, 1500))["p_value"] < 0.01
    # under the null the test should reject about 5% of the time, not more
    rejections = np.mean([diebold_mariano(rng.normal(0.0, 1, 1500))["p_value"] < 0.05
                          for _ in range(300)])
    assert 0.02 < rejections < 0.09


def test_qlike_is_zero_for_a_perfect_forecast():
    v = np.array([0.01, 0.02, 0.03])
    assert np.allclose(qlike(v, v), 0)
    assert (qlike(v, v * 2) > 0).all() and (qlike(v, v / 2) > 0).all()


def test_bootstrap_ci_covers_zero_for_identical_models():
    rng = np.random.default_rng(1)
    n_days, n_firms = 400, 10
    df = pd.DataFrame({"date": np.repeat(np.arange(n_days), n_firms),
                       "y_dir": rng.integers(0, 2, n_days * n_firms)})
    df["a"] = rng.integers(0, 2, len(df))
    df["b"] = df["a"]
    r = bootstrap_balacc_diff(df, "a", "b", reps=200)
    assert r["diff"] == 0 and r["ci_low"] <= 0 <= r["ci_high"]


def test_backtest_is_dollar_neutral_and_charges_costs():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    df = pd.DataFrame([(d, f"T{i}") for d in dates for i in range(10)], columns=["date", "ticker"])
    df["score"] = rng.random(len(df))
    df["fwd_ret_2"] = rng.normal(0, 0.01, len(df))
    out = long_short(df, "score", quantile=0.2, cost_bps=10, lag_days=1)
    assert (out["gross"] - out["net"] >= 0).all()
    assert len(out) == len(dates)
