"""Targets, feature timing, EPU lag and the walk-forward splitter."""
import numpy as np
import pandas as pd

from marketmood import panel as P
from marketmood.features import aggregate as A
from marketmood.features import price as PF


def test_direction_target_is_next_day_sign(prices, cal):
    t = P.make_targets(prices, cal)
    a = prices[prices.ticker == "AAA"].sort_values("date").reset_index(drop=True)
    ta = t[t.ticker == "AAA"].sort_values("date").reset_index(drop=True)
    r_next = np.log(a.close.shift(-1) / a.close)
    ok = r_next.notna()
    assert ((r_next[ok] > 0).astype(float) == ta.y_dir[ok]).all()
    assert ta.y_dir.iloc[-1] != ta.y_dir.iloc[-1]  # NaN on the last row


def test_vol_target_uses_next_five_days_only(prices, cal):
    t = P.make_targets(prices, cal, vol_horizon=5)
    a = prices[prices.ticker == "BBB"].sort_values("date").reset_index(drop=True)
    r = np.log(a.close).diff()
    i = 100
    expected = 0.5 * np.log(np.mean(r.iloc[i + 1:i + 6] ** 2) + 1e-8)
    got = t[t.ticker == "BBB"].sort_values("date").y_logvol.iloc[i]
    assert np.isclose(expected, got)


def test_price_features_do_not_peek(prices):
    """Changing tomorrow's price must not change any feature dated today."""
    base = PF.price_features(prices)
    bumped = prices.copy()
    day = bumped.date.sort_values().unique()[300]
    m = bumped.date > day
    bumped.loc[m, ["close", "high", "low", "open"]] *= 1.5
    bumped.loc[m, "volume"] *= 3
    after = PF.price_features(bumped)
    cols = PF.price_feature_names()
    a = base[base.date <= day].set_index(["ticker", "date"])[cols]
    b = after[after.date <= day].set_index(["ticker", "date"])[cols]
    pd.testing.assert_frame_equal(a, b)


def test_sentiment_features_use_only_text_before_decision_point(prices, cal, news_items):
    grid = prices[["ticker", "date"]].copy()
    items = A.assign_sessions(news_items, cal)
    agg = A.aggregate_items(items, "news")
    feats = A.sentiment_features(grid, agg, "news")
    has = feats.dropna(subset=["news_text_max_time"])
    decision = cal.decision_time(has["date"])
    assert (has["news_text_max_time"].values < decision.values).all()


def test_epu_is_lagged(prices):
    grid = prices[["ticker", "date"]].drop_duplicates("date").copy()
    epu = pd.DataFrame({"date": pd.date_range("2014-12-01", "2020-01-31"), "epu": 100.0})
    out = P.attach_epu(grid, epu)
    assert (out["epu_date"] < out["date"]).all()


def test_folds_are_ordered_and_embargoed(prices, cal):
    t = P.make_targets(prices, cal, vol_horizon=5)
    folds = P.walk_forward_folds(prices["date"], 2017, 2019, embargo=5)
    assert [f.name for f in folds] == ["2017", "2018", "2019"]
    for f in folds:
        assert f.train_end < f.test_start
        assert not set(f.train_sessions) & set(f.test_sessions)
        gap = cal.position([f.test_start])[0] - cal.position([f.train_end])[0] - 1
        assert gap >= 5
        # no training target is realised after the first test decision point
        tt = t[t.date.isin(f.train_sessions)]["target_time"]
        assert tt.max() < cal.decision_time([f.test_start])[0]
    # expanding window
    assert all(len(a.train_sessions) < len(b.train_sessions) for a, b in zip(folds, folds[1:]))


def test_inner_split_is_ordered_with_gap(prices):
    sessions = pd.DatetimeIndex(np.sort(prices.date.unique()))
    fit, val = P.inner_split(sessions[:500], 0.2, 5)
    assert fit.max() < val.min()
    assert sessions.get_loc(val.min()) - sessions.get_loc(fit.max()) - 1 >= 5
