"""Fit-on-train-only preprocessing and the shuffled-label honesty check."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from marketmood.models import make_estimator, param_grid, tune_and_fit


def _signal_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    logit = 1.5 * X.a - 1.0 * X.b
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_scaler_is_fitted_on_training_rows_only():
    X, y = _signal_data()
    X.loc[3000:, "a"] += 10           # test rows live on a different scale
    est = make_estimator("logistic", "clf", {"C": 1.0})
    est.fit(X.iloc[:3000], y[:3000])
    mu = est.named_steps["scale"].mean_[0]
    assert abs(mu - X.a.iloc[:3000].mean()) < 1e-9
    assert abs(mu - X.a.mean()) > 1


def test_each_fold_gets_a_fresh_estimator():
    X, y = _signal_data()
    r1 = tune_and_fit("logistic", "clf", [{"C": 1.0}], X[:2000], y[:2000], X[2000:2500],
                      y[2000:2500], X[:2500], y[:2500])
    r2 = tune_and_fit("logistic", "clf", [{"C": 1.0}], X[:4000], y[:4000], X[4000:4500],
                      y[4000:4500], X[:4500], y[:4500])
    assert r1.estimator is not r2.estimator
    s1, s2 = r1.estimator.named_steps["scale"], r2.estimator.named_steps["scale"]
    assert s1 is not s2 and s1.n_samples_seen_ != s2.n_samples_seen_


def test_same_grid_for_every_feature_set():
    cfg = {"lightgbm": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [7, 15]}}
    g1 = param_grid(cfg, "lightgbm", budget=5, seed=42)
    g2 = param_grid(cfg, "lightgbm", budget=5, seed=42)
    assert g1 == g2 and len(g1) == 5


def test_shuffled_labels_destroy_performance():
    """With real signal the model works; with permuted labels it must fall to chance."""
    X, y = _signal_data(8000)
    tr, te = slice(0, 6000), slice(6000, 8000)
    grid = [{"C": 1.0}]

    real = tune_and_fit("logistic", "clf", grid, X[:5000], y[:5000], X[5000:6000], y[5000:6000],
                        X[tr], y[tr])
    p = real.estimator.predict_proba(X[te])[:, 1]
    ba_real = balanced_accuracy_score(y[te], (p > real.threshold).astype(int))

    rng = np.random.default_rng(7)
    ys = rng.permutation(y[tr])
    fake = tune_and_fit("lightgbm", "clf", [{"n_estimators": 200, "learning_rate": 0.05,
                                             "num_leaves": 15}],
                        X[:5000], ys[:5000], X[5000:6000], ys[5000:6000], X[tr], ys)
    p = fake.estimator.predict_proba(X[te])[:, 1]
    ba_fake = balanced_accuracy_score(y[te], (p > fake.threshold).astype(int))

    assert ba_real > 0.70
    assert abs(ba_fake - 0.5) < 0.04
