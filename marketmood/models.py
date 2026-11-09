"""Model layer: baselines and learners behind one interface.

The price-only and sentiment-augmented runs differ only in the list of feature
columns passed in. Learner, tuning grid and tuning budget are identical.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ baselines
class MajorityClass(BaseEstimator, ClassifierMixin):
    """Always predicts the training-set majority; probability = training base rate."""

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p_), np.full(n, self.p_)])


class Persistence(BaseEstimator, ClassifierMixin):
    """Random-walk style baseline: tomorrow's sign equals today's sign."""

    def __init__(self, column: str = "ret_1d"):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        up = (X[self.column].values > 0).astype(float)
        p = 0.45 + 0.1 * up       # weak, so log-loss stays finite
        return np.column_stack([1 - p, p])


class LastVol(BaseEstimator, RegressorMixin):
    """Random-walk volatility: next week's log vol equals last week's."""

    def __init__(self, column: str = "har_weekly"):
        self.column = column

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.column].values


# ------------------------------------------------------------------ learners
def _lgbm(task: str, **params):
    import lightgbm as lgb

    base = dict(verbose=-1, n_jobs=2, subsample_freq=1)
    base.update(params)
    return lgb.LGBMClassifier(**base) if task == "clf" else lgb.LGBMRegressor(**base)


def make_estimator(learner: str, task: str, params: dict, seed: int = 42):
    """Every learner is wrapped in a Pipeline so preprocessing is fitted per fold."""
    if learner == "logistic":
        est = LogisticRegression(C=params["C"], max_iter=2000)
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("model", est)])
    if learner in ("har_ols", "ols"):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("model", LinearRegression())])
    if learner == "lightgbm":
        return Pipeline([("model", _lgbm(task, random_state=seed, **params))])
    if learner == "majority":
        return MajorityClass()
    if learner == "persistence":
        return Persistence()
    if learner == "last_vol":
        return LastVol()
    raise ValueError(learner)


def param_grid(cfg_models: dict, learner: str, budget: int, seed: int) -> list[dict]:
    """A fixed list of configurations, identical for every feature set."""
    if learner not in cfg_models:
        return [{}]
    spec = cfg_models[learner]
    keys = list(spec)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(spec[k] for k in keys))]
    if len(combos) > budget:
        combos = random.Random(seed).sample(combos, budget)
    return combos


@dataclass
class FitResult:
    estimator: object
    params: dict
    val_scores: list = field(default_factory=list)
    threshold: float = 0.5


def tune_and_fit(learner: str, task: str, grid: list[dict],
                 X_fit, y_fit, X_val, y_val, X_full, y_full, seed: int = 42) -> FitResult:
    """Pick the configuration with the best validation loss, then refit on the full window."""
    scores = []
    for params in grid:
        est = make_estimator(learner, task, params, seed)
        est.fit(X_fit, y_fit)
        if task == "clf":
            p = np.clip(est.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
            scores.append(log_loss(y_val, p, labels=[0, 1]))
        else:
            scores.append(mean_squared_error(y_val, est.predict(X_val)))
    best = grid[int(np.argmin(scores))]
    final = make_estimator(learner, task, best, seed)
    final.fit(X_full, y_full)
    # decision threshold = training base rate, so a model cannot look good on
    # balanced accuracy just by calling "up" every day during a bull market
    thr = float(np.mean(y_full)) if task == "clf" else 0.0
    return FitResult(final, best, scores, thr)


def fresh(est):
    return clone(est)
