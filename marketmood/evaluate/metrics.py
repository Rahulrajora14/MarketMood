"""Point metrics for both targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def direction_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p > thr).astype(int)
    out = {
        "bal_acc": balanced_accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "share_up_pred": float(yhat.mean()),
    }
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 and np.std(p) > 0 else np.nan
    return out


def qlike(actual_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE loss (Patton, 2011). Robust to noise in the volatility proxy."""
    r = actual_var / forecast_var
    return r - np.log(r) - 1.0


def vol_metrics(y_logvol: np.ndarray, pred_logvol: np.ndarray) -> dict:
    a = np.exp(2 * y_logvol)
    f = np.exp(2 * pred_logvol)
    return {
        "rmse_logvol": float(np.sqrt(np.mean((y_logvol - pred_logvol) ** 2))),
        "qlike": float(np.mean(qlike(a, f))),
        "r2_logvol": float(1 - np.sum((y_logvol - pred_logvol) ** 2)
                           / np.sum((y_logvol - y_logvol.mean()) ** 2)),
    }


def summarise_folds(per_fold: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and standard deviation across walk-forward folds."""
    agg = per_fold.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
