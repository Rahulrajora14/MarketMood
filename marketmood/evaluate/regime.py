"""Policy-uncertainty regimes and the sentiment x EPU interaction test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .metrics import direction_metrics, vol_metrics

LABELS = ["low", "mid", "high"]


def epu_terciles(train_epu: pd.Series, test_epu: pd.Series) -> pd.Series:
    """Cut points come from the training window only, then applied to test days."""
    lo, hi = np.nanpercentile(train_epu, [100 / 3, 200 / 3])
    return pd.cut(test_epu, [-np.inf, lo, hi, np.inf], labels=LABELS)


def regime_table(preds: pd.DataFrame, models: list[str], task: str) -> pd.DataFrame:
    """Metrics recomputed inside each EPU tercile, per fold, then averaged."""
    rows = []
    for (fold, reg), d in preds.groupby(["fold", "epu_regime"], observed=True):
        for m in models:
            if task == "clf":
                met = direction_metrics(d["y_dir"].values, d[f"p_{m}"].values, d[f"thr_{m}"].iloc[0])
            else:
                met = vol_metrics(d["y_logvol"].values, d[f"v_{m}"].values)
            rows.append({"fold": fold, "regime": reg, "model": m, "n": len(d), **met})
    t = pd.DataFrame(rows)
    keep = ["bal_acc", "macro_f1", "auc"] if task == "clf" else ["rmse_logvol", "qlike", "r2_logvol"]
    out = t.groupby(["regime", "model"], observed=True)[keep].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def interaction_test(panel: pd.DataFrame, y: str, sent: str, controls: list[str],
                     task: str) -> pd.DataFrame:
    """y ~ controls + sent + z(log EPU) + sent * z(log EPU), SEs clustered by date.

    A significant interaction coefficient means the link between sentiment
    and the target changes with the policy-uncertainty level.
    """
    d = panel[[y, sent, "log_epu", "date"] + controls].dropna().copy()
    d["epu_z"] = (d["log_epu"] - d["log_epu"].mean()) / d["log_epu"].std()
    d["sent_z"] = (d[sent] - d[sent].mean()) / d[sent].std()
    d["sent_x_epu"] = d["sent_z"] * d["epu_z"]
    X = sm.add_constant(d[controls + ["sent_z", "epu_z", "sent_x_epu"]])
    groups = pd.factorize(d["date"])[0]
    if task == "clf":
        res = sm.Logit(d[y], X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    tab = pd.DataFrame({"coef": res.params, "se": res.bse, "z": res.tvalues, "p_value": res.pvalues})
    return tab.loc[["sent_z", "epu_z", "sent_x_epu"]].reset_index(names="term")
