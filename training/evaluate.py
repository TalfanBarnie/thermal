"""Held-out evaluation of a background model artefact."""

from __future__ import annotations

from typing import Any

import numpy as np

from thermal import features
from thermal.model import BackgroundModel


def _stats(r: np.ndarray) -> dict[str, float]:
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0}
    return {
        "n": int(r.size),
        "bias": float(r.mean()),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "mad": float(np.median(np.abs(r - np.median(r)))),
        "p01": float(np.percentile(r, 1)),
        "p99": float(np.percentile(r, 99)),
    }


def evaluate(
    model: BackgroundModel,
    X,
    y,
    names,
    groups,
    sza_col: str = "sza",
    platform="MSG3",
    month_of: np.ndarray | None = None,
) -> dict[str, Any]:
    mean, std = model.predict(X, names)
    resid = y - mean
    z = resid / std
    # residual in kelvin via the IR3.9 BT conversion of observed and predicted radiance
    bt_obs = features.brightness_temperature(y, platform, "IR_039")
    bt_pred = features.brightness_temperature(mean, platform, "IR_039")
    resid_k = bt_obs - bt_pred
    sza = X[:, names.index(sza_col)]
    out: dict[str, Any] = {
        "overall": {"radiance": _stats(resid), "kelvin": _stats(resid_k)},
        "zscore": {
            "frac_abs_lt_1": float(np.mean(np.abs(z) < 1)),
            "frac_abs_lt_2": float(np.mean(np.abs(z) < 2)),
            "frac_abs_lt_3": float(np.mean(np.abs(z) < 3)),
            "std": float(np.std(z)),
        },
        "by_sza": {},
        "by_month": {},
    }
    for label, lo, hi in (
        ("day_lt80", -1, 80),
        ("twilight_80_100", 80, 100),
        ("night_gt100", 100, 181),
    ):
        m = (sza >= lo) & (sza < hi)
        out["by_sza"][label] = {"radiance": _stats(resid[m]), "kelvin": _stats(resid_k[m])}
    if month_of is not None:
        for mo in sorted(set(month_of.tolist())):
            m = month_of == mo
            out["by_month"][str(mo)] = {"radiance": _stats(resid[m]), "kelvin": _stats(resid_k[m])}
    return out
