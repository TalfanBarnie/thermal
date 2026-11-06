"""Residual (anomaly) products and monitoring-region statistics from a model prediction."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config, geo, regions
from .features import FeatureSet
from .model import BackgroundModel


def predict_images(model: BackgroundModel, fs: FeatureSet) -> dict[str, np.ndarray]:
    """Observed, predicted, residual, z-score and predicted-std images on the cutout grid."""
    mean, std = model.predict(fs.X, fs.names)
    obs = fs.to_image(fs.y) if fs.y is not None else None
    pred = fs.to_image(mean)
    out = {"predicted": pred, "pred_std": fs.to_image(std)}
    if obs is not None:
        out["observed"] = obs
        out["residual"] = obs - pred
        out["zscore"] = fs.to_image((fs.y - mean) / std)
    return out


def region_stats(
    images: dict[str, np.ndarray],
    fs: FeatureSet,
    region: str,
    z_threshold: float = 4.0,
    cloud_mask: np.ndarray | None = None,
    sat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summary statistics of the residual over one monitoring region."""
    mask = regions.region_mask(region, fs.lat, fs.lon) & fs.valid
    resid = images["residual"]
    z = images["zscore"]
    area = geo.pixel_area_m2(fs.lat, fs.lon)
    r = resid[mask]
    zz = z[mask]
    pos = np.clip(r, 0, None)
    stats: dict[str, Any] = {
        "region": region,
        "n_valid": int(mask.sum()),
        "sum_pos": float(np.nansum(pos)),
        "max": float(np.nanmax(r)) if r.size else float("nan"),
        "max_z": float(np.nanmax(zz)) if zz.size else float("nan"),
        "n_above": int(np.nansum(zz > z_threshold)),
        "mean_resid": float(np.nanmean(r)) if r.size else float("nan"),
        "sza_mean": float(np.nanmean(fs.X[:, fs.names.index("sza")])),
        "sum_pos_area_weighted_m2": float(np.nansum(pos * area[mask])),
    }
    if cloud_mask is not None:
        cm = cloud_mask[mask]
        stats["n_cloud"] = int(np.nansum(cm == 2))
        stats["n_clear"] = int(np.nansum((cm == 0) | (cm == 1)))
        stats["n_cloud_unknown"] = int(np.sum(~np.isfinite(cm)))
    return stats


def default_regions(sat: dict[str, Any]) -> list[str]:
    return list(sat.get("monitoring_regions") or config.monitoring_regions().keys())
