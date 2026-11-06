"""Per-timestamp anomaly products: arrays for the private bucket, PNGs for the public one."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import render
from .features import FeatureSet
from .storage import Storage, time_stamp

# Fixed colour scales so that images are comparable between timestamps (radiance units).
PNG_SCALES: dict[str, tuple[float | None, float | None]] = {
    "ir39": (0.0, 0.6),
    "ir39_pred": (0.0, 0.6),
    "residual": (-0.05, 0.3),
    "zscore": (-5.0, 20.0),
    "ir108": (20.0, 110.0),
    "vis06": (0.0, 20.0),
    "clm": (0.0, 2.0),
}


def product_prefix(sat: str, t: datetime) -> str:
    return f"products/{sat}/{t:%Y/%m/%d}/{time_stamp(t)}"


def saturation_radiance(ds: xr.Dataset, variable: str) -> float | None:
    """Radiance of the maximum 10-bit count for a channel, from the chNN_cal attribute."""
    num = variable.split("_")[-1]
    attr = ds.attrs.get(f"ch{int(num):02d}_cal")
    if not attr:
        return None
    offset, slope = (float(v) for v in str(attr).split())
    return offset + slope * 1023.0


def build_products(
    images: dict[str, np.ndarray],
    fs: FeatureSet,
    l15_path: Path,
    clm_path: Path | None,
    sat: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return (arrays, extras): arrays for residual.npz, extras with saturation/cloud info."""
    inv = {v: k for k, v in sat["variable_map"].items()}
    with xr.open_dataset(l15_path, decode_times=False) as ds:
        mir_var = inv[sat["mir_channel"]]
        sat_rad = saturation_radiance(ds, mir_var)
        ir108 = np.asarray(ds[inv["IR_108"]].values, dtype=float) if "IR_108" in inv else None
        vis06 = np.asarray(ds[inv["VIS006"]].values, dtype=float) if "VIS006" in inv else None
    cloud = None
    if clm_path is not None and clm_path.exists():
        with xr.open_dataset(clm_path, decode_times=False) as ds:
            cloud = np.asarray(ds[sat.get("cloudmask_variable", "cloud_mask")].values, dtype=float)
    observed = images["observed"]
    saturated = (
        np.isfinite(observed) & (observed >= 0.995 * sat_rad)
        if sat_rad
        else np.zeros(observed.shape, bool)
    )
    flags = np.zeros(observed.shape, dtype=np.uint8)
    flags[~fs.valid] |= 1  # invalid / outside box
    flags[saturated] |= 2
    if cloud is not None:
        flags[cloud == 2] |= 4
        flags[~np.isfinite(cloud)] |= 8  # cloud mask unavailable
    else:
        flags[fs.valid] |= 8  # no cloud mask cutout at all
    arrays = {
        "residual": images["residual"].astype(np.float32),
        "zscore": images["zscore"].astype(np.float32),
        "pred_std": images["pred_std"].astype(np.float32),
        "flags": flags,
        "lat": fs.lat.astype(np.float32),
        "lon": fs.lon.astype(np.float32),
    }
    extras = {
        "saturation_radiance": sat_rad,
        "saturated": saturated,
        "cloud": cloud,
        "ir108": ir108,
        "vis06": vis06,
    }
    return arrays, extras


def _north_up(img: np.ndarray, fs: FeatureSet) -> np.ndarray:
    return img[::-1] if np.nanmean(fs.lat[0]) < np.nanmean(fs.lat[-1]) else img


def write_products(
    storage: Storage | None,
    sat: dict[str, Any],
    t: datetime,
    images: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    extras: dict[str, Any],
    fs: FeatureSet,
    out_dir: Path,
    model_id: str,
) -> dict[str, Any]:
    """Write residual.npz (private) and the PNG set (public). Returns the keys written."""
    prefix = product_prefix(sat["key"], t)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "residual.npz"
    np.savez_compressed(
        npz_path,
        **arrays,
        meta=np.array(f'{{"sat":"{sat["key"]}","time":"{t.isoformat()}","model":"{model_id}"}}'),
    )
    written: dict[str, Any] = {"npz": f"{prefix}/residual.npz", "png": {}}
    if storage is not None:
        storage.private.put_file(written["npz"], npz_path)
    layers = {
        "ir39": images["observed"],
        "ir39_pred": images["predicted"],
        "residual": images["residual"],
        "zscore": images["zscore"],
        "ir108": extras.get("ir108"),
        "vis06": extras.get("vis06"),
        "clm": extras.get("cloud"),
    }
    for name, img in layers.items():
        if img is None:
            continue
        if name == "vis06" and np.nanmean(fs.X[:, fs.names.index("sza")]) > 95:
            continue  # night: nothing to show
        vmin, vmax = PNG_SCALES[name]
        path = out_dir / f"{name}.png"
        render.to_png(_north_up(img, fs), path, vmin=vmin, vmax=vmax)
        key = f"{prefix}/{name}.png"
        written["png"][name] = key
        if storage is not None:
            storage.public.put_file(key, path, content_type="image/png")
    return written


def load_residual(storage: Storage, sat: str, t: datetime) -> dict[str, np.ndarray]:
    data = storage.private.get_bytes(f"{product_prefix(sat, t)}/residual.npz")
    with np.load(io.BytesIO(data)) as npz:
        return {k: npz[k] for k in npz.files}
