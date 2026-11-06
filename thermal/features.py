"""Per-pixel feature matrix for the background model, from one cutout. numpy only.

The same function serves training (with the target) and inference (target optional),
so the two can never disagree about what a feature is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import config, geo

BT_CHANNELS = ("WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134")


def brightness_temperature(radiance: np.ndarray, platform: str, channel: str) -> np.ndarray:
    cal = config.load_yaml("calibration")
    c1, c2 = cal["constants"]["C1"], cal["constants"]["C2"]
    plat = cal["platforms"].get(platform) or cal["platforms"]["MSG3"]
    k = plat[channel]
    nu = k["nu_c"]
    r = np.asarray(radiance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (c2 * nu / np.log1p(c1 * nu**3 / r) - k["B"]) / k["A"]
    return np.where(r > 0, t, np.nan)


def platform_of(ds: xr.Dataset) -> str:
    name = str(ds.attrs.get("EPCT_product_name", ""))
    return name.split("-", 1)[0] if name else "unknown"


@dataclass
class FeatureSet:
    names: list[str]
    X: np.ndarray  # (n_valid, n_features)
    y: np.ndarray | None  # (n_valid,) target radiance, None at inference without the MIR channel
    valid: np.ndarray  # (rows, cols) boolean mask of pixels included
    lat: np.ndarray
    lon: np.ndarray
    platform: str
    time: datetime
    shape: tuple[int, int]

    def to_image(self, values: np.ndarray, fill=np.nan) -> np.ndarray:
        """Scatter a per-valid-pixel vector back onto the cutout grid."""
        img = np.full(self.shape, fill, dtype=float)
        img[self.valid] = values
        return img


def feature_names(sat: dict[str, Any]) -> list[str]:
    channels = [c for c in sat["channels"] if c != sat["mir_channel"]]
    names = [f"rad_{c}" for c in channels]
    names += [f"bt_{c}" for c in BT_CHANNELS if c in channels]
    names += [
        "sza",
        "cos_sza",
        "cos_rel_az",
        "sin_rel_az",
        "vza",
        "lat",
        "lon",
        "sin_doy",
        "cos_doy",
    ]
    return names


def build_features(
    nc_path: Path, t: datetime, sat: dict[str, Any], require_target: bool = True
) -> FeatureSet:
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        ds.load()
        platform = platform_of(ds)
        lat, lon, _ = geo.geolocate(ds)
        inv = {v: k for k, v in sat["variable_map"].items()}  # channel -> variable name
        rad: dict[str, np.ndarray] = {}
        for ch in sat["channels"]:
            var = inv.get(ch)
            if var in ds.variables:
                rad[ch] = np.asarray(ds[var].values, dtype=float)
    channels = [c for c in sat["channels"] if c != sat["mir_channel"]]
    missing = [c for c in channels if c not in rad]
    if missing:
        raise ValueError(f"cutout lacks channels {missing}")
    mir = rad.get(sat["mir_channel"])
    if require_target and mir is None:
        raise ValueError(f"cutout lacks target channel {sat['mir_channel']}")

    valid = np.isfinite(lat) & np.isfinite(lon)
    for c in channels:
        valid &= np.isfinite(rad[c])
    if mir is not None:
        valid &= np.isfinite(mir)

    sza, saz = geo.solar_position(t, lat, lon)
    vza, vaz = geo.satellite_angles(lat, lon, sat["sub_lon"])
    rel = np.radians(saz - vaz)
    doy = 2 * np.pi * (t.timetuple().tm_yday - 1) / 365.25

    cols: list[np.ndarray] = [rad[c] for c in channels]
    cols += [brightness_temperature(rad[c], platform, c) for c in BT_CHANNELS if c in channels]
    cols += [
        sza,
        np.cos(np.radians(sza)),
        np.cos(rel),
        np.sin(rel),
        vza,
        lat,
        lon,
        np.full(lat.shape, np.sin(doy)),
        np.full(lat.shape, np.cos(doy)),
    ]
    for c in cols:
        valid &= np.isfinite(c)
    X = np.stack([c[valid] for c in cols], axis=1).astype(np.float32)
    y = mir[valid].astype(np.float32) if mir is not None else None
    return FeatureSet(
        names=feature_names(sat),
        X=X,
        y=y,
        valid=valid,
        lat=lat,
        lon=lon,
        platform=platform,
        time=t,
        shape=lat.shape,
    )
