"""Metadata summaries of cutout files. Never includes the data arrays themselves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import geo


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 16 else f"<array shape={v.shape}>"
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def summarise(path: Path) -> dict[str, Any]:
    """Dimensions, variables, attributes and (for 2-D data) coarse value ranges."""
    with xr.open_dataset(path, decode_times=False) as ds:
        out: dict[str, Any] = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "attrs": {k: _jsonable(v) for k, v in ds.attrs.items()},
            "variables": {},
        }
        for name, var in ds.variables.items():
            entry: dict[str, Any] = {
                "dims": list(var.dims),
                "dtype": str(var.dtype),
                "attrs": {k: _jsonable(v) for k, v in var.attrs.items()},
            }
            if var.ndim == 2 and np.issubdtype(var.dtype, np.number):
                vals = var.values.astype(float)
                finite = np.isfinite(vals)
                entry["finite_fraction"] = float(finite.mean()) if vals.size else 0.0
                if finite.any():
                    entry["min"] = float(np.nanmin(vals))
                    entry["max"] = float(np.nanmax(vals))
            out["variables"][name] = entry
        try:
            lat, lon, method = geo.geolocate(ds)
            out["geolocation"] = {
                "method": method,
                "shape": list(lat.shape),
                "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
                "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
                "finite_fraction": float(np.isfinite(lat).mean()),
                "median_pixel_area_km2": float(np.nanmedian(geo.pixel_area_m2(lat, lon)) / 1e6),
            }
        except Exception as exc:
            out["geolocation"] = {"error": str(exc)}
    return out
