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
                    uniq, counts = np.unique(vals[finite], return_counts=True)
                    if uniq.size <= 16:
                        entry["value_counts"] = {
                            str(u): int(c) for u, c in zip(uniq, counts, strict=True)
                        }
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
            out["footprint"] = footprint_check(ds, lat, lon)
        except Exception as exc:
            out["geolocation"] = {"error": str(exc)}
    return out


def footprint_check(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> dict[str, Any]:
    """Where do the finite data pixels fall in lat/lon, under the grid as given and flipped?

    The Data Tailor masks pixels outside the requested lat/lon box, so the finite
    footprint of a correctly georeferenced cutout spans exactly that box. If a flipped
    orientation matches the box better, the file's georeferencing is inconsistent.
    """
    data_var = next(
        (v for v in ds.data_vars.values() if v.ndim == 2 and np.issubdtype(v.dtype, np.number)),
        None,
    )
    if data_var is None:
        return {}
    finite = np.isfinite(data_var.values.astype(float))
    out: dict[str, Any] = {"variable": data_var.name, "finite_fraction": float(finite.mean())}
    for name, f in (
        ("as_is", finite),
        ("flip_rows", finite[::-1, :]),
        ("flip_cols", finite[:, ::-1]),
        ("flip_both", finite[::-1, ::-1]),
    ):
        if f.any():
            out[name] = {
                "lat_range": [float(np.nanmin(lat[f])), float(np.nanmax(lat[f]))],
                "lon_range": [float(np.nanmin(lon[f])), float(np.nanmax(lon[f]))],
            }
    return out
