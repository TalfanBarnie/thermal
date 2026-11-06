"""Regions of interest: the Iceland box and the monitoring polygons."""

from __future__ import annotations

import numpy as np
import shapely

from . import config


def box_nswe(box: dict[str, float] | None = None) -> list[float]:
    """Iceland box as the Data Tailor NSWE list (the service rejects a comma string)."""
    b = box or config.iceland_box()
    return [float(b["north"]), float(b["south"]), float(b["west"]), float(b["east"])]


def box_mask(lat: np.ndarray, lon: np.ndarray, box: dict[str, float] | None = None) -> np.ndarray:
    b = box or config.iceland_box()
    return (lat <= b["north"]) & (lat >= b["south"]) & (lon >= b["west"]) & (lon <= b["east"])


def polygon(name: str) -> shapely.Polygon:
    regions = config.monitoring_regions()
    if name not in regions:
        raise KeyError(f"unknown monitoring region {name!r}; known: {sorted(regions)}")
    return shapely.Polygon(regions[name]["polygon"])


def polygon_mask(poly: shapely.Polygon, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean mask of pixel centres inside the polygon (lon, lat order)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    out = np.zeros(lat.shape, dtype=bool)
    out[valid] = shapely.contains_xy(poly, lon[valid], lat[valid])
    return out


def region_mask(name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return polygon_mask(polygon(name), lat, lon)
