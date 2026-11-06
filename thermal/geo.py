"""Geometry: solar position, geostationary viewing angles, pixel geolocation and area.

Everything is numpy-only so it can run in the Action without satpy.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0
GEO_HEIGHT_M = 35_786_000.0


def _to_utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def solar_position(t: datetime, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solar zenith and azimuth (degrees) at time t for arrays of lat/lon (degrees).

    NOAA solar calculator algorithm; accurate to a small fraction of a degree,
    which is ample for use as a model feature.
    """
    t = _to_utc(t)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    jd = t.timestamp() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    r = np.radians
    seoc = (
        np.sin(r(gmas)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(r(2 * gmas)) * (0.019993 - 0.000101 * jc)
        + np.sin(r(3 * gmas)) * 0.000289
    )
    stl = gmls + seoc
    sal = stl - 0.00569 - 0.00478 * np.sin(r(125.04 - 1934.136 * jc))
    moe = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    oc = moe + 0.00256 * np.cos(r(125.04 - 1934.136 * jc))
    decl = np.arcsin(np.sin(r(oc)) * np.sin(r(sal)))  # radians
    vary = np.tan(r(oc / 2.0)) ** 2
    eot = 4.0 * np.degrees(
        vary * np.sin(2 * r(gmls))
        - 2 * eeo * np.sin(r(gmas))
        + 4 * eeo * vary * np.sin(r(gmas)) * np.cos(2 * r(gmls))
        - 0.5 * vary * vary * np.sin(4 * r(gmls))
        - 1.25 * eeo * eeo * np.sin(2 * r(gmas))
    )
    minutes = t.hour * 60.0 + t.minute + t.second / 60.0 + t.microsecond / 6e7
    tst = (minutes + eot + 4.0 * lon) % 1440.0
    ha = np.where(tst / 4.0 < 0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)
    latr = r(lat)
    cos_zen = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(r(ha))
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zen = np.arccos(cos_zen)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = (np.sin(latr) * np.cos(zen) - np.sin(decl)) / (np.cos(latr) * np.sin(zen))
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az0 = np.degrees(np.arccos(cos_az))
    az = np.where(ha > 0, (az0 + 180.0) % 360.0, (540.0 - az0) % 360.0)
    return np.degrees(zen), az


def _ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, radius: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.stack(
        [
            radius * np.cos(lat) * np.cos(lon),
            radius * np.cos(lat) * np.sin(lon),
            radius * np.sin(lat),
        ],
        axis=-1,
    )


def satellite_angles(
    lat: np.ndarray, lon: np.ndarray, sub_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Viewing zenith and azimuth (degrees) of a geostationary satellite at sub_lon.

    Spherical Earth; adequate for model features and for the plume projection azimuth.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    p = _ecef(lat, lon, EARTH_RADIUS_M)
    s = _ecef(np.zeros_like(lat), np.full_like(lat, sub_lon), EARTH_RADIUS_M + GEO_HEIGHT_M)
    d = s - p
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    latr, lonr = np.radians(lat), np.radians(lon)
    up = p / EARTH_RADIUS_M
    east = np.stack([-np.sin(lonr), np.cos(lonr), np.zeros_like(lonr)], axis=-1)
    north = np.stack(
        [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)], axis=-1
    )
    cos_zen = np.clip(np.sum(d * up, axis=-1), -1.0, 1.0)
    zen = np.degrees(np.arccos(cos_zen))
    az = np.degrees(np.arctan2(np.sum(d * east, axis=-1), np.sum(d * north, axis=-1))) % 360.0
    return zen, az


def pixel_area_m2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approximate ground area of each pixel from the lat/lon of pixel centres (2-D arrays)."""
    p = _ecef(np.asarray(lat, float), np.asarray(lon, float), EARTH_RADIUS_M)
    d_row = np.gradient(p, axis=0)
    d_col = np.gradient(p, axis=1)
    return np.linalg.norm(np.cross(d_row, d_col), axis=-1)


_LAT_NAMES = ("lat", "latitude", "Latitude")
_LON_NAMES = ("lon", "longitude", "Longitude")


def geolocate(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (lat, lon, method) 2-D arrays for a cutout dataset.

    Handles cutouts that carry explicit lat/lon arrays and those on a native
    geostationary grid with projection coordinates and a CF grid mapping.
    Which of these the Data Tailor produces is a milestone M1 question.
    """
    lat_name = next((n for n in _LAT_NAMES if n in ds.variables), None)
    lon_name = next((n for n in _LON_NAMES if n in ds.variables), None)
    if lat_name and lon_name:
        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float)
        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        return lat, lon, f"variables {lat_name}/{lon_name}"

    grid_var = None
    for v in ds.variables.values():
        gm = v.attrs.get("grid_mapping")
        if gm and gm in ds.variables:
            grid_var = ds[gm]
            break
    if grid_var is None:
        for v in ds.variables.values():
            if "grid_mapping_name" in v.attrs:
                grid_var = v
                break
    if grid_var is None:
        raise ValueError("cannot geolocate: no lat/lon variables and no grid mapping found")
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("cannot geolocate: grid mapping found but no x/y coordinates")

    from pyproj import CRS, Transformer

    attrs = grid_var.attrs
    h = float(attrs.get("perspective_point_height", GEO_HEIGHT_M))
    lon0 = float(attrs.get("longitude_of_projection_origin", 0.0))
    sweep = attrs.get("sweep_angle_axis", "y")
    a = float(attrs.get("semi_major_axis", 6378169.0))
    b = float(attrs.get("semi_minor_axis", 6356583.8))
    crs = CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} +a={a} +b={b} +units=m")
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    units = str(ds["x"].attrs.get("units", "m")).lower()
    if units in ("rad", "radian", "radians"):
        x, y = x * h, y * h
    xx, yy = np.meshgrid(x, y)
    tr = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = tr.transform(xx, yy)
    lon = np.where(np.isfinite(lon) & (np.abs(lon) < 1e30), lon, np.nan)
    lat = np.where(np.isfinite(lat) & (np.abs(lat) < 1e30), lat, np.nan)
    return lat, lon, f"geos projection via {grid_var.name}"
