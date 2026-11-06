"""Synthetic cutout generator shared by tests: a geos-projected grid over Iceland with
SEVIRI-like channel variables whose IR3.9 radiance is a known smooth function of the others."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

H = 35785831.0


def make_cutout(
    path: Path, rows: int = 12, cols: int = 20, seed: int = 0, hot: tuple | None = None
):
    rng = np.random.default_rng(seed)
    crs = CRS.from_proj4(f"+proj=geos +h={H} +lon_0=0 +sweep=y +a=6378137 +b=6356752.3 +units=m")
    fwd = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    x0, y0 = fwd.transform(-19.0, 65.0)
    x = x0 + (np.arange(cols) - cols / 2) * 3000.0
    y = y0 + (rows / 2 - np.arange(rows)) * 3000.0  # north at the top (decreasing y)
    data = {}
    base = 280 + 10 * rng.standard_normal((rows, cols))  # pseudo temperature field
    for i in range(1, 12):
        if i in (1, 2, 3):
            data[f"channel_{i}"] = (
                ("y", "x"),
                (rng.random((rows, cols)) * 0.5).astype(np.float32),
                {"grid_mapping": "geostationary"},
            )
        elif i == 4:
            continue
        else:
            data[f"channel_{i}"] = (
                ("y", "x"),
                (base / 5 + i + rng.standard_normal((rows, cols)) * 0.1).astype(np.float32),
                {"grid_mapping": "geostationary"},
            )
    ch9 = data["channel_9"][1]
    ch4 = (0.02 * ch9 - 0.8 + 0.001 * ch9**2 + 0.002 * rng.standard_normal((rows, cols))).astype(
        np.float32
    )
    if hot is not None:
        r, c, amp = hot
        ch4[r, c] += amp
    data["channel_4"] = (("y", "x"), ch4, {"grid_mapping": "geostationary"})
    data["geostationary"] = (
        (),
        np.array(b"", dtype="S1"),
        {
            "grid_mapping_name": "geostationary",
            "perspective_point_height": H,
            "longitude_of_projection_origin": 0.0,
            "sweep_angle_axis": "y",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.3,
        },
    )
    ds = xr.Dataset(
        data,
        coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"})},
        attrs={"EPCT_product_name": "MSG3-SEVI-MSG15-0100-NA-20240316215741.846000000Z-NA.nat"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path
