"""Level 1.5 cutouts made locally from the native SEVIRI product, bypassing the Data Tailor.

The Data Tailor mis-georeferences rapid scan (RSS) level 1.5 images (docs/SPEC.md section
19), so for sources configured with `l15_method: native` the original `.nat` file is
downloaded from the Data Store and cropped here with satpy. The cutout is written in the
same layout the Data Tailor produces (radiances in `channel_1` .. `channel_11` on the
native geostationary grid, a CF `geostationary` grid mapping, x/y in metres, `chNN_cal`
offset/slope attributes and `EPCT_product_name`), so the rest of the pipeline is unchanged.

satpy is an optional dependency (`pip install -e ".[native]"`); it is imported lazily.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# SEVIRI channel order of the Data Tailor output (channel_N) and of the native header.
SEVIRI_CHANNELS = [
    "VIS006",
    "VIS008",
    "IR_016",
    "IR_039",
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
]


def native_entry(product: Any) -> str:
    """Name of the `.nat` file inside a Data Store product."""
    names = [e for e in product.entries if str(e).lower().endswith(".nat")]
    if not names:
        raise RuntimeError(f"no .nat entry in {product}: {list(product.entries)}")
    return names[0]


def download_native(product: Any, out_dir: Path) -> Path:
    """Stream the product's `.nat` file into out_dir and return its path."""
    entry = native_entry(product)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(entry).name
    t0 = time.monotonic()
    with product.open(entry=entry) as stream, open(target, "wb") as fh:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            fh.write(chunk)
    size = target.stat().st_size
    dt = time.monotonic() - t0
    log.info("downloaded %s (%.1f MB in %.1f s)", target.name, size / 1e6, dt)
    return target


def cutout_dataset(
    radiances: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    proj: dict[str, float],
    calibration: dict[str, tuple[float, float]],
    product_name: str,
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Build a Data-Tailor-compatible cutout dataset from cropped radiance arrays.

    radiances: channel name -> 2-D array (y, x), north up and east to the right.
    x, y: projection coordinates of pixel centres in metres.
    proj: h (satellite height above the ellipsoid), lon_0, a, b.
    calibration: channel name -> (offset, slope) of the radiance = offset + slope * count law.
    """
    data_vars = {}
    attrs: dict[str, Any] = {
        "Conventions": "CF-1.5",
        "EPCT_product_name": product_name,
        "radiometric_parameters_format": "offset slope",
        "source": "satpy seviri_l1b_native, cropped locally (Data Tailor bypassed)",
    }
    for i, ch in enumerate(SEVIRI_CHANNELS, start=1):
        if ch not in radiances:
            continue
        arr = np.asarray(radiances[ch], dtype=np.float32)
        if arr.shape != (y.size, x.size):
            raise ValueError(f"{ch} has shape {arr.shape}, expected {(y.size, x.size)}")
        data_vars[f"channel_{i}"] = xr.DataArray(
            arr,
            dims=("y", "x"),
            attrs={
                "long_name": f"channel_{i}",
                "short_name": f"channel_{i}",
                "grid_mapping": "geostationary",
                "comment": f"band {i:02d} ({ch}) radiance, mW m-2 sr-1 (cm-1)-1",
            },
        )
        if ch in calibration:
            offset, slope = calibration[ch]
            attrs[f"ch{i:02d}_cal"] = f"{offset:.12e} {slope:.12e}"
    data_vars["geostationary"] = xr.DataArray(
        np.array(b"", dtype="S1"),
        attrs={
            "grid_mapping_name": "geostationary",
            "longitude_of_projection_origin": float(proj["lon_0"]),
            "perspective_point_height": float(proj["h"]),
            "semi_major_axis": float(proj["a"]),
            "semi_minor_axis": float(proj["b"]),
            "sweep_angle_axis": "y",
            "false_easting": 0.0,
            "false_northing": 0.0,
        },
    )
    coords = {
        "x": xr.DataArray(
            np.asarray(x, dtype=float),
            dims="x",
            attrs={"standard_name": "projection_x_coordinate", "units": "m"},
        ),
        "y": xr.DataArray(
            np.asarray(y, dtype=float),
            dims="y",
            attrs={"standard_name": "projection_y_coordinate", "units": "m"},
        ),
    }
    attrs.update(extra_attrs or {})
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def _header_calibration(scn: Any) -> dict[str, tuple[float, float]]:
    """Nominal (offset, slope) per channel from the native file header."""
    handlers = [
        fh for fhs in scn._readers["seviri_l1b_native"].file_handlers.values() for fh in fhs
    ]
    cal = handlers[0].header["15_DATA_HEADER"]["RadiometricProcessing"]["Level15ImageCalibration"]
    slopes, offsets = np.asarray(cal["CalSlope"]), np.asarray(cal["CalOffset"])
    return {ch: (float(offsets[i]), float(slopes[i])) for i, ch in enumerate(SEVIRI_CHANNELS)}


def crop_native(nat_path: Path, roi_nswe: list[float], out_nc: Path, product_name: str) -> Path:
    """Crop a native SEVIRI file to the N, S, W, E box and write a cutout netCDF."""
    from satpy import Scene

    north, south, west, east = roi_nswe
    scn = Scene(filenames=[str(nat_path)], reader="seviri_l1b_native")
    scn.load(SEVIRI_CHANNELS, calibration="radiance", upper_right_corner="NE")
    cropped = scn.crop(ll_bbox=(west, south, east, north))
    first = cropped[SEVIRI_CHANNELS[0]]
    area = first.attrs["area"]
    x, y = area.get_proj_vectors()
    crs = area.crs.to_dict()
    a = float(crs.get("a", 6378169.0))
    if "b" in crs:
        b = float(crs["b"])
    elif "rf" in crs:
        b = a * (1.0 - 1.0 / float(crs["rf"]))
    else:
        b = 6356583.8
    proj = {"h": float(crs["h"]), "lon_0": float(crs.get("lon_0", 0.0)), "a": a, "b": b}
    radiances = {}
    for ch in SEVIRI_CHANNELS:
        da = cropped[ch]
        if da.attrs["area"].shape != area.shape:
            raise ValueError(f"{ch} cropped to {da.attrs['area'].shape}, expected {area.shape}")
        radiances[ch] = np.asarray(da.values, dtype=np.float32)
    ds = cutout_dataset(
        radiances,
        x,
        y,
        proj,
        _header_calibration(scn),
        product_name,
        extra_attrs={
            "platform_name": str(first.attrs.get("platform_name", "")),
            "start_time": str(first.attrs.get("start_time", "")),
            "end_time": str(first.attrs.get("end_time", "")),
        },
    )
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    log.info("cropped %s to %s (%d x %d)", nat_path.name, out_nc.name, y.size, x.size)
    return out_nc
