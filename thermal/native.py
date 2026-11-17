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


def extract_member(zip_path: Path, out_dir: Path, suffix: str, name: str | None = None) -> Path:
    """Extract the first member ending in `suffix` from a product zip into out_dir.

    `name` renames the extracted file (satpy picks readers by file name pattern).
    """
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(suffix)]
        if not names:
            raise RuntimeError(f"no {suffix} member in {zip_path.name}: {zf.namelist()}")
        target = out_dir / (name or Path(names[0]).name)
        with zf.open(names[0]) as src, open(target, "wb") as dst:
            for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
                dst.write(chunk)
    return target


def extract_native(zip_path: Path, out_dir: Path) -> Path:
    """Extract the `.nat` member of a Data Store product zip into out_dir."""
    return extract_member(zip_path, out_dir, ".nat")


def download_member(product: Any, out_dir: Path, suffix: str, name: str | None = None) -> Path:
    """Download the product zip, extract its `suffix` file into out_dir and return its path.

    The zip is the lighter route: for an RSS image it is 36 MB against the 102 MB `.nat`
    that the per-entry endpoint serves uncompressed. The zip is deleted after extraction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{product}.zip"
    t0 = time.monotonic()
    with product.open() as stream, open(zip_path, "wb") as fh:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            fh.write(chunk)
    zip_mb = zip_path.stat().st_size / 1e6
    t1 = time.monotonic()
    try:
        target = extract_member(zip_path, out_dir, suffix, name)
    finally:
        zip_path.unlink(missing_ok=True)
    log.info(
        "downloaded %s (%.2f MB zip in %.1f s, extracted in %.1f s)",
        target.name,
        zip_mb,
        t1 - t0,
        time.monotonic() - t1,
    )
    return target


def download_native(product: Any, out_dir: Path) -> Path:
    """Download a level 1.5 product and return its extracted `.nat` file."""
    return download_member(product, out_dir, ".nat")


def download_cloud_mask(product: Any, out_dir: Path) -> Path:
    """Download a cloud mask product and return its GRIB file, named as satpy expects."""
    return download_member(product, out_dir, ".grb", name=f"{product}.grb")


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


def mask_outside_box(ds: xr.Dataset, roi_nswe: list[float]) -> xr.Dataset:
    """Set channel pixels outside the N, S, W, E box to NaN, as the Data Tailor does.

    A lat/lon box is not a rectangle on the geostationary grid, so the bounding rectangle
    that satpy crops contains extra pixels; masking them keeps the domain identical to
    Data Tailor cutouts (and to the 0 degree training domain).
    """
    from . import geo

    north, south, west, east = roi_nswe
    lat, lon, _ = geo.geolocate(ds)
    inside = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    for name in [v for v in ds.data_vars if v.startswith("channel_")]:
        ds[name] = ds[name].where(inside)
    return ds


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
    proj = _proj_of(area)
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
    ds = mask_outside_box(ds, roi_nswe)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    log.info("cropped %s to %s (%d x %d)", nat_path.name, out_nc.name, y.size, x.size)
    return out_nc


CLOUD_NO_DATA = 3  # flag value of the MSG cloud mask meaning "no data"


def cloud_mask_dataset(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    proj: dict[str, float],
    product_name: str,
) -> xr.Dataset:
    """Data-Tailor-compatible cloud mask cutout: 0 clear sea, 1 clear land, 2 cloud, NaN
    no data (the product's flag 3 and anything outside the source grid)."""
    v = np.asarray(values, dtype=np.float32).copy()
    v[v == CLOUD_NO_DATA] = np.nan
    if v.shape != (y.size, x.size):
        raise ValueError(f"cloud mask has shape {v.shape}, expected {(y.size, x.size)}")
    ds = cutout_dataset({}, x, y, proj, {}, product_name)
    ds["cloud_mask"] = xr.DataArray(
        v,
        dims=("y", "x"),
        attrs={
            "long_name": "cloud_mask",
            "short_name": "cloud_mask",
            "grid_mapping": "geostationary",
            "comment": "0 clear sky over water, 1 clear sky over land, 2 cloudy, NaN no data",
        },
    )
    ds.attrs["source"] = "satpy seviri_l2_grib, cropped locally (Data Tailor bypassed)"
    ds.attrs.pop("radiometric_parameters_format", None)
    return ds


def _proj_of(area: Any) -> dict[str, float]:
    crs = area.crs.to_dict()
    a = float(crs.get("a", 6378169.0))
    if "b" in crs:
        b = float(crs["b"])
    elif "rf" in crs:
        b = a * (1.0 - 1.0 / float(crs["rf"]))
    else:
        b = 6356583.8
    return {"h": float(crs["h"]), "lon_0": float(crs.get("lon_0", 0.0)), "a": a, "b": b}


def place_on_image_grid(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, image_nc: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Move a field onto an image cutout's grid and adopt the image's projection.

    Matching on x/y is exact whatever ellipsoid each product declares, because
    geostationary x and y are scan angles scaled by the satellite height. The result
    carries the image's projection so that the two files geolocate identically (the
    cloud mask GRIB declares a different ellipsoid from the level 1.5 image).
    """
    from . import geo

    with xr.open_dataset(image_nc, decode_times=False) as img:
        img_x = np.asarray(img["x"].values)
        img_y = np.asarray(img["y"].values)
        g = img["geostationary"].attrs
        proj = {
            "h": float(g["perspective_point_height"]),
            "lon_0": float(g["longitude_of_projection_origin"]),
            "a": float(g["semi_major_axis"]),
            "b": float(g["semi_minor_axis"]),
        }
    return geo.nearest_on_grid(values, x, y, img_x, img_y), img_x, img_y, proj


def crop_cloud_mask(
    grb_path: Path,
    roi_nswe: list[float],
    out_nc: Path,
    product_name: str,
    image_nc: Path | None = None,
) -> Path:
    """Crop a cloud mask GRIB file to the box and write a cutout netCDF.

    With `image_nc` (the image cutout of the same timestamp) the mask is put exactly on
    the image's grid, so the two can be overlaid pixel for pixel; without it the mask is
    cropped to the box on its own grid.
    """
    from satpy import Scene

    from . import geo

    north, south, west, east = roi_nswe
    scn = Scene(filenames=[str(grb_path)], reader="seviri_l2_grib")
    scn.load(["cloud_mask"], upper_right_corner="NE")
    # crop with a margin so the image grid is fully covered before alignment
    cropped = scn.crop(ll_bbox=(west - 2.0, south - 1.0, east + 2.0, north + 1.0))
    da = cropped["cloud_mask"]
    area = da.attrs["area"]
    x, y = area.get_proj_vectors()
    values = np.asarray(da.values, dtype=float)
    values[values == CLOUD_NO_DATA] = np.nan
    proj = _proj_of(area)
    if image_nc is not None:
        values, x, y, proj = place_on_image_grid(values, x, y, image_nc)
    ds = cloud_mask_dataset(values, x, y, proj, product_name)
    lat, lon, _ = geo.geolocate(ds)
    inside = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    ds["cloud_mask"] = ds["cloud_mask"].where(inside)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    log.info("cropped %s to %s (%d x %d)", grb_path.name, out_nc.name, y.size, x.size)
    return out_nc
