"""PNG quicklooks of cutouts: colour-mapped, 8-bit, fixed or percentile stretch, upscaled.

These are derived images for inspection and display; they are not a numerical
representation of the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

# A compact perceptual colour map (16 control points of matplotlib's inferno), interpolated.
_INFERNO = np.array(
    [
        [0, 0, 4],
        [12, 8, 48],
        [40, 11, 84],
        [70, 10, 93],
        [101, 21, 110],
        [131, 28, 108],
        [160, 37, 100],
        [188, 50, 85],
        [213, 67, 66],
        [232, 93, 45],
        [245, 123, 22],
        [251, 158, 6],
        [251, 196, 22],
        [245, 236, 62],
        [252, 255, 164],
        [255, 255, 255],
    ],
    dtype=float,
)


def colormap(values01: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] (NaN -> dark grey) to RGB uint8."""
    v = np.asarray(values01, dtype=float)
    nan = ~np.isfinite(v)
    v = np.clip(np.where(nan, 0.0, v), 0.0, 1.0)
    pos = v * (len(_INFERNO) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_INFERNO) - 1)
    frac = (pos - lo)[..., None]
    rgb = _INFERNO[lo] * (1 - frac) + _INFERNO[hi] * frac
    rgb[nan] = (60, 60, 60)
    return rgb.astype(np.uint8)


def stretch(values: np.ndarray, vmin: float | None = None, vmax: float | None = None):
    """Scale to [0, 1] with explicit limits or the 1st/99th percentiles."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full(v.shape, np.nan), (np.nan, np.nan)
    lo = np.nanpercentile(finite, 1) if vmin is None else vmin
    hi = np.nanpercentile(finite, 99) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    return (v - lo) / (hi - lo), (float(lo), float(hi))


def to_png(values: np.ndarray, path: Path, scale: int = 4, vmin=None, vmax=None) -> tuple:
    """Write a colour-mapped PNG. Row 0 of the array is the top of the image."""
    v01, limits = stretch(values, vmin, vmax)
    rgb = colormap(v01)
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)
    return limits


def quicklooks(nc_path: Path, out_dir: Path, variables: list[str], scale: int = 4) -> dict:
    """Render the named variables of a cutout to PNGs; returns {variable: {path, vmin, vmax}}."""
    out = {}
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        for name in variables:
            if name not in ds.variables or ds[name].ndim != 2:
                continue
            arr = np.asarray(ds[name].values, dtype=float)
            # projection y increases northwards; make north the top of the image
            if "y" in ds.coords and ds["y"].values[0] < ds["y"].values[-1]:
                arr = arr[::-1]
            path = out_dir / f"{name}.png"
            lo, hi = to_png(arr, path, scale=scale)
            out[name] = {"path": str(path), "vmin": lo, "vmax": hi}
    return out
