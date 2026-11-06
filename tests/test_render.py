import numpy as np
import xarray as xr
from PIL import Image

from thermal import render


def test_colormap_shape_and_nan():
    rgb = render.colormap(np.array([[0.0, 1.0], [np.nan, 0.5]]))
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[1, 0]) == (60, 60, 60)
    assert rgb[0, 1].sum() > rgb[0, 0].sum()


def test_quicklooks(tmp_path):
    data = np.arange(12, dtype=float).reshape(3, 4)
    ds = xr.Dataset(
        {"channel_4": (("y", "x"), data)},
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )
    nc = tmp_path / "cut.nc"
    ds.to_netcdf(nc)
    out = render.quicklooks(nc, tmp_path / "ql", ["channel_4", "missing"], scale=2)
    assert set(out) == {"channel_4"}
    img = Image.open(out["channel_4"]["path"])
    assert img.size == (8, 6)
