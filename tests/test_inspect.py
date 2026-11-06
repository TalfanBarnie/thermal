import numpy as np
import xarray as xr

from thermal import inspect


def test_summarise_value_counts(tmp_path):
    ds = xr.Dataset({"cloud_mask": (("y", "x"), np.array([[0.0, 2.0], [2.0, np.nan]]))})
    nc = tmp_path / "c.nc"
    ds.to_netcdf(nc)
    s = inspect.summarise(nc)
    v = s["variables"]["cloud_mask"]
    assert v["value_counts"] == {"0.0": 1, "2.0": 2}
    assert v["finite_fraction"] == 0.75
    assert "error" in s["geolocation"]
