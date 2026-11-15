"""The locally cropped (native) cutout must be read by the pipeline exactly like a
Data Tailor cutout. satpy itself is not needed for these tests."""

from datetime import UTC, datetime

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from thermal import config, features, geo, inspect, native, products

H = 35785831.0
A, B = 6378169.0, 6356583.8


def _rss_grid(rows=10, cols=16):
    crs = CRS.from_proj4(f"+proj=geos +h={H} +lon_0=9.5 +sweep=y +a={A} +b={B} +units=m")
    fwd = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    x0, y0 = fwd.transform(-22.2, 63.95)
    x = x0 + (np.arange(cols) - cols / 2) * 3000.0
    y = y0 + (rows / 2 - np.arange(rows)) * 3000.0
    return x, y


def _write(tmp_path, hot=(4, 7)):
    x, y = _rss_grid()
    rng = np.random.default_rng(0)
    rad = {}
    for i, ch in enumerate(native.SEVIRI_CHANNELS, start=1):
        rad[ch] = (rng.random((y.size, x.size)) + i).astype(np.float32)
    rad["IR_039"][hot] = 99.0
    cal = {ch: (-1.0 * i, 0.02 * i) for i, ch in enumerate(native.SEVIRI_CHANNELS, start=1)}
    ds = native.cutout_dataset(
        rad,
        x,
        y,
        {"h": H, "lon_0": 9.5, "a": A, "b": B},
        cal,
        "MSG4-SEVI-MSG15-0100-NA-20240316215918.384000000Z-NA.nat",
    )
    path = tmp_path / "rss.nc"
    ds.to_netcdf(path)
    return path, x, y


def test_cutout_layout_matches_tailor_conventions(tmp_path):
    path, x, y = _write(tmp_path)
    with xr.open_dataset(path, decode_times=False) as ds:
        assert [f"channel_{i}" for i in range(1, 12)] == sorted(
            (v for v in ds.data_vars if v.startswith("channel_")), key=lambda s: int(s[8:])
        )
        assert ds["channel_4"].attrs["grid_mapping"] == "geostationary"
        assert ds["geostationary"].attrs["longitude_of_projection_origin"] == 9.5
        assert features.platform_of(ds) == "MSG4"
        assert products.saturation_radiance(ds, "channel_4") == -4.0 + 0.08 * 1023.0
        lat, lon, _ = geo.geolocate(ds)
    # centre pixel back on Reykjanes, north up
    assert abs(lat[5, 8] - 63.95) < 0.1 and abs(lon[5, 8] + 22.2) < 0.2
    assert lat[0, 8] > lat[-1, 8]


def test_pipeline_reads_native_cutout(tmp_path):
    path, _, _ = _write(tmp_path)
    sat = config.satellite("msg-rss")
    fs = features.build_features(path, datetime(2024, 3, 16, 22, tzinfo=UTC), sat)
    assert fs.platform == "MSG4"
    assert fs.X.shape[1] == len(fs.names)
    assert fs.valid.all()
    summary = inspect.summarise(path)
    hot = summary["hottest_channel_4"]
    assert (hot["row"], hot["col"]) == (4, 7) and hot["radiance"] == 99.0


def test_native_entry_picks_nat_file():
    class P:
        entries = ("EOPMetadata.xml", "MSG4-SEVI-MSG15-0100-NA-x.nat", "manifest.xml")

    assert native.native_entry(P()).endswith(".nat")


def test_extract_native_from_zip(tmp_path):
    import zipfile

    zp = tmp_path / "p.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("EOPMetadata.xml", "<x/>")
        zf.writestr("MSG4-SEVI-MSG15-0100-NA-x.nat", b"\x01" * 1000)
    out = native.extract_native(zp, tmp_path)
    assert out.name.endswith(".nat") and out.read_bytes() == b"\x01" * 1000
