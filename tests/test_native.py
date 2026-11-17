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


def test_mask_outside_box_matches_tailor(tmp_path):
    path, _, _ = _write(tmp_path)
    with xr.open_dataset(path, decode_times=False) as ds:
        ds.load()
    lat, lon, _ = geo.geolocate(ds)
    box = [64.0, 63.9, -22.4, -22.0]
    masked = native.mask_outside_box(ds, box)
    inside = (lat >= 63.9) & (lat <= 64.0) & (lon >= -22.4) & (lon <= -22.0)
    assert inside.any() and (~inside).any()
    for i in range(1, 12):
        v = masked[f"channel_{i}"].values
        assert np.isfinite(v[inside]).all() and np.isnan(v[~inside]).all()


def test_nearest_on_grid_aligns_offset_grids():
    src_x = np.arange(10) * 3000.0
    src_y = 100000.0 - np.arange(6) * 3000.0
    vals = np.arange(60, dtype=float).reshape(6, 10)
    # destination: columns 2..5 and rows 1..3, with a small sub-pixel offset, plus one
    # column beyond the source extent
    dst_x = np.array([2, 3, 4, 5, 12]) * 3000.0 + 400.0
    dst_y = 100000.0 - np.array([1, 2, 3]) * 3000.0 - 300.0
    out = geo.nearest_on_grid(vals, src_x, src_y, dst_x, dst_y)
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out[:, :4], vals[1:4, 2:6])
    assert np.isnan(out[:, 4]).all()


def test_cloud_mask_dataset_maps_no_data_to_nan():
    x = np.arange(4) * 3000.0
    y = 5.0e6 - np.arange(3) * 3000.0
    vals = np.array([[0, 1, 2, 3], [2, 2, 3, 1], [0, 0, 1, 2]], dtype=float)
    ds = native.cloud_mask_dataset(
        vals, x, y, {"h": H, "lon_0": 9.5, "a": A, "b": B}, "MSG4-SEVI-MSGCLMK-x.grb"
    )
    cm = ds["cloud_mask"].values
    assert np.isnan(cm[0, 3]) and np.isnan(cm[1, 2])
    assert np.nansum(cm == 2) == 4
    assert ds["cloud_mask"].attrs["grid_mapping"] == "geostationary"
    assert "x" in ds.coords and "y" in ds.coords


def _mask_like(image_path, path, extra_cols=0, shift_cols=0, with_coords=True):
    with xr.open_dataset(image_path, decode_times=False) as img:
        x = np.asarray(img["x"].values)
        y = np.asarray(img["y"].values)
    step = x[1] - x[0]
    xs = x[0] + (np.arange(x.size + extra_cols) - shift_cols) * step
    rng = np.random.default_rng(1)
    vals = rng.integers(0, 3, size=(y.size, xs.size)).astype(float)
    coords = {"x": ("x", xs), "y": ("y", y)} if with_coords else {}
    xr.Dataset({"cloud_mask": (("y", "x"), vals)}, coords=coords).to_netcdf(path)
    return vals, xs, y


def test_cloud_on_image_grid(tmp_path):
    from tests.synthetic import make_cutout

    image = make_cutout(tmp_path / "l15.nc")
    sat = config.satellite("msg-0deg")
    # same shape: used as it is
    same, _, _ = _mask_like(image, tmp_path / "same.nc")
    np.testing.assert_array_equal(
        products.cloud_on_image_grid(tmp_path / "same.nc", image, sat), same
    )
    # wider and shifted by one column (like the Data Tailor's 203 against 201): aligned
    wide, _, _ = _mask_like(image, tmp_path / "wide.nc", extra_cols=2, shift_cols=1)
    aligned = products.cloud_on_image_grid(tmp_path / "wide.nc", image, sat)
    np.testing.assert_array_equal(aligned, wide[:, 1:-1])
    # different shape and no coordinates: unavailable rather than an error
    _mask_like(image, tmp_path / "bare.nc", extra_cols=2, with_coords=False)
    assert products.cloud_on_image_grid(tmp_path / "bare.nc", image, sat) is None


def test_pipeline_routes_rss_cloud_mask_natively(tmp_path, monkeypatch):
    from thermal import pipeline

    calls = []

    def fake(client, sat, t, coll_id, roi, out_dir, entry, *a, kind="l15", image_nc=None, **k):
        calls.append((kind, image_nc))
        path = tmp_path / f"{kind}.nc"
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(pipeline, "_native_with_retries", fake)
    sat = config.satellite("msg-rss")
    report = pipeline.fetch_timestamp(
        None, None, sat, datetime(2024, 11, 21, 1, tzinfo=UTC), tmp_path, summarise=False
    )
    assert report["ok"]
    assert [c[0] for c in calls] == ["l15", "clm"]
    assert calls[1][1] == tmp_path / "l15.nc"  # the mask is put on the image grid


def test_quicklooks_only_for_the_channels_we_use():
    from thermal import pipeline

    for key in ("msg-0deg", "msg-rss"):
        assert pipeline.quicklook_variables(config.satellite(key)) == [
            "channel_1",
            "channel_4",
            "channel_9",
        ]
    custom = {"variable_map": {"channel_9": "IR_108"}, "quicklook_channels": ["IR_108"]}
    assert pipeline.quicklook_variables(custom) == ["channel_9"]
