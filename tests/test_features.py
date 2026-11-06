from datetime import UTC, datetime

import numpy as np

from tests.synthetic import make_cutout
from thermal import config, features


def test_build_features_shapes_and_names(tmp_path):
    sat = config.satellite("msg-0deg")
    nc = make_cutout(tmp_path / "c.nc")
    fs = features.build_features(nc, datetime(2024, 3, 16, 22, tzinfo=UTC), sat)
    assert fs.platform == "MSG3"
    assert fs.X.shape == (12 * 20, len(features.feature_names(sat)))
    assert fs.names == features.feature_names(sat)
    assert fs.y.shape == (240,)
    assert np.isfinite(fs.X).all() and np.isfinite(fs.y).all()
    sza = fs.X[:, fs.names.index("sza")]
    assert np.all(sza > 100)  # night over Iceland in March at 22 UTC
    vza = fs.X[:, fs.names.index("vza")]
    assert np.all((vza > 70) & (vza < 80))
    img = fs.to_image(fs.y)
    assert img.shape == fs.shape and np.isfinite(img).all()


def test_brightness_temperature_monotonic():
    r = np.array([20.0, 40.0, 80.0])
    bt = features.brightness_temperature(r, "MSG3", "IR_108")
    assert np.all(np.diff(bt) > 0) and 200 < bt[0] < 320
    assert np.isnan(features.brightness_temperature(np.array([0.0]), "MSG3", "IR_108"))[0]
