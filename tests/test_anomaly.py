from datetime import UTC, datetime

import numpy as np

from tests.synthetic import make_cutout
from thermal import anomaly, config, features
from thermal.model import BackgroundModel, save_model
from training import data, models


def test_end_to_end_hot_pixel_detected(tmp_path):
    sat = config.satellite("msg-0deg")
    t = datetime(2024, 3, 16, 22, tzinfo=UTC)
    # train on three clean synthetic cutouts
    paths = {
        t.replace(day=10 + i): make_cutout(tmp_path / f"train{i}.nc", seed=i) for i in range(3)
    }
    table = data.collect(
        None, sat, list(paths), tmp_path, local_paths=paths, max_pixels_per_image=100
    )
    assert table.X.shape[0] == 300 and len(table.images) == 3
    train, test = data.holdout_split(table.group, 0.34)
    assert test.sum() == 100 and train.sum() == 200
    arrays = models.fit_ridge_poly(table.X, table.y, table.names, sat, "MSG3", degree=2, alpha=1e-2)
    path = tmp_path / "ridge.npz"
    save_model(path, arrays)
    model = BackgroundModel.load(path)
    # a cutout with a hot pixel inside the Reykjanes region
    hot_nc = make_cutout(tmp_path / "hot.nc", seed=9, hot=(6, 10, 0.5))
    fs = features.build_features(hot_nc, t, sat)
    images = anomaly.predict_images(model, fs)
    assert images["residual"].shape == fs.shape
    r, c = np.unravel_index(np.nanargmax(images["residual"]), fs.shape)
    assert (r, c) == (6, 10)
    assert images["zscore"][6, 10] > 10
    # the region stats: use a region covering the whole grid via the iceland box helper
    stats = anomaly.region_stats(images, fs, "reykjanes")
    assert set(stats) >= {"sum_pos", "max", "n_above", "n_valid", "sza_mean"}
