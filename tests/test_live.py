from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tests.synthetic import make_cutout
from tests.test_backfill import FakeStorage
from thermal import config, live, products, timeseries
from thermal.model import save_model
from training import data, models


def test_nominal_time_from_id():
    t = live.nominal_time_from_id("MSG3-SEVI-MSG15-0100-NA-20240316215741.846000000Z-NA", 15)
    assert t == datetime(2024, 3, 16, 22, 0, tzinfo=UTC)
    assert live.nominal_time_from_id("garbage", 15) is None


def _train_model(tmp_path, sat):
    t0 = datetime(2024, 3, 10, 22, tzinfo=UTC)
    paths = {
        t0.replace(day=10 + i): make_cutout(tmp_path / f"train{i}.nc", seed=i) for i in range(3)
    }
    table = data.collect(
        None, sat, list(paths), tmp_path, local_paths=paths, max_pixels_per_image=150
    )
    arrays = models.fit_ridge_poly(table.X, table.y, table.names, sat, "MSG3", degree=2, alpha=1e-2)
    path = tmp_path / "ridge.npz"
    save_model(path, arrays)
    return path


class FakeExists:
    """Bucket.exists for the FakeClient (which has no head_object)."""


def test_process_timestamp_end_to_end(tmp_path, monkeypatch):
    sat = config.satellite("msg-0deg")
    storage = FakeStorage()
    objects = storage.private._client.objects
    storage.private.exists = lambda key: key in objects
    storage.public.exists = storage.private.exists
    storage.private.put_file = lambda key, path, content_type=None: objects.__setitem__(
        key, Path(path).read_bytes()
    )
    storage.public.put_file = storage.private.put_file
    storage.private.get_file = lambda key, path: (
        Path(path).parent.mkdir(parents=True, exist_ok=True),
        Path(path).write_bytes(objects[key]),
    )
    storage.public.list_keys = lambda prefix, max_keys=None: [
        k for k in objects if k.startswith(prefix)
    ]

    model_path = _train_model(tmp_path, sat)
    objects["models/msg-0deg/MSG3/latest.json"] = (
        b'{"model_id": "ridge_test", "key": "models/msg-0deg/MSG3/ridge_test.npz"}'
    )
    objects["models/msg-0deg/MSG3/ridge_test.npz"] = model_path.read_bytes()

    t = datetime(2024, 3, 16, 22, tzinfo=UTC)
    hot = make_cutout(tmp_path / "hot.nc", seed=9, hot=(6, 10, 0.5))
    # pretend the cutout is already in the private bucket so no Data Tailor is needed
    from thermal.storage import raw_key

    objects[raw_key("msg-0deg", t, "l15")] = hot.read_bytes()

    cache = live.ModelCache(storage, tmp_path / "models")
    result = live.process_timestamp(None, storage, sat, t, cache, tmp_path / "work")
    assert result["ok"], result
    assert result["platform"] == "MSG3" and result["model"] == "ridge_test"
    assert "products/msg-0deg/2024/03/16/20240316T220000/residual.npz" in objects
    assert "products/msg-0deg/2024/03/16/20240316T220000/residual.png" in objects
    assert "vis06" not in result["products"]["png"]  # night
    rows = result["rows"]
    assert rows[0]["region"] == "reykjanes" and rows[0]["max_z"] > 10 and rows[0]["n_above"] >= 1
    assert rows[0]["vrp_w"] > 0
    shard = timeseries.read_shard(storage, "timeseries/msg-0deg/reykjanes/2024-03.jsonl")
    assert len(shard) == 1 and shard[0]["time"] == "2024-03-16T22:00:00Z"
    # second call is a no-op
    again = live.process_timestamp(None, storage, sat, t, cache, tmp_path / "work2")
    assert again.get("skipped") == "exists"
    arr = products.load_residual(storage, "msg-0deg", t)
    assert arr["flags"].dtype == np.uint8 and arr["residual"].shape == (12, 20)


def test_candidate_window_respects_min_age(monkeypatch):
    from datetime import timedelta

    class P:
        def __init__(self, end):
            self.id = f"MSG3-SEVI-MSG15-0100-NA-{end:%Y%m%d%H%M%S}.000000000Z-NA"

        def __str__(self):
            return self.id

    now = datetime(2026, 11, 6, 21, 22, tzinfo=UTC)
    ends = [now - timedelta(minutes=m) for m in (20, 50, 80, 140, 200)]

    class C:
        def search(self, coll, start, end):
            return [P(e) for e in ends]

    sat = {"collection": "c", "cadence_minutes": 15}
    got = live.candidate_times(C(), sat, now - timedelta(hours=3), now - timedelta(minutes=75))
    assert all(t <= now - timedelta(minutes=75) for t in got)
    assert len(got) == 3
