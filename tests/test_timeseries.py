from tests.test_backfill import FakeStorage
from thermal import timeseries


def test_radiance_conversion_factor():
    # per-wavenumber (mW) to per-micron (W) at 2547.8 cm-1 is about 0.65
    assert abs(timeseries.radiance_per_cm_to_per_um(1.0) - 0.649) < 0.01


def test_vrp_order_of_magnitude():
    # one 42 km2 pixel with 0.2 mW m-2 sr-1 (cm-1)-1 excess: ~ 42e6 * 18.9 * 0.13 W ~ 1e8 W
    vrp = timeseries.MIR_METHOD_CONSTANT * 42e6 * timeseries.radiance_per_cm_to_per_um(0.2)
    assert 5e7 < vrp < 2e8


def test_shard_upsert_merges_and_sorts():
    storage = FakeStorage()
    rows = [
        {"time": "2024-03-16T22:00:00Z", "sat": "msg-0deg", "region": "reykjanes", "sum_pos": 1.0},
        {"time": "2024-03-16T21:45:00Z", "sat": "msg-0deg", "region": "reykjanes", "sum_pos": 0.1},
    ]
    keys = timeseries.upsert_rows(storage, rows)
    assert keys == ["timeseries/msg-0deg/reykjanes/2024-03.jsonl"]
    got = timeseries.read_shard(storage, keys[0])
    assert [r["time"] for r in got] == ["2024-03-16T21:45:00Z", "2024-03-16T22:00:00Z"]
    # replace a row and add one in another month
    timeseries.upsert_rows(
        storage,
        [
            {
                "time": "2024-03-16T22:00:00Z",
                "sat": "msg-0deg",
                "region": "reykjanes",
                "sum_pos": 2.0,
            },
            {
                "time": "2024-04-01T00:00:00Z",
                "sat": "msg-0deg",
                "region": "reykjanes",
                "sum_pos": 0.0,
            },
        ],
    )
    got = timeseries.read_shard(storage, keys[0])
    assert len(got) == 2 and got[1]["sum_pos"] == 2.0
    assert (
        timeseries.read_shard(storage, "timeseries/msg-0deg/reykjanes/2024-04.jsonl")[0]["sum_pos"]
        == 0.0
    )
    assert timeseries.read_shard(storage, "timeseries/msg-0deg/nowhere/2024-04.jsonl") == []


def test_build_index_lists_months(monkeypatch):
    storage = FakeStorage()
    timeseries.upsert_rows(
        storage,
        [
            {
                "time": "2024-03-16T22:00:00Z",
                "sat": "msg-0deg",
                "region": "reykjanes",
                "sum_pos": 1.0,
            },
            {
                "time": "2024-05-01T00:00:00Z",
                "sat": "msg-0deg",
                "region": "reykjanes",
                "sum_pos": 1.0,
            },
        ],
    )
    # FakeClient has no paginator; emulate list_keys
    storage.public.list_keys = lambda prefix, max_keys=None: [
        k for k in storage.public._client.objects if k.startswith(prefix)
    ]
    index = timeseries.build_index(
        storage, {"msg-0deg": {"monitoring_regions": ["reykjanes"], "cadence_minutes": 15}}
    )
    assert index["sources"]["msg-0deg"]["regions"]["reykjanes"]["months"] == ["2024-03", "2024-05"]
    assert storage.public.get_json("index.json")["sources"]["msg-0deg"]["cadence_minutes"] == 15
