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


def test_shard_buffer_reads_each_month_once_and_writes_on_flush():
    from datetime import UTC, datetime

    from tests.test_backfill import FakeStorage
    from thermal import timeseries

    storage = FakeStorage()
    client = storage.public._client
    timeseries.write_shard(
        storage,
        "timeseries/msg-rss/reykjanes/2024-03.jsonl",
        [{"time": "2024-03-01T00:00:00Z", "sat": "msg-rss", "region": "reykjanes", "v": 0}],
    )
    reads, writes = [], []
    get, put = client.get_object, client.put_object
    client.get_object = lambda **kw: (reads.append(kw["Key"]), get(**kw))[1]
    client.put_object = lambda **kw: (writes.append(kw["Key"]), put(**kw))[1]

    buf = timeseries.ShardBuffer(storage)
    for day in range(2, 8):
        month = 3 if day < 6 else 4
        buf.add(
            [
                {
                    "time": f"2024-{month:02d}-{day:02d}T00:00:00Z",
                    "sat": "msg-rss",
                    "region": "reykjanes",
                    "v": day,
                }
            ]
        )
    assert writes == [] and buf.dirty == 2
    assert sorted(set(reads)) == sorted(reads)  # each month read once
    assert buf.has("msg-rss", "reykjanes", datetime(2024, 3, 3, tzinfo=UTC))
    assert not buf.has("msg-rss", "reykjanes", datetime(2024, 3, 9, tzinfo=UTC))
    written = buf.flush()
    assert len(written) == 2 and buf.dirty == 0 and len(writes) == 2
    march = timeseries.read_shard(storage, "timeseries/msg-rss/reykjanes/2024-03.jsonl")
    assert [r["time"][:10] for r in march] == [f"2024-03-0{d}" for d in range(1, 6)]
    assert buf.flush() == []  # nothing new to write
