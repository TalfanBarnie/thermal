from datetime import UTC, date, datetime, timedelta

from thermal import backfill, live, sampling, validate
from thermal.calendar import Calendar, Eruption


def test_eruption_windows_counts_and_alignment():
    e = Eruption("X", date(2024, 3, 16), date(2024, 3, 18))
    ts = sampling.eruption_windows(
        [e], margin_days=1, step_minutes=180, onset_step_minutes=60, onset_days=1
    )
    # window Mar 15 00:00 .. Mar 20 00:00 (5 days) at 3 h = 40; onset Mar 16 hourly = 24
    assert ts[0] == datetime(2024, 3, 15, tzinfo=UTC)
    assert all(t.minute % 15 == 0 for t in ts)
    hourly = [
        t for t in ts if datetime(2024, 3, 16, tzinfo=UTC) <= t < datetime(2024, 3, 17, tzinfo=UTC)
    ]
    assert len(hourly) == 24
    assert len(ts) == 40 + 24 - 8  # onset hours overlapping the 3 h grid are shared
    assert sampling.eruption_windows([e], since=datetime(2025, 1, 1, tzinfo=UTC)) == []


def test_onset_report_and_false_alarms():
    e = Eruption("X", date(2024, 3, 16), date(2024, 3, 17))
    cal = Calendar([e], buffer_days=1)
    t0 = datetime(2024, 3, 14, tzinfo=UTC)
    rows = []
    for h in range(0, 24 * 6, 6):
        t = t0 + timedelta(hours=h)
        z = (
            20.0
            if datetime(2024, 3, 16, 6, tzinfo=UTC) <= t < datetime(2024, 3, 17, 12, tzinfo=UTC)
            else 1.0
        )
        if t == datetime(2024, 3, 19, 12, tzinfo=UTC):
            z = 9.0  # a false alarm well after the window and buffer
        rows.append({"time": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "_t": t, "max_z": z})
    rep = validate.onset_report(rows, [e], margin_days=1)
    assert rep[0]["first_hit"] == "2024-03-16T06:00:00Z" and rep[0]["delay_hours"] == 6.0
    assert rep[0]["n_hits_before_start"] == 0 and rep[0]["peak"] == 20.0
    fa = validate.false_alarms(rows, cal)
    assert fa["n_false_alarms"] == 1 and fa["examples"] == ["2024-03-19T12:00:00Z"]


def test_backfill_process_mode(monkeypatch):
    from tests.test_backfill import FakeStorage

    storage = FakeStorage()
    storage.public.list_keys = lambda prefix, max_keys=None: []
    sat = {
        "key": "msg-0deg",
        "cadence_minutes": 15,
        "collection": "c",
        "tailor_product": "p",
        "monitoring_regions": ["reykjanes"],
    }
    ts = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(3)]
    calls = []

    def fake_process(client, storage_, sat_, t, cache, out_dir, force=False):
        calls.append((t, force))
        return (
            {"ok": True, "platform": "MSG3", "model": "m"}
            if t.hour != 1
            else {"ok": True, "skipped": "exists"}
        )

    monkeypatch.setattr(live, "process_timestamp", fake_process)
    monkeypatch.setattr(live, "ModelCache", lambda storage_, d: None)
    s = backfill.run_backfill(
        None, storage, sat, ts, "proc", max_minutes=10, concurrency=1, mode="process", force=True
    )
    assert s["mode"] == "process" and s["done"] == 3
    assert all(f for _, f in calls)
    assert storage.public.get_json("index.json")["sources"]["msg-0deg"]["regions"] == {
        "reykjanes": {"months": []}
    }


def test_backfill_only_stored(monkeypatch):
    from tests.test_backfill import FakeStorage
    from thermal.storage import raw_key

    storage = FakeStorage()
    ts = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(3)]
    stored = raw_key("msg-0deg", ts[1], "l15")
    # FakeStorage shares one bucket object for private and public
    storage.private.list_keys = lambda prefix, max_keys=None: (
        [stored] if prefix.startswith("raw/") else []
    )
    sat = {
        "key": "msg-0deg",
        "cadence_minutes": 15,
        "collection": "c",
        "tailor_product": "p",
        "monitoring_regions": [],
    }
    calls = []
    monkeypatch.setattr(
        live,
        "process_timestamp",
        lambda c, s_, sat_, t, cache, out, force=False: (calls.append(t), {"ok": True})[1],
    )
    monkeypatch.setattr(live, "ModelCache", lambda s_, d: None)
    s = backfill.run_backfill(
        None, storage, sat, ts, "p", max_minutes=5, concurrency=1, mode="process", only_stored=True
    )
    assert calls == [ts[1]] and s["skipped_unstored"] == 2 and s["done"] == 1
