from datetime import UTC, datetime

from tests.test_storage import FakeClient
from thermal import backfill, pipeline


class FakeStorage:
    def __init__(self):
        from thermal.config import BucketCredentials
        from thermal.storage import Bucket

        self.private = Bucket(BucketCredentials("b", "k", "a"), "e", client=FakeClient())
        self.public = self.private


def test_backfill_resumes_and_records_failures(monkeypatch):
    storage = FakeStorage()
    sat = {
        "key": "msg-0deg",
        "cadence_minutes": 15,
        "collection": "c",
        "tailor_product": "p",
        "cloudmask_collection": "cc",
        "tailor_cloudmask_product": "pp",
    }
    ts = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(4)]
    calls = []

    def fake_fetch(client, storage_, sat_, t, out_dir, **kw):
        calls.append(t)
        ok = t.hour != 2
        return {
            "ok": ok,
            "products": {
                "l15": {
                    "product": {"satellite": "MSG3", "id": "x"},
                    **({} if ok else {"error": "boom"}),
                }
            },
        }

    monkeypatch.setattr(pipeline, "fetch_timestamp", fake_fetch)
    monkeypatch.setattr(pipeline, "already_stored", lambda *a, **k: False)

    s1 = backfill.run_backfill(None, storage, sat, ts, "job", max_minutes=10, concurrency=2)
    assert s1["done"] == 3 and s1["failed"] == 1 and s1["processed_this_run"] == 4
    state = storage.private.get_json("state/backfill/job.json")
    assert state["failed"]["20240101T020000"]["attempts"] == 1
    assert state["done"]["20240101T000000"]["platform"] == "MSG3"

    # second run only retries the failure
    calls.clear()
    s2 = backfill.run_backfill(None, storage, sat, ts, "job", max_minutes=10, concurrency=2)
    assert calls == [ts[2]]
    assert s2["processed_this_run"] == 1
    state = storage.private.get_json("state/backfill/job.json")
    assert state["failed"]["20240101T020000"]["attempts"] == 2
    assert len(state["runs"]) == 2


def test_backfill_time_budget(monkeypatch):
    storage = FakeStorage()
    sat = {"key": "s", "cadence_minutes": 15, "collection": "c", "tailor_product": "p"}
    ts = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(5)]
    monkeypatch.setattr(pipeline, "already_stored", lambda *a, **k: False)
    monkeypatch.setattr(pipeline, "fetch_timestamp", lambda *a, **k: {"ok": True, "products": {}})
    s = backfill.run_backfill(None, storage, sat, ts, "j", max_minutes=0, concurrency=1)
    assert s["stopped_for_time"] and s["processed_this_run"] == 0 and s["remaining"] == 5
