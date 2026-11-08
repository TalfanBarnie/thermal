"""Resumable batch fetching of many timestamps through the Data Tailor.

State lives in the private bucket at state/backfill/<job_id>.json so that repeated
workflow dispatches continue where the previous run stopped. Timestamps whose raw
cutouts already exist are skipped without touching the Data Tailor.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import pipeline, timeseries
from .redact import redact
from .storage import Storage, raw_key

log = logging.getLogger(__name__)


def backfill_state_key(job_id: str) -> str:
    return f"state/backfill/{job_id}.json"


class BackfillState:
    def __init__(self, storage: Storage, job_id: str):
        self.storage = storage
        self.key = backfill_state_key(job_id)
        self.data: dict[str, Any] = storage.private.get_json(self.key, default=None) or {
            "job_id": job_id,
            "done": {},
            "failed": {},
            "runs": [],
        }
        self._lock = threading.Lock()

    def is_done(self, ts: str) -> bool:
        return ts in self.data["done"]

    def attempts(self, ts: str) -> int:
        return self.data["failed"].get(ts, {}).get("attempts", 0)

    def mark_done(self, ts: str, info: dict[str, Any]) -> None:
        with self._lock:
            self.data["done"][ts] = info
            self.data["failed"].pop(ts, None)

    def mark_failed(self, ts: str, error: str) -> None:
        error = redact(error)
        with self._lock:
            entry = self.data["failed"].setdefault(ts, {"attempts": 0})
            entry["attempts"] += 1
            entry["error"] = error[:500]
            entry["at"] = datetime.now(UTC).isoformat()
        log.warning("%s failed (attempt %d): %s", ts, entry["attempts"], error[:300])

    def save(self) -> None:
        with self._lock:
            self.storage.private.put_json(self.key, self.data)

    def summary(self) -> dict[str, int]:
        return {"done": len(self.data["done"]), "failed": len(self.data["failed"])}

    def error_kinds(self, limit: int = 8) -> dict[str, int]:
        """Count of still-failed timestamps by error message (truncated), most common first."""
        counts: dict[str, int] = {}
        for entry in self.data["failed"].values():
            kind = str(entry.get("error", ""))[:120]
            counts[kind] = counts.get(kind, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return dict(ranked[:limit])


def run_backfill(
    client: Any,
    storage: Storage,
    sat: dict[str, Any],
    timestamps: list[datetime],
    job_id: str,
    *,
    max_minutes: float = 330.0,
    concurrency: int = 2,
    max_attempts: int = 3,
    cloudmask: bool = True,
    save_every: int = 5,
    mode: str = "fetch",
    force: bool = False,
    sat_config: dict[str, Any] | None = None,
    grace_minutes: float = 10.0,
    only_stored: bool = False,
) -> dict[str, Any]:
    """Run every timestamp not yet done, within a wall-clock budget. Returns a summary.

    mode "fetch": store raw cutouts only. mode "process": fetch (or reuse), apply the
    background model, and publish products and time series rows; with force=True existing
    products are re-made (used after a retrain). With only_stored=True, timestamps whose
    image cutout is not yet in the bucket are left for later instead of fetched (no Data
    Tailor use at all). State is per job id and resumable.
    """
    if mode not in ("fetch", "process"):
        raise ValueError(f"unknown mode {mode}")
    state = BackfillState(storage, job_id)
    model_cache = None
    if mode == "process":
        from .live import ModelCache

        model_cache = ModelCache(storage, Path(tempfile.mkdtemp(prefix="models-")))
    deadline = time.monotonic() + max_minutes * 60.0
    # No wait, retry or poll continues past this: the slice must end before the job timeout.
    hard_deadline = deadline + grace_minutes * 60.0
    started = datetime.now(UTC)
    todo = [
        t
        for t in timestamps
        if not state.is_done(t.strftime("%Y%m%dT%H%M%S"))
        and state.attempts(t.strftime("%Y%m%dT%H%M%S")) < max_attempts
    ]
    skipped_unstored = 0
    if only_stored:
        have = set(storage.private.list_keys(f"raw/{sat['key']}/"))
        kept = [t for t in todo if raw_key(sat["key"], t, "l15") in have]
        skipped_unstored = len(todo) - len(kept)
        todo = kept
    log.info(
        "backfill %s: %d timestamps, %d to do, budget %.0f min, concurrency %d",
        job_id,
        len(timestamps),
        len(todo),
        max_minutes,
        concurrency,
    )
    # Process in a fixed pseudo-random order so that a partially complete backfill is a
    # representative sample of the whole list rather than its earliest dates.
    import random

    random.Random(job_id).shuffle(todo)
    processed = 0
    stopped_for_time = False
    tmp_root = Path(tempfile.mkdtemp(prefix=f"backfill-{job_id}-"))

    def work(t: datetime) -> tuple[str, dict[str, Any]]:
        ts = t.strftime("%Y%m%dT%H%M%S")
        out_dir = tmp_root / ts
        if mode == "process":
            from .live import process_timestamp

            r = process_timestamp(client, storage, sat, t, model_cache, out_dir, force=force)
            info = {"ok": bool(r.get("ok")), "platform": r.get("platform"), "model": r.get("model")}
            if r.get("skipped"):
                info["skipped"] = r["skipped"]
            if not info["ok"]:
                info["errors"] = r.get("error")
            _cleanup(out_dir)
            return ts, info
        if pipeline.already_stored(storage, sat, t, cloudmask):
            return ts, {"ok": True, "reused": True}
        report = pipeline.fetch_timestamp(
            client,
            storage,
            sat,
            t,
            out_dir,
            cloudmask=cloudmask,
            summarise=False,
            deadline=hard_deadline,
        )
        # In fetch mode the image is what matters; a missing cloud mask is recorded, not fatal.
        l15 = report["products"].get("l15", {})
        info: dict[str, Any] = {"ok": "error" not in l15}
        if "product" in l15:
            info["platform"] = l15["product"].get("satellite")
            info["product_id"] = l15["product"].get("id")
        errors = {k: v.get("error") for k, v in report["products"].items() if "error" in v}
        if errors:
            info["errors"] = errors
        _cleanup(out_dir)
        return ts, info

    abandoned = 0
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        pending = set()
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < concurrency:
                if time.monotonic() > deadline:
                    stopped_for_time = True
                    queue.clear()
                    break
                pending.add(pool.submit(work, queue.pop(0)))
            if not pending:
                break
            remaining_s = hard_deadline + 60.0 - time.monotonic()
            if remaining_s <= 0:
                abandoned = len(pending)
                log.warning(
                    "hard deadline reached with %d timestamps in flight; abandoning", abandoned
                )
                break
            for fut in as_completed(list(pending), timeout=remaining_s):
                pending.discard(fut)
                try:
                    ts, info = fut.result()
                except Exception:  # unexpected crash in worker
                    log.exception("backfill worker crashed")
                    continue
                if info.get("ok"):
                    state.mark_done(ts, info)
                else:
                    state.mark_failed(ts, str(info.get("errors")))
                processed += 1
                if processed % save_every == 0:
                    state.save()
                    log.info(
                        "progress %s: %s, %d this run, %d queued, %d in flight, %.0f min left",
                        job_id,
                        state.summary(),
                        processed,
                        len(queue),
                        len(pending),
                        max(0.0, (deadline - time.monotonic()) / 60.0),
                    )
                break  # go back to topping up the queue
    except TimeoutError:
        abandoned = len(pending)
        log.warning("hard deadline reached with %d timestamps in flight; abandoning", abandoned)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    state.data["runs"].append(
        {
            "started": started.isoformat(),
            "finished": datetime.now(UTC).isoformat(),
            "processed": processed,
            "stopped_for_time": stopped_for_time,
            "abandoned_in_flight": abandoned,
        }
    )
    state.save()
    if mode == "process" and processed:
        timeseries.build_index(storage, {sat["key"]: sat})
    summary = {
        "job_id": job_id,
        "mode": mode,
        "requested": len(timestamps),
        "processed_this_run": processed,
        "stopped_for_time": stopped_for_time,
        "skipped_unstored": skipped_unstored,
        "remaining": len(
            [
                t
                for t in timestamps
                if not state.is_done(t.strftime("%Y%m%dT%H%M%S"))
                and state.attempts(t.strftime("%Y%m%dT%H%M%S")) < max_attempts
            ]
        ),
        **state.summary(),
    }
    if state.data["failed"]:
        summary["error_kinds"] = state.error_kinds()
    log.info("backfill summary: %s", summary)
    return summary


def _cleanup(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
