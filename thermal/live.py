"""Process timestamps end to end (fetch -> model -> products -> time series) and the hourly
catch-up loop that drives the live service."""

from __future__ import annotations

import logging
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import anomaly, config, features, pipeline, products, regions, timeseries
from .datastore import nominal_time
from .model import BackgroundModel
from .storage import Storage, raw_key, state_key

log = logging.getLogger(__name__)

_ID_TIME = re.compile(r"-(\d{14})\.\d+Z-")


def nominal_time_from_id(product_id: str, cadence_minutes: int) -> datetime | None:
    """Nominal time from the sensing-end timestamp embedded in a product id (no request)."""
    m = _ID_TIME.search(product_id)
    if not m:
        return None
    end = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return nominal_time(end, cadence_minutes)


class ModelCache:
    def __init__(self, storage: Storage, cache_dir: Path):
        self.storage = storage
        self.cache_dir = cache_dir
        self._models: dict[tuple[str, str], tuple[BackgroundModel, str]] = {}

    def get(self, sat_key: str, platform: str) -> tuple[BackgroundModel, str]:
        k = (sat_key, platform)
        if k not in self._models:
            pointer = self.storage.private.get_json(f"models/{sat_key}/{platform}/latest.json")
            if not pointer:
                raise RuntimeError(f"no background model for {sat_key}/{platform}")
            path = self.cache_dir / pointer["key"]
            if not path.exists():
                self.storage.private.get_file(pointer["key"], path)
            self._models[k] = (BackgroundModel.load(path), pointer["model_id"])
        return self._models[k]


def product_exists(storage: Storage, sat: dict[str, Any], t: datetime) -> bool:
    return storage.private.exists(f"{products.product_prefix(sat['key'], t)}/residual.npz")


def process_timestamp(
    client: Any,
    storage: Storage,
    sat: dict[str, Any],
    t: datetime,
    model_cache: ModelCache,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch (or reuse) the cutouts, apply the model, write products and time series rows."""
    result: dict[str, Any] = {"sat": sat["key"], "time": t.isoformat(), "ok": False}
    if not force and product_exists(storage, sat, t):
        result.update(ok=True, skipped="exists")
        return result
    report = pipeline.fetch_timestamp(client, storage, sat, t, out_dir, summarise=False)
    result["fetch_ok"] = report["ok"]
    if "error" in report["products"].get("l15", {}):
        result["error"] = report["products"]["l15"]["error"]
        return result
    l15 = out_dir / "l15" / raw_key(sat["key"], t, "l15").rsplit("/", 1)[-1]
    clm = out_dir / "clm" / raw_key(sat["key"], t, "clm").rsplit("/", 1)[-1]
    fs = features.build_features(l15, t, sat)
    try:
        model, model_id = model_cache.get(sat["key"], fs.platform)
    except RuntimeError as exc:
        result["error"] = str(exc)
        return result
    images = anomaly.predict_images(model, fs)
    arrays, extras = products.build_products(images, fs, l15, clm if clm.exists() else None, sat)
    written = products.write_products(
        storage, sat, t, images, arrays, extras, fs, out_dir / "products", model_id
    )
    rows = []
    for region in sat.get("monitoring_regions", []):
        mask = regions.region_mask(region, fs.lat, fs.lon)
        rows.append(
            timeseries.record(
                sat, t, fs.platform, model_id, region, mask, images, fs, arrays["flags"]
            )
        )
    shards = timeseries.upsert_rows(storage, rows)
    result.update(
        ok=True, platform=fs.platform, model=model_id, products=written, rows=rows, shards=shards
    )
    return result


def candidate_times(
    client: Any, sat: dict[str, Any], start: datetime, end: datetime
) -> list[datetime]:
    """Nominal times of the image products available in [start, end]."""
    out = set()
    for p in client.search(sat["collection"], start, end):
        t = nominal_time_from_id(str(p), sat["cadence_minutes"])
        if t is not None and start <= t <= end:
            out.add(t)
    return sorted(out)


def run_live(
    client: Any,
    storage: Storage,
    sat: dict[str, Any],
    *,
    lookback_hours: float = 6.0,
    max_timestamps: int = 12,
    min_age_minutes: float | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process every new timestamp since the stored state (bounded), then update state and index.

    Only nominal times at least `min_age_minutes` old are considered, because the Data
    Store refuses near-real-time image downloads for this account.
    """
    now = now or datetime.now(UTC)
    if min_age_minutes is None:
        min_age_minutes = config.pipeline().get("live", {}).get("min_age_minutes", 75)
    end = now - timedelta(minutes=min_age_minutes)
    state = storage.private.get_json(state_key(sat["key"]), default={}) or {}
    last = state.get("last_time")
    start = now - timedelta(hours=lookback_hours)
    if last:
        start = max(start, datetime.fromisoformat(last).astimezone(UTC) + timedelta(minutes=1))
    times = candidate_times(client, sat, start, end)
    todo = [t for t in times if force or not product_exists(storage, sat, t)][-max_timestamps:]
    log.info(
        "%s: %d candidates since %s, %d to process",
        sat["key"],
        len(times),
        start.isoformat(),
        len(todo),
    )
    model_cache = ModelCache(storage, Path(tempfile.mkdtemp(prefix="models-")))
    results = []
    for t in todo:
        out_dir = Path(tempfile.mkdtemp(prefix=f"live-{t:%Y%m%dT%H%M}-"))
        try:
            r = process_timestamp(client, storage, sat, t, model_cache, out_dir, force=force)
        except Exception as exc:
            log.exception("%s failed", t.isoformat())
            r = {"time": t.isoformat(), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        if r.get("ok"):
            state["last_time"] = max(state.get("last_time", ""), t.isoformat())
        state.setdefault("runs", []).append(
            {"at": now.isoformat(), "time": t.isoformat(), "ok": r.get("ok")}
        )
        state["runs"] = state["runs"][-200:]
        storage.private.put_json(state_key(sat["key"]), state)
    timeseries.build_index(storage, {sat["key"]: sat})
    return {
        "sat": sat["key"],
        "processed": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "errors": [r for r in results if not r.get("ok")],
        "last_time": state.get("last_time"),
    }
