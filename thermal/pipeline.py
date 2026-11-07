"""Per-timestamp processing steps shared by the CLI, the live workflow and the backfill."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import inspect, regions, render
from .redact import redact
from .storage import Storage, fetch_report_key, quicklook_key, raw_key

log = logging.getLogger(__name__)


def kinds_for(sat: dict[str, Any], cloudmask: bool = True) -> list[tuple[str, str, str]]:
    """(kind, collection id, Data Tailor product) tuples to fetch for a source."""
    kinds = [("l15", sat["collection"], sat["tailor_product"])]
    if cloudmask and sat.get("cloudmask_collection"):
        kinds.append(("clm", sat["cloudmask_collection"], sat["tailor_cloudmask_product"]))
    return kinds


def already_stored(storage: Storage, sat: dict[str, Any], t: datetime, cloudmask=True) -> bool:
    return all(
        storage.private.exists(raw_key(sat["key"], t, k)) for k, _, _ in kinds_for(sat, cloudmask)
    )


def fetch_timestamp(
    client: Any,
    storage: Storage | None,
    sat: dict[str, Any],
    t: datetime,
    out_dir: Path,
    *,
    cloudmask: bool = True,
    force: bool = False,
    quicklook: bool = False,
    summarise: bool = True,
    retries: int = 2,
    retry_wait_s: float = 30.0,
) -> dict[str, Any]:
    """Fetch (or reuse) the cutouts for one nominal time. Returns the fetch report.

    The report has report["ok"] False if any kind failed; per-kind errors are in
    report["products"][kind]["error"]. Raw cutouts are stored in the private bucket
    and quicklooks (if requested) in the public bucket.
    """
    roi = regions.box_nswe()
    report: dict[str, Any] = {
        "sat": sat["key"],
        "time": t.isoformat(),
        "roi_nswe": roi,
        "run_at": datetime.now(tz=t.tzinfo).isoformat(),
        "products": {},
        "ok": True,
    }
    for kind, coll_id, tailor_product in kinds_for(sat, cloudmask):
        entry: dict[str, Any] = {"collection": coll_id, "tailor_product": tailor_product}
        report["products"][kind] = entry
        try:
            key = raw_key(sat["key"], t, kind)
            nc_path: Path | None = None
            if storage is not None and not force and storage.private.exists(key):
                nc_path = out_dir / kind / key.rsplit("/", 1)[-1]
                storage.private.get_file(key, nc_path)
                entry["reused_key"] = key
                log.info("%s %s: reusing %s", sat["key"], t.isoformat(), key)
            else:
                nc_path = _tailor_with_retries(
                    client,
                    sat,
                    t,
                    coll_id,
                    tailor_product,
                    roi,
                    out_dir / kind,
                    entry,
                    retries,
                    retry_wait_s,
                )
                if storage is not None:
                    storage.private.put_file(key, nc_path, content_type="application/x-netcdf")
                    entry["stored_key"] = key
                    log.info("stored %s", key)
            entry["local_path"] = str(nc_path)
            if summarise:
                entry["summary"] = inspect.summarise(nc_path)
            if quicklook:
                variables = (
                    list(sat.get("variable_map", {}))
                    if kind == "l15"
                    else [sat.get("cloudmask_variable", "cloud_mask")]
                )
                ql = render.quicklooks(nc_path, out_dir / "quicklook" / kind, variables)
                entry["quicklooks"] = ql
                if storage is not None:
                    for var, info in ql.items():
                        qkey = quicklook_key(sat["key"], t, kind, var)
                        storage.public.put_file(qkey, Path(info["path"]), content_type="image/png")
                        info["public_url"] = storage.public_url_for(qkey)
        except Exception as exc:
            report["ok"] = False
            entry["error"] = redact(f"{type(exc).__name__}: {exc}")
            log.exception("%s %s %s failed", sat["key"], t.isoformat(), kind)
    if storage is not None:
        storage.private.put_json(fetch_report_key(sat["key"], t), report)
    return report


def _tailor_with_retries(
    client, sat, t, coll_id, tailor_product, roi, out_dir, entry, retries, retry_wait_s
) -> Path:
    product = client.product_at(
        coll_id, t, sat["cadence_minutes"], platform=sat.get("platform_filter")
    )
    if product is None:
        raise RuntimeError(f"no product found in {coll_id} near {t.isoformat()}")
    entry["product"] = client.product_summary(product)
    last_exc: Exception | None = None
    cap_wait_s = 900.0  # how long to keep waiting for a free Data Tailor slot
    for attempt in range(retries + 1):
        try:
            waited = 0.0
            while True:
                try:
                    result = client.tailor(product, tailor_product, roi, out_dir)
                    break
                except Exception as exc:
                    # The account's queued+running cap is a transient condition, not a failure:
                    # wait for a slot instead of spending a retry on it.
                    if "exceeding your maximum number" in str(exc) and waited < cap_wait_s:
                        time.sleep(20.0)
                        waited += 20.0
                        continue
                    raise
            entry["customisation"] = {
                "id": result.customisation_id,
                "duration_s": round(result.duration_s, 1),
                "attempt": attempt + 1,
            }
            nc = [p for p in result.outputs if p.suffix == ".nc"]
            if not nc:
                raise RuntimeError(f"no netCDF output among {[p.name for p in result.outputs]}")
            return nc[0]
        except Exception as exc:  # Data Tailor is flaky; retry a couple of times
            last_exc = exc
            if "403 Client Error" in str(exc):
                raise RuntimeError(
                    f"Data Store refused the download of {product} (403): the account is not "
                    "entitled to this product, most likely a near-real-time embargo"
                ) from exc
            log.warning("tailor attempt %d failed for %s: %s", attempt + 1, product, exc)
            if attempt < retries:
                time.sleep(retry_wait_s * (attempt + 1))
    raise RuntimeError(f"Data Tailor failed after {retries + 1} attempts: {last_exc}")
