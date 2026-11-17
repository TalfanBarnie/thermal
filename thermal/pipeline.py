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


# Channels rendered as quicklooks by `thermal fetch --quicklook`: visible 0.6 um, the
# 3.9 um target and the 10.8 um window channel. The other channels are not looked at,
# so rendering them only filled the public bucket.
QUICKLOOK_CHANNELS = ("VIS006", "IR_039", "IR_108")


def quicklook_variables(sat: dict[str, Any]) -> list[str]:
    """Cutout variable names of the quicklook channels (a source may override the list)."""
    inv = {ch: var for var, ch in sat.get("variable_map", {}).items()}
    channels = sat.get("quicklook_channels", QUICKLOOK_CHANNELS)
    return [inv[ch] for ch in channels if ch in inv]


def is_native_cutout(path: Path) -> bool:
    """True for a cutout cropped locally from the native file (see thermal.native)."""
    import xarray as xr

    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            return "cropped locally" in str(ds.attrs.get("source", ""))
    except Exception:  # unreadable: treat as stale so that it is fetched again
        return False


def already_stored(storage: Storage, sat: dict[str, Any], t: datetime, cloudmask=True) -> bool:
    """Every cutout is in the bucket. For natively cropped sources this is not decided
    from the key alone (older Data Tailor cutouts may sit under the same key), so it is
    False and the fetch itself checks, and reuses, what is stored."""
    if any(sat.get(f"{k}_method") == "native" for k, _, _ in kinds_for(sat, cloudmask)):
        return False
    return all(
        storage.private.exists(raw_key(sat["key"], t, k)) for k, _, _ in kinds_for(sat, cloudmask)
    )


def _fetch_fresh(
    client: Any,
    sat: dict[str, Any],
    t: datetime,
    kind: str,
    coll_id: str,
    tailor_product: str,
    roi: list[float],
    out_dir: Path,
    entry: dict[str, Any],
    retries: int,
    retry_wait_s: float,
    *,
    deadline: float | None = None,
    image_nc: Path | None = None,
) -> Path:
    """Make one cutout from the Data Store: cropped locally from the native file, or
    through the Data Tailor, as configured for the source and kind."""
    if sat.get(f"{kind}_method") == "native":
        entry["method"] = "native"
        return _native_with_retries(
            client,
            sat,
            t,
            coll_id,
            roi,
            out_dir,
            entry,
            retries,
            retry_wait_s,
            deadline=deadline,
            kind=kind,
            image_nc=image_nc,
        )
    return _tailor_with_retries(
        client, sat, t, coll_id, tailor_product, roi, out_dir, entry, retries, retry_wait_s
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
    deadline: float | None = None,
) -> dict[str, Any]:
    """Fetch (or reuse) the cutouts for one nominal time. Returns the fetch report.

    `deadline` (time.monotonic()) bounds every wait and retry so batch runs end on time.

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
    local_paths: dict[str, Path] = {}
    for kind, coll_id, tailor_product in kinds_for(sat, cloudmask):
        entry: dict[str, Any] = {"collection": coll_id, "tailor_product": tailor_product}
        report["products"][kind] = entry
        try:
            key = raw_key(sat["key"], t, kind)
            nc_path: Path | None = None
            if storage is not None and not force and storage.private.exists(key):
                nc_path = out_dir / kind / key.rsplit("/", 1)[-1]
                storage.private.get_file(key, nc_path)
                if sat.get(f"{kind}_method") == "native" and not is_native_cutout(nc_path):
                    # stored before the native route (e.g. a mis-georeferenced Data
                    # Tailor RSS cutout): fetch it again, overwriting the stored one
                    log.warning(
                        "%s %s: stored %s predates the native route; fetching again",
                        sat["key"],
                        t.isoformat(),
                        key,
                    )
                    entry["replaced_stale"] = key
                    nc_path.unlink(missing_ok=True)
                    nc_path = None
                else:
                    entry["reused_key"] = key
                    log.info("%s %s: reusing %s", sat["key"], t.isoformat(), key)
            if nc_path is None:
                nc_path = _fetch_fresh(
                    client,
                    sat,
                    t,
                    kind,
                    coll_id,
                    tailor_product,
                    roi,
                    out_dir / kind,
                    entry,
                    retries,
                    retry_wait_s,
                    deadline=deadline,
                    image_nc=local_paths.get("l15"),
                )
                if storage is not None:
                    storage.private.put_file(key, nc_path, content_type="application/x-netcdf")
                    entry["stored_key"] = key
                    log.info("stored %s", key)
            entry["local_path"] = str(nc_path)
            local_paths[kind] = nc_path
            if summarise:
                entry["summary"] = inspect.summarise(nc_path)
            if quicklook:
                variables = (
                    quicklook_variables(sat)
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


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() > deadline


def _native_with_retries(
    client,
    sat,
    t,
    coll_id,
    roi,
    out_dir,
    entry,
    retries,
    retry_wait_s,
    deadline=None,
    kind: str = "l15",
    image_nc: Path | None = None,
) -> Path:
    """Download the original product and crop it locally (see thermal/native.py).

    kind "l15": the `.nat` image file; kind "clm": the cloud mask GRIB, put on the grid of
    `image_nc` (the image cutout of the same timestamp) when it is available.
    """
    from . import native

    product = client.product_at(
        coll_id, t, sat["cadence_minutes"], platform=sat.get("platform_filter")
    )
    if product is None:
        raise RuntimeError(f"no product found in {coll_id} near {t.isoformat()}")
    entry["product"] = client.product_summary(product)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        nat: Path | None = None
        try:
            if _past(deadline):
                raise TimeoutError("batch deadline reached before the download started")
            t0 = time.monotonic()
            out_nc = out_dir / f"{product}_cutout.nc"
            if kind == "clm":
                nat = native.download_cloud_mask(product, out_dir / "native")
                t1 = time.monotonic()
                nc = native.crop_cloud_mask(nat, roi, out_nc, f"{product}.grb", image_nc=image_nc)
            else:
                nat = native.download_native(product, out_dir / "native")
                t1 = time.monotonic()
                nc = native.crop_native(nat, roi, out_nc, f"{product}.nat")
            entry["native"] = {
                "attempt": attempt + 1,
                "file_bytes": nat.stat().st_size,
                "download_s": round(t1 - t0, 1),
                "crop_s": round(time.monotonic() - t1, 1),
            }
            return nc
        except Exception as exc:
            last_exc = exc
            if "403" in str(exc):
                raise RuntimeError(
                    f"Data Store refused the download of {product} (403): the account is not "
                    "entitled to this product, most likely a near-real-time embargo"
                ) from exc
            log.warning("native attempt %d failed for %s: %s", attempt + 1, product, exc)
            if _past(deadline):
                break
            if attempt < retries:
                time.sleep(retry_wait_s * (attempt + 1))
        finally:
            if nat is not None:
                nat.unlink(missing_ok=True)  # the full file is never kept
    raise RuntimeError(f"native download failed after {retries + 1} attempts: {last_exc}")


def _tailor_with_retries(
    client,
    sat,
    t,
    coll_id,
    tailor_product,
    roi,
    out_dir,
    entry,
    retries,
    retry_wait_s,
    deadline=None,
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
                if _past(deadline):
                    raise TimeoutError("batch deadline reached before the customisation ran")
                try:
                    result = client.tailor(product, tailor_product, roi, out_dir, deadline=deadline)
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
            if _past(deadline):
                break
            if attempt < retries:
                time.sleep(retry_wait_s * (attempt + 1))
    raise RuntimeError(f"Data Tailor failed after {retries + 1} attempts: {last_exc}")
