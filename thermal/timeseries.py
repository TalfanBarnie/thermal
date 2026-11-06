"""Monitoring-region time series: one JSON object per line, sharded by month in the public
bucket, plus the index.json the web app reads first."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from . import geo
from .storage import Storage

# Wooster et al. (2003) MIR method: VRP [W] = 1.89e7 * A_pixel [m2] * dL [W m-2 sr-1 um-1].
# SEVIRI radiances are in mW m-2 sr-1 (cm-1)-1; converting per-wavenumber to per-micron at
# the IR3.9 central wavenumber multiplies by nu_c^2 * 1e-4, and mW -> W by 1e-3. [verify]
MIR_METHOD_CONSTANT = 1.89e7


def radiance_per_cm_to_per_um(delta_l_mw: float | np.ndarray, nu_c: float = 2547.771):
    return delta_l_mw * 1e-3 * nu_c**2 * 1e-4


def record(
    sat: dict[str, Any],
    t: datetime,
    platform: str,
    model_id: str,
    region: str,
    region_mask: np.ndarray,
    images: dict[str, np.ndarray],
    fs,
    flags: np.ndarray,
    z_threshold: float = 4.0,
) -> dict[str, Any]:
    """One time-series row for one region."""
    m = region_mask & fs.valid
    resid = images["residual"][m]
    z = images["zscore"][m]
    area = geo.pixel_area_m2(fs.lat, fs.lon)[m]
    pos = np.clip(resid, 0, None)
    sza = fs.to_image(fs.X[:, fs.names.index("sza")])[m]
    fl = flags[m]
    row = {
        "time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sat": sat["key"],
        "platform": platform,
        "region": region,
        "model": model_id,
        "n_valid": int(m.sum()),
        "sum_pos": _f(np.nansum(pos)),
        "max": _f(np.nanmax(resid)) if resid.size else None,
        "max_z": _f(np.nanmax(z)) if z.size else None,
        "n_above": int(np.nansum(z > z_threshold)),
        "mean": _f(np.nanmean(resid)) if resid.size else None,
        "vrp_w": _f(MIR_METHOD_CONSTANT * np.nansum(area * radiance_per_cm_to_per_um(pos))),
        "n_sat": int(np.sum(fl & 2 > 0)),
        "n_cloud": int(np.sum(fl & 4 > 0)),
        "n_cloud_unknown": int(np.sum(fl & 8 > 0)),
        "sza_mean": _f(np.nanmean(sza)) if sza.size else None,
    }
    return row


def _f(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def shard_key(sat: str, region: str, t: datetime) -> str:
    return f"timeseries/{sat}/{region}/{t:%Y-%m}.jsonl"


def read_shard(storage: Storage, key: str) -> list[dict[str, Any]]:
    try:
        text = storage.public.get_bytes(key).decode()
    except Exception as exc:  # NoSuchKey or similar
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound") or "NoSuchKey" in type(exc).__name__:
            return []
        raise
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_shard(storage: Storage, key: str, rows: list[dict[str, Any]]) -> None:
    body = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n"
    storage.public.put_bytes(key, body.encode(), content_type="application/x-ndjson")


def upsert_rows(storage: Storage, rows: list[dict[str, Any]]) -> list[str]:
    """Merge rows into their monthly shards (replacing rows with the same time). Returns keys."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        t = datetime.strptime(r["time"], "%Y-%m-%dT%H:%M:%SZ")
        by_key.setdefault(shard_key(r["sat"], r["region"], t), []).append(r)
    written = []
    for key, new in by_key.items():
        existing = {r["time"]: r for r in read_shard(storage, key)}
        for r in new:
            existing[r["time"]] = r
        write_shard(storage, key, [existing[k] for k in sorted(existing)])
        written.append(key)
    return written


def build_index(storage: Storage, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """index.json: per source, the regions and the monthly shards that exist."""
    index: dict[str, Any] = {"generated": datetime.now(UTC).isoformat(), "sources": {}}
    for key, sat in sources.items():
        regions = {}
        for region in sat.get("monitoring_regions", []):
            prefix = f"timeseries/{key}/{region}/"
            shards = sorted(
                k.rsplit("/", 1)[-1].removesuffix(".jsonl")
                for k in storage.public.list_keys(prefix)
                if k.endswith(".jsonl")
            )
            regions[region] = {"months": shards}
        index["sources"][key] = {
            "description": sat.get("description", ""),
            "cadence_minutes": sat.get("cadence_minutes"),
            "regions": regions,
            "products_prefix": f"products/{key}/",
        }
    storage.public.put_json("index.json", index)
    return index
