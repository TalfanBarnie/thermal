"""EUMETSAT Data Store search and Data Tailor region cutouts via eumdac."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import config
from .config import Settings
from .redact import redact

log = logging.getLogger(__name__)

TAILOR_DONE = {"DONE"}
TAILOR_FAILED = {"FAILED", "INACTIVE", "KILLED"}


@dataclass
class TailorResult:
    product_id: str
    customisation_id: str
    outputs: list[Path]
    duration_s: float
    log: str


def nominal_time(sensing_end: datetime, cadence_minutes: int) -> datetime:
    """Round a sensing end time to the nearest repeat cycle; used as the product timestamp."""
    t = sensing_end.astimezone(UTC) if sensing_end.tzinfo else sensing_end
    step = cadence_minutes * 60
    seconds = (t - t.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    rounded = int(round(seconds / step)) * step
    return t.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC) + timedelta(
        seconds=rounded
    )


def wait_for_customisation(
    customisation: Any, poll_seconds: float, timeout_seconds: float, sleep=time.sleep
) -> str:
    """Poll a Data Tailor customisation until it finishes. Returns the final status."""
    deadline = time.monotonic() + timeout_seconds
    status = customisation.status
    while status not in TAILOR_DONE | TAILOR_FAILED:
        if time.monotonic() > deadline:
            raise TimeoutError(f"customisation {customisation} still {status} after timeout")
        sleep(poll_seconds)
        status = customisation.status
    return status


class DataStoreClient:
    def __init__(self, settings: Settings):
        import eumdac

        self._eumdac = eumdac
        self.token = eumdac.AccessToken((settings.eumetsat_key, settings.eumetsat_secret))
        self.datastore = eumdac.DataStore(self.token)
        self.datatailor = eumdac.DataTailor(self.token)
        self._pipeline = config.pipeline()

    # -- diagnostics -------------------------------------------------------------------------
    def whoami(self) -> dict[str, Any]:
        return {
            "token_expires": str(self.token.expiration),
            "datatailor_info": dict(self.datatailor.info),
            "datatailor_quota": dict(self.datatailor.quota),
        }

    def collection_info(self, collection_id: str) -> dict[str, Any]:
        coll = self.datastore.get_collection(collection_id)
        return {
            "id": collection_id,
            "title": coll.title,
            "search_options": {k: v.get("options") for k, v in coll.search_options.items()},
        }

    # -- search ------------------------------------------------------------------------------
    def search(self, collection_id: str, start: datetime, end: datetime, **extra) -> list[Any]:
        """Products in [start, end], in search order (newest first). Reading a product's
        metadata properties may cost one request each, so callers keep windows small."""
        coll = self.datastore.get_collection(collection_id)
        return list(coll.search(dtstart=start, dtend=end, **extra))

    @staticmethod
    def platform_from_id(product: Any) -> str:
        """Platform short name from the product id, e.g. MSG3-SEVI-... -> MSG3 (no request)."""
        return str(product).split("-", 1)[0]

    def product_at(self, collection_id: str, t: datetime, cadence_minutes: int, platform=None):
        """Product whose nominal time is t, searching +-tolerance around it."""
        tol = timedelta(minutes=self._pipeline["search"]["tolerance_minutes"])
        candidates = self.search(collection_id, t - tol, t + tol)
        if platform:
            candidates = [p for p in candidates if p.satellite == platform]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(nominal_time(p.sensing_end, cadence_minutes) - t))

    @staticmethod
    def product_summary(p: Any) -> dict[str, Any]:
        return {
            "id": str(p),
            "collection": str(p.collection),
            "satellite": p.satellite,
            "instrument": p.instrument,
            "sensing_start": p.sensing_start.isoformat(),
            "sensing_end": p.sensing_end.isoformat(),
            "size_kb": p.size,
        }

    def download_access(self, product: Any) -> int:
        """HTTP status of a (streamed, immediately closed) direct download of a product.

        200 means the account may download it; 403 means it is not entitled (e.g. an
        embargo on near-real-time data). No Data Tailor customisation is created.
        """
        import requests

        url = self.datastore.urls.get(
            "datastore",
            "download product",
            vars={"product_id": product._id, "collection_id": product.collection._id},
        )
        with requests.get(url, auth=self.token.auth, stream=True, timeout=60) as resp:
            return resp.status_code

    # -- Data Tailor -------------------------------------------------------------------------
    def tailor(
        self,
        product: Any,
        tailor_product: str,
        roi_nswe: list[float],
        out_dir: Path,
        fmt: str | None = None,
        projection: str | None = None,
    ) -> TailorResult:
        """Run one Data Tailor customisation and download its outputs into out_dir.

        projection=None keeps the native (geostationary) grid; only the region is cut.
        The customisation is deleted afterwards to free Data Tailor workspace.
        """
        dt_cfg = self._pipeline["datatailor"]
        chain = self._eumdac.tailor_models.Chain(
            product=tailor_product,
            format=fmt or dt_cfg["format"],
            projection=projection,
            roi=self._eumdac.tailor_models.RegionOfInterest(NSWE=roi_nswe),
        )
        log.info("submitting Data Tailor chain %s for %s", chain.asdict(), product)
        t0 = time.monotonic()
        customisation = self.datatailor.new_customisation(product, chain)
        try:
            status = wait_for_customisation(
                customisation, dt_cfg["poll_seconds"], dt_cfg["timeout_seconds"]
            )
            try:
                logtext = redact(customisation.logfile)
            except Exception as exc:  # log retrieval is best-effort
                logtext = redact(f"<log unavailable: {exc}>")
            if status in TAILOR_FAILED:
                raise RuntimeError(
                    f"Data Tailor customisation {customisation} {status}:\n{logtext}"
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for output in customisation.outputs:
                target = out_dir / output.split("/")[-1]
                with customisation.stream_output(output) as stream, open(target, "wb") as fh:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        fh.write(chunk)
                paths.append(target)
                log.info("downloaded %s (%d bytes)", target.name, target.stat().st_size)
            return TailorResult(
                product_id=str(product),
                customisation_id=str(customisation),
                outputs=paths,
                duration_s=time.monotonic() - t0,
                log=logtext,
            )
        finally:
            try:
                customisation.delete()
            except Exception as exc:
                log.warning("could not delete customisation %s: %s", customisation, exc)
