"""Command line entry points: thermal smoke | fetch | inspect."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from . import config, inspect, regions, render
from .storage import Storage, fetch_report_key, quicklook_key, raw_key

log = logging.getLogger("thermal")


def parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def cmd_smoke(args) -> int:
    """Check that every credential works: both buckets and the Data Store."""
    settings = config.load_settings()
    ok = True

    storage = Storage(settings)
    for label, bucket in (("private", storage.private), ("public", storage.public)):
        try:
            probe = f"state/_smoke/{datetime.now(UTC):%Y%m%dT%H%M%S}.txt"
            bucket.put_bytes(probe, b"ok", content_type="text/plain")
            keys = bucket.list_keys("state/_smoke/", max_keys=5)
            assert probe in keys
            print(f"[ok] B2 {label} bucket {bucket.name}: write and list succeed")
        except Exception as exc:
            ok = False
            print(f"[FAIL] B2 {label} bucket {bucket.name}: {type(exc).__name__}: {exc}")

    try:
        from .datastore import DataStoreClient

        client = DataStoreClient(settings)
        info = client.whoami()
        print("[ok] EUMETSAT token obtained; Data Tailor quota:")
        print(json.dumps(info["datatailor_quota"], indent=1, default=str))
        for key in args.sats:
            sat = config.satellite(key)
            for coll_id in (sat["collection"], sat.get("cloudmask_collection")):
                if not coll_id:
                    continue
                try:
                    ci = client.collection_info(coll_id)
                    print(f"[ok] collection {coll_id}: {ci['title']}")
                    print("     search options:", json.dumps(ci["search_options"], default=str))
                except Exception as exc:
                    ok = False
                    print(f"[FAIL] collection {coll_id}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] EUMETSAT Data Store: {type(exc).__name__}: {exc}")

    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_fetch(args) -> int:
    """Fetch one timestamp (image and cloud mask) through the Data Tailor."""
    from .datastore import DataStoreClient

    settings = config.load_settings()
    sat = config.satellite(args.sat)
    t = parse_time(args.time)
    client = DataStoreClient(settings)
    roi = regions.box_nswe()
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="thermal-"))
    report = {
        "sat": sat["key"],
        "time": t.isoformat(),
        "roi_nswe": roi,
        "run_at": datetime.now(UTC).isoformat(),
        "products": {},
    }
    storage = Storage(settings) if not args.no_upload else None
    failures = 0

    kinds = [("l15", sat["collection"], sat["tailor_product"])]
    if sat.get("cloudmask_collection") and not args.no_cloudmask:
        kinds.append(("clm", sat["cloudmask_collection"], sat["tailor_cloudmask_product"]))

    for kind, coll_id, tailor_product in kinds:
        entry: dict = {"collection": coll_id, "tailor_product": tailor_product}
        report["products"][kind] = entry
        try:
            key = raw_key(sat["key"], t, kind)
            nc_path = None
            if storage is not None and not args.force and storage.private.exists(key):
                nc_path = out_dir / kind / key.rsplit("/", 1)[-1]
                storage.private.get_file(key, nc_path)
                entry["reused_key"] = key
                log.info("%s: reusing existing cutout %s", kind, key)
            else:
                product = client.product_at(
                    coll_id, t, sat["cadence_minutes"], platform=sat.get("platform_filter")
                )
                if product is None:
                    raise RuntimeError(f"no product found in {coll_id} near {t.isoformat()}")
                entry["product"] = client.product_summary(product)
                log.info("%s: using %s", kind, entry["product"])
                result = client.tailor(product, tailor_product, roi, out_dir / kind)
                entry["customisation"] = {
                    "id": result.customisation_id,
                    "duration_s": round(result.duration_s, 1),
                    "log": result.log,
                }
                nc = [p for p in result.outputs if p.suffix == ".nc"]
                if not nc:
                    raise RuntimeError(f"no netCDF output among {[p.name for p in result.outputs]}")
                nc_path = nc[0]
                if storage is not None:
                    storage.private.put_file(key, nc_path, content_type="application/x-netcdf")
                    entry["stored_key"] = key
                    log.info("stored %s", key)
            entry["summary"] = inspect.summarise(nc_path)
            if args.quicklook:
                variables = (
                    list(sat.get("variable_map", {}))
                    if kind == "l15"
                    else [sat["cloudmask_variable"]]
                )
                ql = render.quicklooks(nc_path, out_dir / "quicklook" / kind, variables)
                entry["quicklooks"] = ql
                if storage is not None:
                    for var, info in ql.items():
                        qkey = quicklook_key(sat["key"], t, kind, var)
                        storage.public.put_file(qkey, Path(info["path"]), content_type="image/png")
                        info["public_url"] = storage.public_url_for(qkey)
        except Exception as exc:
            failures += 1
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("%s failed", kind)

    if storage is not None:
        storage.private.put_json(fetch_report_key(sat["key"], t), report)
    report_path = out_dir / f"fetch_{sat['key']}_{t:%Y%m%dT%H%M%S}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print(f"report written to {report_path}")
    return 1 if failures else 0


def cmd_inspect(args) -> int:
    for path in args.files:
        print(json.dumps(inspect.summarise(Path(path)), indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thermal")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="check credentials for B2 and the EUMETSAT Data Store")
    s.add_argument("--sats", nargs="*", default=["msg-0deg"], help="satellite keys to check")
    s.set_defaults(func=cmd_smoke)

    f = sub.add_parser("fetch", help="fetch one timestamp through the Data Tailor")
    f.add_argument("--sat", default="msg-0deg")
    f.add_argument("--time", required=True, help="nominal image time, ISO 8601 UTC")
    f.add_argument("--out", help="local output directory (default: temp dir)")
    f.add_argument("--no-upload", action="store_true", help="do not write to the private bucket")
    f.add_argument("--no-cloudmask", action="store_true")
    f.add_argument("--force", action="store_true", help="re-run the Data Tailor even if stored")
    f.add_argument("--quicklook", action="store_true", help="render PNG quicklooks")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("inspect", help="summarise local cutout files")
    i.add_argument("files", nargs="+")
    i.set_defaults(func=cmd_inspect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
