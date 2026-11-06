"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import config, inspect, pipeline, sampling
from .storage import Storage

log = logging.getLogger("thermal")


def parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _client(settings):
    from .datastore import DataStoreClient

    return DataStoreClient(settings)


def cmd_smoke(args) -> int:
    """Check that every credential works: both buckets and the Data Store."""
    settings = config.load_settings()
    ok = True
    storage = Storage(settings)
    for label, bucket in (("private", storage.private), ("public", storage.public)):
        try:
            probe = f"state/_smoke/{datetime.now(UTC):%Y%m%dT%H%M%S}.txt"
            bucket.put_bytes(probe, b"ok", content_type="text/plain")
            assert probe in bucket.list_keys("state/_smoke/", max_keys=50)
            print(f"[ok] B2 {label} bucket {bucket.name}: write and list succeed")
        except Exception as exc:
            ok = False
            print(f"[FAIL] B2 {label} bucket {bucket.name}: {type(exc).__name__}: {exc}")
    try:
        client = _client(settings)
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
    settings = config.load_settings()
    sat = config.satellite(args.sat)
    t = parse_time(args.time)
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="thermal-"))
    storage = None if args.no_upload else Storage(settings)
    report = pipeline.fetch_timestamp(
        _client(settings),
        storage,
        sat,
        t,
        out_dir,
        cloudmask=not args.no_cloudmask,
        force=args.force,
        quicklook=args.quicklook,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"fetch_{sat['key']}_{t:%Y%m%dT%H%M%S}.json"
    report_path.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print(f"report written to {report_path}")
    return 0 if report["ok"] else 1


def cmd_inspect(args) -> int:
    for path in args.files:
        print(json.dumps(inspect.summarise(Path(path)), indent=1, default=str))
    return 0


def cmd_platforms(args) -> int:
    """Which platform served a source's collection over time (one search per step)."""
    settings = config.load_settings()
    sat = config.satellite(args.sat)
    client = _client(settings)
    t = parse_time(args.start).replace(hour=12, minute=0, second=0, microsecond=0)
    end = parse_time(args.end)
    rows = []
    while t < end:
        try:
            products = client.search(sat["collection"], t, t + timedelta(hours=2))
            platforms = sorted({p.satellite for p in products})
        except Exception as exc:
            platforms = [f"error: {exc}"]
        rows.append((t.date().isoformat(), ",".join(platforms) or "none"))
        print(f"{rows[-1][0]}  {rows[-1][1]}", flush=True)
        t += timedelta(days=args.step_days)
    print("\nTransitions:")
    prev = None
    for d, p in rows:
        if p != prev:
            print(f"  from {d}: {p}")
            prev = p
    return 0


def cmd_sample(args) -> int:
    """Draw a stratified training sample of timestamps, excluding eruption windows."""
    from .calendar import Calendar

    sat = config.satellite(args.sat)
    cal = Calendar.from_file(buffer_days=args.buffer_days)
    ts = sampling.stratified_sample(
        parse_time(args.start),
        parse_time(args.end),
        args.n,
        sat["cadence_minutes"],
        exclude=cal.excluded,
        seed=args.seed,
    )
    out = Path(args.out)
    sampling.write_list(out, ts, sat["key"])
    print(f"wrote {len(ts)} timestamps to {out}")
    return 0


def cmd_backfill(args) -> int:
    """Fetch many timestamps, resumably, within a wall-clock budget."""
    from .backfill import run_backfill

    settings = config.load_settings()
    sat = config.satellite(args.sat)
    if args.list:
        timestamps = [t for s, t in sampling.read_list(Path(args.list)) if s == sat["key"]]
    elif args.start and args.end:
        timestamps = sampling.regular_range(
            parse_time(args.start), parse_time(args.end), args.step_minutes
        )
    else:
        raise SystemExit("backfill needs --list or --start and --end")
    summary = run_backfill(
        _client(settings),
        Storage(settings),
        sat,
        timestamps,
        args.job_id,
        max_minutes=args.max_minutes,
        concurrency=args.concurrency,
        cloudmask=not args.no_cloudmask,
    )
    print(json.dumps(summary, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thermal")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="check credentials for B2 and the EUMETSAT Data Store")
    s.add_argument("--sats", nargs="*", default=["msg-0deg"], help="source keys to check")
    s.set_defaults(func=cmd_smoke)

    f = sub.add_parser("fetch", help="fetch one timestamp through the Data Tailor")
    f.add_argument("--sat", default="msg-0deg")
    f.add_argument("--time", required=True, help="nominal image time, ISO 8601 UTC")
    f.add_argument("--out", help="local output directory (default: temp dir)")
    f.add_argument("--no-upload", action="store_true", help="do not write to the buckets")
    f.add_argument("--no-cloudmask", action="store_true")
    f.add_argument("--force", action="store_true", help="re-run the Data Tailor even if stored")
    f.add_argument("--quicklook", action="store_true", help="render PNG quicklooks")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("inspect", help="summarise local cutout files")
    i.add_argument("files", nargs="+")
    i.set_defaults(func=cmd_inspect)

    pl = sub.add_parser("platforms", help="list which platform served a source over time")
    pl.add_argument("--sat", default="msg-0deg")
    pl.add_argument("--start", default="2004-01-15")
    pl.add_argument("--end", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    pl.add_argument("--step-days", type=int, default=30)
    pl.set_defaults(func=cmd_platforms)

    sm = sub.add_parser("sample", help="draw a stratified training sample of timestamps")
    sm.add_argument("--sat", default="msg-0deg")
    sm.add_argument("--start", required=True)
    sm.add_argument("--end", required=True)
    sm.add_argument("--n", type=int, default=1000)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--buffer-days", type=int, default=30)
    sm.add_argument("--out", required=True, help="CSV path, e.g. data/samples/msg-0deg_train.csv")
    sm.set_defaults(func=cmd_sample)

    b = sub.add_parser("backfill", help="fetch many timestamps resumably")
    b.add_argument("--sat", default="msg-0deg")
    b.add_argument("--job-id", required=True, help="state file name; reuse to resume")
    b.add_argument("--list", help="CSV of sat,time rows (from `thermal sample`)")
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--step-minutes", type=int, default=60)
    b.add_argument("--max-minutes", type=float, default=330.0, help="wall-clock budget")
    b.add_argument("--concurrency", type=int, default=2)
    b.add_argument("--no-cloudmask", action="store_true")
    b.set_defaults(func=cmd_backfill)
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
