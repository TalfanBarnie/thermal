"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

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


def cmd_predict(args) -> int:
    """Apply the latest (or given) background model to one timestamp: residual images and stats."""
    from . import anomaly, features, render
    from .model import BackgroundModel
    from .storage import quicklook_key

    settings = config.load_settings()
    sat = config.satellite(args.sat)
    t = parse_time(args.time)
    storage = Storage(settings)
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="thermal-"))
    report = pipeline.fetch_timestamp(_client(settings), storage, sat, t, out_dir, summarise=False)
    if not report["ok"]:
        print(json.dumps(report, indent=1, default=str))
        return 1
    l15 = out_dir / "l15" / raw_key_name(sat["key"], t, "l15")
    fs = features.build_features(l15, t, sat)
    platform = args.platform or fs.platform
    if args.model:
        model_path = Path(args.model)
    else:
        pointer = storage.private.get_json(f"models/{sat['key']}/{platform}/latest.json")
        if not pointer:
            raise SystemExit(f"no model for {sat['key']}/{platform}")
        model_path = out_dir / "model.npz"
        storage.private.get_file(pointer["key"], model_path)
    model = BackgroundModel.load(model_path)
    images = anomaly.predict_images(model, fs)
    cloud = None
    clm = out_dir / "clm" / raw_key_name(sat["key"], t, "clm")
    if clm.exists():
        import xarray as xr

        with xr.open_dataset(clm, decode_times=False) as ds:
            cloud = np.asarray(ds[sat.get("cloudmask_variable", "cloud_mask")].values, dtype=float)
    stats = [
        anomaly.region_stats(images, fs, r, cloud_mask=cloud) for r in anomaly.default_regions(sat)
    ]
    result = {
        "sat": sat["key"],
        "time": t.isoformat(),
        "platform": fs.platform,
        "model": model.meta.get("model_type"),
        "model_path": str(model_path),
        "regions": stats,
        "quicklooks": {},
    }
    ql_dir = out_dir / "quicklook" / "predict"
    scales = {"residual": (-0.05, 0.3), "zscore": (-5, 20), "observed": None, "predicted": None}
    for name, (vmin, vmax) in ((k, v if v else (None, None)) for k, v in scales.items()):
        img = images[name]
        arr = img[::-1] if np.nanmean(fs.lat[0]) < np.nanmean(fs.lat[-1]) else img
        path = ql_dir / f"{name}.png"
        lo, hi = render.to_png(arr, path, vmin=vmin, vmax=vmax)
        result["quicklooks"][name] = {"path": str(path), "vmin": lo, "vmax": hi}
        if not args.no_upload:
            key = quicklook_key(sat["key"], t, "predict", name)
            storage.public.put_file(key, path, content_type="image/png")
            result["quicklooks"][name]["public_url"] = storage.public_url_for(key)
    print(json.dumps(result, indent=1, default=str))
    (out_dir / f"predict_{sat['key']}_{t:%Y%m%dT%H%M%S}.json").write_text(
        json.dumps(result, indent=1, default=str)
    )
    return 0


def cmd_process(args) -> int:
    """Fetch, model and publish products and time series rows for one timestamp."""
    from .live import ModelCache, process_timestamp

    settings = config.load_settings()
    sat = config.satellite(args.sat)
    storage = Storage(settings)
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="thermal-"))
    cache = ModelCache(storage, out_dir / "models")
    result = process_timestamp(
        _client(settings), storage, sat, parse_time(args.time), cache, out_dir, force=args.force
    )
    print(json.dumps(result, indent=1, default=str))
    return 0 if result.get("ok") else 1


def cmd_live(args) -> int:
    """Catch up on new timestamps since the stored state and publish products."""
    from .live import run_live

    settings = config.load_settings()
    storage = Storage(settings)
    client = _client(settings)
    rc = 0
    for key in args.sats:
        sat = config.satellite(key)
        summary = run_live(
            client,
            storage,
            sat,
            lookback_hours=args.lookback_hours,
            max_timestamps=args.max_timestamps,
            force=args.force,
        )
        print(json.dumps(summary, indent=1, default=str))
        if summary["errors"]:
            rc = 1
    return rc


def raw_key_name(sat_key: str, t: datetime, kind: str) -> str:
    from .storage import raw_key

    return raw_key(sat_key, t, kind).rsplit("/", 1)[-1]


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
            products = client.search(sat["collection"], t, t + timedelta(hours=1))
            platforms = sorted({client.platform_from_id(p) for p in products})
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


def cmd_probe(args) -> int:
    """Check direct download access to image products of various ages (embargo finder)."""
    settings = config.load_settings()
    sat = config.satellite(args.sat)
    client = _client(settings)
    now = datetime.now(UTC)
    print(
        "age_hours  nominal_time          product                                            status"
    )
    for age in args.ages:
        t = now - timedelta(hours=age)
        try:
            p = client.product_at(sat["collection"], t, sat["cadence_minutes"])
            if p is None:
                print(f"{age:>8.1f}  {t:%Y-%m-%dT%H:%M}Z  (no product found)")
                continue
            status = client.download_access(p)
            print(f"{age:>8.1f}  {t:%Y-%m-%dT%H:%M}Z  {str(p):<50} {status}")
        except Exception as exc:
            print(f"{age:>8.1f}  {t:%Y-%m-%dT%H:%M}Z  error: {exc}")
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


def cmd_validate(args) -> int:
    """Onset detection and false alarms against the eruption calendar, from public shards."""
    from . import validate
    from .calendar import Calendar

    public_url = args.public_url or config.load_settings().b2_public_url
    cal = Calendar.from_file()
    rows = validate.load_rows(public_url, args.sat, args.region)
    print(f"{len(rows)} samples for {args.sat}/{args.region}")
    report = validate.onset_report(rows, cal.eruptions, stat=args.stat, threshold=args.threshold)
    print(f"\nOnsets ({args.stat} >= {args.threshold}):")
    widths = (32, 10, 5, 5, 20, 8, 5, 9)
    head = ("volcano", "start", "n", "hits", "first hit", "delay h", "early", "peak")
    print(
        " ".join(
            h.ljust(w) if i < 2 or i == 4 else h.rjust(w)
            for i, (h, w) in enumerate(zip(head, widths, strict=True))
        )
    )
    for r in report:
        print(
            f"{r['volcano'][:32]:32} {r['start']:10} {r['n_samples']:5d} {r['n_hits']:5d} "
            f"{r['first_hit']:20} {str(r['delay_hours']):>8} {r['n_hits_before_start']:5d} "
            f"{str(r['peak'])[:9]:>9}"
        )
    fa = validate.false_alarms(rows, cal, stat=args.stat, threshold=args.threshold)
    print("\nOutside eruption windows:", json.dumps(fa, indent=1))
    if args.export:
        paths = validate.export_eruption_series(rows, cal.eruptions, Path(args.export))
        print(f"\nexported {len(paths)} eruption series to {args.export}")
    return 0


def cmd_eruption_windows(args) -> int:
    """Write the timestamps covering every eruption window (plus margins) to a CSV."""
    from .calendar import load

    sat = config.satellite(args.sat)
    ts = sampling.eruption_windows(
        load(),
        since=parse_time(args.since) if args.since else None,
        margin_days=args.margin_days,
        step_minutes=args.step_minutes,
        onset_step_minutes=args.onset_step_minutes,
        onset_days=args.onset_days,
        cadence_minutes=sat["cadence_minutes"],
    )
    sampling.write_list(Path(args.out), ts, sat["key"])
    print(f"wrote {len(ts)} timestamps to {args.out}")
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
        mode=args.mode,
        force=args.force,
    )
    print(json.dumps(summary, indent=1))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary))
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

    pr = sub.add_parser("predict", help="apply the background model to one timestamp")
    pr.add_argument("--sat", default="msg-0deg")
    pr.add_argument("--time", required=True)
    pr.add_argument("--platform", help="model platform (default: the cutout's platform)")
    pr.add_argument("--model", help="local .npz instead of the bucket's latest model")
    pr.add_argument("--out")
    pr.add_argument("--no-upload", action="store_true")
    pr.set_defaults(func=cmd_predict)

    pc = sub.add_parser("process", help="fetch, model and publish one timestamp")
    pc.add_argument("--sat", default="msg-0deg")
    pc.add_argument("--time", required=True)
    pc.add_argument("--out")
    pc.add_argument("--force", action="store_true", help="re-process even if products exist")
    pc.set_defaults(func=cmd_process)

    lv = sub.add_parser("live", help="process new timestamps since the stored state")
    lv.add_argument("--sats", nargs="*", default=["msg-0deg"])
    lv.add_argument("--lookback-hours", type=float, default=6.0)
    lv.add_argument("--max-timestamps", type=int, default=12)
    lv.add_argument("--min-age-minutes", type=float, default=None, help="default from config")
    lv.add_argument("--force", action="store_true")
    lv.set_defaults(func=cmd_live)

    i = sub.add_parser("inspect", help="summarise local cutout files")
    i.add_argument("files", nargs="+")
    i.set_defaults(func=cmd_inspect)

    pl = sub.add_parser("platforms", help="list which platform served a source over time")
    pl.add_argument("--sat", default="msg-0deg")
    pl.add_argument("--start", default="2004-01-15")
    pl.add_argument("--end", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    pl.add_argument("--step-days", type=int, default=30)
    pl.set_defaults(func=cmd_platforms)

    pb = sub.add_parser("probe", help="check download access to products by age")
    pb.add_argument("--sat", default="msg-0deg")
    pb.add_argument(
        "--ages", type=float, nargs="*", default=[0.5, 1, 2, 3, 4, 6, 12, 24, 48, 72, 168]
    )
    pb.set_defaults(func=cmd_probe)

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
    b.add_argument("--summary-json", help="write the run summary to this file")
    b.add_argument(
        "--mode",
        choices=["fetch", "process"],
        default="fetch",
        help="fetch raw cutouts only, or also model and publish products",
    )
    b.add_argument("--force", action="store_true", help="process: re-make existing products")
    b.set_defaults(func=cmd_backfill)

    va = sub.add_parser("validate", help="onset detection against the calendar from public shards")
    va.add_argument("--sat", default="msg-0deg")
    va.add_argument("--region", default="reykjanes")
    va.add_argument("--stat", default="max_z")
    va.add_argument("--threshold", type=float, default=4.0)
    va.add_argument("--public-url", help="default: B2_PUBLIC_URL from the environment")
    va.add_argument("--export", help="directory for per-eruption CSV exports")
    va.set_defaults(func=cmd_validate)

    ew = sub.add_parser("eruption-windows", help="timestamps covering the eruption calendar")
    ew.add_argument("--sat", default="msg-0deg")
    ew.add_argument("--since", default="2021-01-01", help="ignore eruptions ending before this")
    ew.add_argument("--margin-days", type=float, default=2.0)
    ew.add_argument("--step-minutes", type=int, default=180)
    ew.add_argument("--onset-step-minutes", type=int, default=60)
    ew.add_argument("--onset-days", type=float, default=3.0)
    ew.add_argument("--out", required=True)
    ew.set_defaults(func=cmd_eruption_windows)
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
