"""Train and evaluate background models for one source/platform.

    python -m training.run --sat msg-0deg --platform MSG3 \\
        --list data/samples/msg-0deg_msg3_train.csv

Writes model artefacts and an evaluation report locally and, unless --no-upload, to the private
bucket under models/<sat>/<platform>/ and reports/<sat>/<platform>/<model_id>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from thermal import config, sampling
from thermal.model import BackgroundModel, save_model
from thermal.storage import Storage

from . import data, evaluate, models

log = logging.getLogger("training")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sat", default="msg-0deg")
    p.add_argument(
        "--platform", required=True, help="e.g. MSG3; cutouts from other platforms are skipped"
    )
    p.add_argument(
        "--list", required=True, help="sample CSV; only timestamps already fetched are used"
    )
    p.add_argument("--models", default="ridge,mlp")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--max-images", type=int, default=0, help="0 = all available")
    p.add_argument("--max-pixels-per-image", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", default="128,128,128")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--cache-dir", default=".cache/raw")
    p.add_argument("--out", default="reports/train")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sat = config.satellite(args.sat)
    storage = None if args.no_upload else Storage(config.load_settings())
    cache = Path(args.cache_dir)
    if storage is not None:
        timestamps = data.available_timestamps(storage, sat, Path(args.list))
    else:
        timestamps = [t for _, t in sampling.read_list(Path(args.list))]
    if args.max_images:
        timestamps = timestamps[: args.max_images]
    log.info("%d timestamps available", len(timestamps))
    table = data.collect(
        storage,
        sat,
        timestamps,
        cache,
        platform=args.platform,
        max_pixels_per_image=args.max_pixels_per_image,
        seed=args.seed,
    )
    log.info(
        "table: %s rows, %d features, %d images",
        table.X.shape[0],
        table.X.shape[1],
        len(table.images),
    )
    train, test = data.holdout_split(table.group, args.holdout_frac, seed=args.seed)
    month_of = np.array([int(g[5:7]) for g in table.group])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out) / args.sat / args.platform / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "sat": args.sat,
        "platform": args.platform,
        "run_id": run_id,
        "n_images": len(table.images),
        "n_rows": int(table.X.shape[0]),
        "n_train_days": int(len(set(table.group[train]))),
        "n_test_days": int(len(set(table.group[test]))),
        "features": table.names,
        "models": {},
    }
    for name in args.models.split(","):
        name = name.strip()
        if name == "ridge":
            arrays = models.fit_ridge_poly(
                table.X[train], table.y[train], table.names, sat, args.platform, degree=args.degree
            )
        elif name == "mlp":
            arrays = models.fit_mlp(
                table.X[train],
                table.y[train],
                table.names,
                sat,
                args.platform,
                hidden=tuple(int(h) for h in args.hidden.split(",")),
                epochs=args.epochs,
                X_val=table.X[test],
                y_val=table.y[test],
                seed=args.seed,
            )
        else:
            raise SystemExit(f"unknown model {name}")
        model_id = f"{name}_{run_id}"
        path = out_dir / f"{model_id}.npz"
        save_model(path, arrays)
        model = BackgroundModel.load(path)
        ev = {
            "train": evaluate.evaluate(
                model,
                table.X[train],
                table.y[train],
                table.names,
                table.group[train],
                platform=args.platform,
                month_of=month_of[train],
            ),
            "test": evaluate.evaluate(
                model,
                table.X[test],
                table.y[test],
                table.names,
                table.group[test],
                platform=args.platform,
                month_of=month_of[test],
            ),
        }
        report["models"][model_id] = {
            "path": str(path),
            "evaluation": ev,
            "meta": {k: v for k, v in arrays["meta"].items() if k != "history"},
        }
        log.info("%s test: %s", model_id, json.dumps(ev["test"]["overall"]))
        if storage is not None:
            storage.private.put_file(f"models/{args.sat}/{args.platform}/{model_id}.npz", path)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=1, default=str))
    if storage is not None:
        storage.private.put_json(f"reports/{args.sat}/{args.platform}/{run_id}/report.json", report)
        # pointer to the best model by test RMSE in kelvin
        best = min(
            report["models"].items(),
            key=lambda kv: kv[1]["evaluation"]["test"]["overall"]["kelvin"].get("rmse", 1e9),
        )
        storage.private.put_json(
            f"models/{args.sat}/{args.platform}/latest.json",
            {
                "model_id": best[0],
                "key": f"models/{args.sat}/{args.platform}/{best[0]}.npz",
                "run_id": run_id,
            },
        )
        log.info("latest model for %s/%s: %s", args.sat, args.platform, best[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
