"""Assemble the training table from raw cutouts in the private bucket."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from thermal import features, sampling
from thermal.storage import Storage, raw_key

log = logging.getLogger(__name__)


@dataclass
class Table:
    names: list[str]
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray  # date string per row, for held-out-by-day splits
    image_index: np.ndarray  # index into `images` per row
    images: list[dict[str, Any]]  # per image: time, platform, n_pixels, key


def available_timestamps(storage: Storage, sat: dict[str, Any], sample_csv: Path) -> list[datetime]:
    """Timestamps from the sample list whose image cutout exists in the private bucket."""
    wanted = [t for s, t in sampling.read_list(sample_csv) if s == sat["key"]]
    have = set(storage.private.list_keys(f"raw/{sat['key']}/"))
    return [t for t in wanted if raw_key(sat["key"], t, "l15") in have]


def collect(
    storage: Storage | None,
    sat: dict[str, Any],
    timestamps: list[datetime],
    cache_dir: Path,
    platform: str | None = None,
    max_pixels_per_image: int = 4000,
    seed: int = 0,
    local_paths: dict[datetime, Path] | None = None,
) -> Table:
    """Download (or read cached) cutouts, build features, subsample pixels, stack."""
    rng = np.random.default_rng(seed)
    xs, ys, groups, idx, images = [], [], [], [], []
    names: list[str] | None = None
    for t in timestamps:
        if local_paths and t in local_paths:
            path = local_paths[t]
        else:
            key = raw_key(sat["key"], t, "l15")
            path = cache_dir / key
            if not path.exists():
                assert storage is not None
                storage.private.get_file(key, path)
        try:
            fs = features.build_features(path, t, sat)
        except Exception as exc:
            log.warning("skipping %s: %s", t.isoformat(), exc)
            continue
        if platform and fs.platform != platform:
            log.info("skipping %s: platform %s != %s", t.isoformat(), fs.platform, platform)
            continue
        n = fs.X.shape[0]
        if n == 0:
            continue
        take = np.arange(n)
        if n > max_pixels_per_image:
            take = rng.choice(n, max_pixels_per_image, replace=False)
        names = fs.names
        xs.append(fs.X[take])
        ys.append(fs.y[take])
        groups.append(np.full(len(take), t.strftime("%Y-%m-%d")))
        idx.append(np.full(len(take), len(images)))
        images.append({"time": t.isoformat(), "platform": fs.platform, "n_pixels": int(len(take))})
    if not xs:
        raise RuntimeError("no usable cutouts")
    return Table(
        names=names or [],
        X=np.concatenate(xs),
        y=np.concatenate(ys),
        group=np.concatenate(groups),
        image_index=np.concatenate(idx),
        images=images,
    )


def holdout_split(groups: np.ndarray, frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, test) masks holding out a fraction of whole days."""
    rng = np.random.default_rng(seed)
    days = np.unique(groups)
    rng.shuffle(days)
    n_test = max(1, int(round(frac * len(days))))
    test_days = set(days[:n_test])
    test = np.array([g in test_days for g in groups])
    return ~test, test
