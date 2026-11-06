"""Stratified sampling of archive timestamps for the training set.

Timestamps are drawn uniformly over (calendar month, UTC hour) cells so that the
model sees every season and every solar geometry, and are rejected if they fall
inside an eruption exclusion window.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path


def stratified_sample(
    start: datetime,
    end: datetime,
    n: int,
    cadence_minutes: int,
    exclude: Callable[[datetime], bool] | None = None,
    seed: int = 0,
    max_tries_per_cell: int = 2000,
) -> list[datetime]:
    """Return n nominal timestamps in [start, end), balanced over month x hour cells."""
    rng = random.Random(seed)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    cells = [(m, h) for m in range(1, 13) for h in range(24)]
    rng.shuffle(cells)
    per_cell, extra = divmod(n, len(cells))
    quotas = {cell: per_cell + (1 if i < extra else 0) for i, cell in enumerate(cells)}
    total_days = (end - start).days
    slots_per_hour = 60 // cadence_minutes
    chosen: set[datetime] = set()
    for (month, hour), quota in quotas.items():
        got, tries = 0, 0
        while got < quota and tries < max_tries_per_cell:
            tries += 1
            day = start + timedelta(days=rng.randrange(total_days))
            if day.month != month:
                continue
            minute = cadence_minutes * rng.randrange(slots_per_hour)
            t = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if t < start or t >= end or t in chosen:
                continue
            if exclude is not None and exclude(t):
                continue
            chosen.add(t)
            got += 1
    return sorted(chosen)


def write_list(path: Path, timestamps: list[datetime], sat: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sat", "time"])
        for t in timestamps:
            w.writerow([sat, t.strftime("%Y-%m-%dT%H:%M:%SZ")])


def read_list(path: Path) -> list[tuple[str, datetime]]:
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = datetime.fromisoformat(row["time"].replace("Z", "+00:00")).astimezone(UTC)
            out.append((row["sat"], t))
    return out


def regular_range(start: datetime, end: datetime, step_minutes: int) -> list[datetime]:
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    out, t = [], start
    while t < end:
        out.append(t)
        t += timedelta(minutes=step_minutes)
    return out
