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


def eruption_windows(
    eruptions,
    since: datetime | None = None,
    margin_days: float = 2.0,
    step_minutes: int = 180,
    onset_step_minutes: int = 60,
    onset_days: float = 3.0,
    cadence_minutes: int = 15,
    today: datetime | None = None,
) -> list[datetime]:
    """Timestamps covering each eruption window plus a margin either side.

    A coarse step over the whole window, and a finer step for the first `onset_days`
    after the start, where the effusion rate changes fastest. Times are aligned to the
    source cadence.
    """
    today = today or datetime.now(UTC)
    out: set[datetime] = set()
    for e in eruptions:
        start = datetime(e.start.year, e.start.month, e.start.day, tzinfo=UTC)
        end_date = e.end or today.date()
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC) + timedelta(days=1)
        if since is not None and end < since:
            continue
        a = start - timedelta(days=margin_days)
        b = min(end + timedelta(days=margin_days), today)
        for t in regular_range(a, b, step_minutes):
            out.add(t)
        for t in regular_range(
            start, min(start + timedelta(days=onset_days), b), onset_step_minutes
        ):
            out.add(t)
    aligned = {
        t.replace(minute=(t.minute // cadence_minutes) * cadence_minutes, second=0, microsecond=0)
        for t in out
    }
    return sorted(t for t in aligned if since is None or t >= since)
