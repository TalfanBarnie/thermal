"""Eruption exclusion calendar: data/eruptions.csv.

An image is excluded from training when its time lies within
[00:00 UTC on start_date, end_date + buffer_days]. An empty end_date means ongoing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

DEFAULT_BUFFER_DAYS = 30
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "eruptions.csv"


@dataclass(frozen=True)
class Eruption:
    volcano: str
    start: date
    end: date | None
    source: str = ""
    notes: str = ""

    def window(self, buffer_days: int = DEFAULT_BUFFER_DAYS, today: date | None = None):
        """Exclusion window as (start_datetime, end_datetime), both UTC."""
        start = datetime(self.start.year, self.start.month, self.start.day, tzinfo=UTC)
        end_date = self.end or (today or datetime.now(UTC).date())
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC) + timedelta(
            days=buffer_days + 1
        )
        return start, end


def load(path: Path | None = None) -> list[Eruption]:
    path = path or _DEFAULT_PATH
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("start_date"):
                continue
            out.append(
                Eruption(
                    volcano=row["volcano"].strip(),
                    start=date.fromisoformat(row["start_date"].strip()),
                    end=date.fromisoformat(row["end_date"].strip())
                    if row.get("end_date", "").strip()
                    else None,
                    source=row.get("source", "").strip(),
                    notes=row.get("notes", "").strip(),
                )
            )
    return sorted(out, key=lambda e: e.start)


class Calendar:
    def __init__(self, eruptions: list[Eruption], buffer_days: int = DEFAULT_BUFFER_DAYS):
        self.eruptions = eruptions
        self.buffer_days = buffer_days
        self._windows = [e.window(buffer_days) for e in eruptions]

    @classmethod
    def from_file(cls, path: Path | None = None, buffer_days: int = DEFAULT_BUFFER_DAYS):
        return cls(load(path), buffer_days)

    def excluded(self, t: datetime) -> bool:
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        return any(s <= t < e for s, e in self._windows)

    def eruption_at(self, t: datetime) -> Eruption | None:
        """The eruption whose actual (unbuffered) dates contain t, if any."""
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        for e in self.eruptions:
            s, end = e.window(buffer_days=0)
            if s <= t < end:
                return e
        return None
