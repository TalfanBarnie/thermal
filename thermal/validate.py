"""Validation against the eruption calendar, from the published time series.

Reads index.json and the monthly shards over HTTP from the public bucket (no credentials
needed), so it can run anywhere. For each eruption it reports the first sample at or
above a threshold inside the window and the detection delay relative to the calendar
start, plus false-alarm statistics outside eruption windows.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .calendar import Calendar, Eruption


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "thermal-validate/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def load_rows(
    public_url: str, sat: str, region: str, months: list[str] | None = None
) -> list[dict]:
    index = json.loads(_get(f"{public_url}/index.json"))
    avail = index["sources"][sat]["regions"][region]["months"]
    rows: list[dict] = []
    for m in months or avail:
        if m not in avail:
            continue
        text = _get(f"{public_url}/timeseries/{sat}/{region}/{m}.jsonl").decode()
        rows += [json.loads(line) for line in text.splitlines() if line.strip()]
    rows.sort(key=lambda r: r["time"])
    for r in rows:
        r["_t"] = datetime.strptime(r["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return rows


def onset_report(
    rows: list[dict],
    eruptions: list[Eruption],
    stat: str = "max_z",
    threshold: float = 4.0,
    margin_days: float = 2.0,
) -> list[dict[str, Any]]:
    """Per eruption: samples in the window, first detection, delay from the calendar start."""
    out = []
    for e in eruptions:
        start = datetime(e.start.year, e.start.month, e.start.day, tzinfo=UTC)
        end_date = e.end or datetime.now(UTC).date()
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC) + timedelta(days=1)
        win = [
            r
            for r in rows
            if start - timedelta(days=margin_days) <= r["_t"] < end + timedelta(days=margin_days)
        ]
        if not win:
            continue
        hits = [r for r in win if r.get(stat) is not None and r[stat] >= threshold]
        first = min((r for r in hits if r["_t"] >= start), key=lambda r: r["_t"], default=None)
        early = [r for r in hits if r["_t"] < start]
        out.append(
            {
                "volcano": e.volcano,
                "start": e.start.isoformat(),
                "end": e.end.isoformat() if e.end else "",
                "n_samples": len(win),
                "n_hits": len(hits),
                "first_hit": first["time"] if first else "",
                "delay_hours": round((first["_t"] - start).total_seconds() / 3600, 1)
                if first
                else None,
                "n_hits_before_start": len(early),
                "peak": max((r[stat] for r in win if r.get(stat) is not None), default=None),
            }
        )
    return out


def false_alarms(
    rows: list[dict], cal: Calendar, stat: str = "max_z", threshold: float = 4.0
) -> dict[str, Any]:
    quiet = [r for r in rows if not cal.excluded(r["_t"]) and r.get(stat) is not None]
    hits = [r for r in quiet if r[stat] >= threshold]
    return {
        "n_quiet_samples": len(quiet),
        "n_false_alarms": len(hits),
        "false_alarm_rate": (len(hits) / len(quiet)) if quiet else None,
        "examples": [r["time"] for r in hits[:10]],
    }


def export_eruption_series(
    rows: list[dict], eruptions: list[Eruption], out_dir: Path, margin_days: float = 2.0
) -> list[Path]:
    """One CSV per eruption window for the effusion-rate analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fields = [
        "time",
        "platform",
        "model",
        "n_valid",
        "sum_pos",
        "max",
        "max_z",
        "n_above",
        "mean",
        "vrp_w",
        "n_sat",
        "n_cloud",
        "n_cloud_unknown",
        "sza_mean",
    ]
    for e in eruptions:
        start = datetime(e.start.year, e.start.month, e.start.day, tzinfo=UTC)
        end_date = e.end or datetime.now(UTC).date()
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC) + timedelta(days=1)
        win = [
            r
            for r in rows
            if start - timedelta(days=margin_days) <= r["_t"] < end + timedelta(days=margin_days)
        ]
        if not win:
            continue
        slug = e.volcano.replace(" ", "_").replace("(", "").replace(")", "")
        name = f"{e.start.isoformat()}_{slug}.csv"
        path = out_dir / name
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in win:
                w.writerow(r)
        paths.append(path)
    return paths
