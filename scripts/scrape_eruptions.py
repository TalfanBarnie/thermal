"""Print eruption tables from Wikipedia and the Global Volcanism Program for curation.

Stdlib only. Run in a GitHub Action (the development container cannot reach these hosts);
the output goes to the job log and is curated by hand into data/eruptions.csv.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from html.parser import HTMLParser

UA = "thermal-eruption-calendar/0.1 (https://github.com/TalfanBarnie/thermal)"

WIKI_PAGES = [
    "https://en.wikipedia.org/wiki/List_of_volcanic_eruptions_in_Iceland",
    "https://en.wikipedia.org/wiki/2023%E2%80%932025_Sundhn%C3%BAkur_eruptions",
    "https://en.wikipedia.org/wiki/Sundhn%C3%BAkur",
    "https://en.wikipedia.org/wiki/2021%E2%80%932023_Reykjanes_Peninsula_eruptions",
    "https://en.wikipedia.org/wiki/Fagradalsfjall",
    "https://en.wikipedia.org/wiki/2014%E2%80%932015_eruption_of_B%C3%A1r%C3%B0arbunga",
    "https://en.wikipedia.org/wiki/Gr%C3%ADmsv%C3%B6tn",
    "https://en.wikipedia.org/wiki/2010_eruptions_of_Eyjafjallaj%C3%B6kull",
]
GVP_WFS = (
    "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=2.0.0"
    "&request=GetFeature&typeName=GVP-VOTW:E3WebApp_Eruptions1960&outputFormat=csv"
)


class TableParser(HTMLParser):
    """Collect every <table> as a list of rows of cell texts."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table += 1
            self.tables.append([])
        elif tag == "tr" and self._in_table:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" / ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._in_table:
            self._in_table -= 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main() -> int:
    for url in WIKI_PAGES:
        print(f"\n######## {url}")
        try:
            html = fetch(url)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue
        parser = TableParser()
        parser.feed(html)
        for i, table in enumerate(parser.tables):
            if len(table) < 2 or len(table[0]) < 2:
                continue
            text = " ".join(" ".join(r) for r in table[:3]).lower()
            if not any(k in text for k in ("date", "start", "eruption", "volcano", "year")):
                continue
            print(f"\n#### table {i} ({len(table)} rows)")
            for row in table:
                print("\t".join(c[:120] for c in row))
    print(f"\n######## {GVP_WFS}")
    try:
        text = fetch(GVP_WFS)
        reader = csv.DictReader(io.StringIO(text))
        print("\t".join(reader.fieldnames or []))
        for row in reader:
            if row.get("Country", "") == "Iceland" and (row.get("StartDate") or "0")[:4] >= "2000":
                print("\t".join(f"{k}={v}" for k, v in row.items() if v))
    except Exception as exc:
        print(f"FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
