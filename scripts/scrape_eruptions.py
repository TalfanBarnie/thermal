"""Print eruption source material for Iceland for curation into data/eruptions.csv.

Stdlib only. Run in a GitHub Action (the development container cannot reach these
hosts); the output goes to the job log and is curated by hand.

Sources:
- Wikipedia (MediaWiki API): wikitext lines mentioning 2000s dates from eruption pages
  found by title search.
- Global Volcanism Program WFS: eruptions since 1960 inside the Iceland box.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request

UA = "thermal-eruption-calendar/0.1 (https://github.com/TalfanBarnie/thermal)"
API = "https://en.wikipedia.org/w/api.php"
SEARCHES = [
    "Sundhnúkur eruptions",
    "Fagradalsfjall eruption 2021",
    "Litli-Hrútur eruption 2023",
    "Meradalir eruption 2022",
    "Holuhraun eruption 2014 Bárðarbunga",
    "Grímsvötn eruption 2011",
    "Grímsvötn eruption 2004",
    "Eyjafjallajökull eruption 2010",
    "List of volcanic eruptions in Iceland",
    "Reykjanes Peninsula volcanic activity 2024 2025",
]
GVP_WFS = (
    "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=2.0.0"
    "&request=GetFeature&typeName=GVP-VOTW:E3WebApp_Eruptions1960&outputFormat=csv"
)
DATE_RE = re.compile(r"20[0-2]\d")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def api(**params) -> dict:
    params.update(format="json", formatversion="2")
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(params)))


def main() -> int:
    titles: list[str] = []
    for q in SEARCHES:
        try:
            hits = api(action="query", list="search", srsearch=q, srlimit=4)["query"]["search"]
        except Exception as exc:
            print(f"search {q!r} FAILED: {exc}")
            continue
        found = [h["title"] for h in hits]
        print(f"search {q!r}: {found}")
        for t in found[:2]:
            if t not in titles:
                titles.append(t)

    for title in titles:
        print(f"\n######## {title}")
        try:
            page = api(action="parse", page=title, prop="wikitext", redirects=1)["parse"]
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue
        text = page["wikitext"]
        n = 0
        for line in text.split("\n"):
            s = line.strip()
            if DATE_RE.search(s) and len(s) < 600:
                print(s)
                n += 1
            if n > 400:
                print("... (truncated)")
                break

    print("\n######## GVP eruptions since 1960 inside the Iceland box")
    try:
        reader = csv.DictReader(io.StringIO(fetch(GVP_WFS)))
        print("\t".join(reader.fieldnames or []))
        for row in reader:
            try:
                lat, lon = float(row["LatitudeDecimal"]), float(row["LongitudeDecimal"])
            except (KeyError, ValueError):
                continue
            if (
                63 <= lat <= 67
                and -25 <= lon <= -13
                and (row.get("StartDate") or "0")[:4] >= "2000"
            ):
                print("\t".join(f"{k}={v}" for k, v in row.items() if v and k != "GeoLocation"))
    except Exception as exc:
        print(f"FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
