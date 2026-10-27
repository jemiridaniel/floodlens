#!/usr/bin/env python3
"""Score every site in data/validation_sites.csv and print how flood and dry sites compare.

    python scripts/validate.py
    python scripts/validate.py --sites ../data/validation_sites.csv --out out/validation.csv

Open data only: no agent, no Meta API calls, so running this is free. Each flood site
carries a wet_date, so its radar layer comes from the flood itself rather than from the
wettest rainfall day we can find.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from app.geo.sources import PlanetaryComputerSource
from app.pipeline import assess
from app.schemas import AssessmentRequest
from app.scoring import FEATURES

SITES_CSV = Path(__file__).resolve().parents[2] / "data" / "validation_sites.csv"
SCORED = [f.metric_id for f in FEATURES]  # the five metrics the score is built from
HEADS = {
    "m_water_history_500": "hist%",
    "m_wet_water_250": "wet%",
    "m_height_above_water": "height",
    "m_dist_water": "dist",
    "m_max_daily_rain": "rain",
}


@dataclass
class Row:
    site_id: str
    label: str
    expected: str
    score: float | None = None
    tier: str = ""
    confidence: str = ""
    values: dict[str, str] | None = None
    missing: str = ""
    error: str = ""


def read_sites(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("site_id")]


def run_site(site: dict[str, str]) -> Row:
    row = Row(site_id=site["site_id"], label=site["label"], expected=site["expected"])
    req = AssessmentRequest(lat=float(site["lat"]), lon=float(site["lon"]), label=site["label"],
                            wet_date=site.get("wet_date") or None)
    try:
        result = assess(req, PlanetaryComputerSource())
    except Exception as exc:  # one bad site must not lose the whole run
        row.error = f"{exc.__class__.__name__}: {exc}"
        return row
    by_id = {m.id: m for m in result.metrics}
    row.score = result.score.score
    row.tier = result.score.tier.value
    row.confidence = result.score.confidence
    row.values = {mid: (by_id[mid].display() if mid in by_id else "-") for mid in SCORED}
    row.missing = " ".join(result.score.missing)
    return row


def print_table(rows: list[Row]) -> None:
    head = f"  {'site':<36s} {'expected':<8s} {'score':>6s} {'tier':<9s}"
    head += "".join(f"{HEADS[m]:>8s}" for m in SCORED)
    print(f"\n{head}")
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        if r.error:
            print(f"  {r.label[:36]:<36s} {r.expected:<8s} {'ERROR':>6s} {r.error[:40]}")
            continue
        line = f"  {r.label[:36]:<36s} {r.expected:<8s} {r.score:>6.1f} {r.tier:<9s}"
        line += "".join(f"{r.values[m]:>8s}" for m in SCORED)
        print(line)

    scored = [r for r in rows if r.score is not None]
    by_class = {c: [r.score for r in scored if r.expected == c] for c in ("flood", "dry")}
    print()
    for c, vals in by_class.items():
        if vals:
            print(f"  {c:<6s} n={len(vals)}  mean {sum(vals) / len(vals):5.1f}  "
                  f"range {min(vals):.1f}-{max(vals):.1f}")
    if by_class["flood"] and by_class["dry"]:
        gap = min(by_class["flood"]) - max(by_class["dry"])
        print(f"\n  Separation: lowest flood minus highest dry = {gap:+.1f} points"
              f"{'  (clean split)' if gap > 0 else '  (OVERLAP: the score does not separate these sites)'}")
    else:
        print("  Separation needs both classes: add the dry sites to the CSV.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", type=Path, default=SITES_CSV)
    ap.add_argument("--out", type=Path, default=Path("out/validation.csv"))
    args = ap.parse_args()

    sites = read_sites(args.sites)
    if not sites:
        sys.exit(f"No sites in {args.sites}")
    rows = []
    for i, site in enumerate(sites, 1):
        print(f"  [{i}/{len(sites)}] {site['label']} ...", flush=True)
        rows.append(run_site(site))

    print_table(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "label", "expected", "score", "tier", "confidence", *SCORED, "missing", "error"])
        for r in rows:
            w.writerow([r.site_id, r.label, r.expected, "" if r.score is None else r.score, r.tier, r.confidence,
                        *[(r.values or {}).get(m, "") for m in SCORED], r.missing, r.error])
    print(f"\n  Saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
