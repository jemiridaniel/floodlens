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

import numpy as np

from app.agent.history import lookup as flood_history_lookup
from app.geo.aoi import Grid, Site, ring_mask
from app.geo.sources import PlanetaryComputerSource, _catalog, landcover
from app.geo.water import share_within
from app.pipeline import assess
from app.schemas import AssessmentRequest
from app.scoring import FEATURES

SITES_CSV = Path(__file__).resolve().parents[2] / "data" / "validation_sites.csv"

# Baseline: random points in built-up Lagos, to answer "does a flood site score higher
# than any old address in this city?". Without it, four Moderate scores prove nothing.
LAGOS_BBOX = (6.40, 6.70, 3.10, 3.60)  # lat_min, lat_max, lon_min, lon_max
BASELINE_SEED = 20261027  # fixed, so the baseline is the same set every run
PROBE = Grid(size_m=500, res_m=10)  # just wide enough for the 250 m built-up test
BUILT_UP_MIN = 0.50
WORLDCOVER_BUILT_UP, WORLDCOVER_WATER = 50, 80
SCORED = [f.metric_id for f in FEATURES]  # the five metrics the score is built from
HEADS = {
    "m_reported_floods": "reports",
    "m_local_low_spot": "lowspot",
    "m_built_up_500": "built%",
    "m_wetland_500": "wetlnd%",
    "m_water_history_500": "hist%",
    "m_wet_water_250": "radar%",
    "m_dist_water": "dist",
    "m_height_above_water": "height",
    "m_max_daily_rain": "rain",
}


HISTORY: dict[str, object] = {}  # site_id -> FloodHistory, for the cost and circularity report


@dataclass
class Row:
    site_id: str
    label: str
    expected: str
    quality: str = "ok"  # an "approximate" pin is shown but kept out of the summary
    score: float | None = None
    tier: str = ""
    confidence: str = ""
    values: dict[str, str] | None = None
    missing: str = ""
    error: str = ""


def read_sites(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("site_id")]


def run_site(site: dict[str, str], history: bool = False) -> Row:
    row = Row(site_id=site["site_id"], label=site["label"], expected=site["expected"],
              quality=site.get("geocode_quality") or "ok")
    req = AssessmentRequest(lat=float(site["lat"]), lon=float(site["lon"]), label=site["label"],
                            wet_date=site.get("wet_date") or None)

    def remember(s: Site):
        got = flood_history_lookup(s)
        HISTORY[site["site_id"]] = got
        return got

    try:
        result = assess(req, PlanetaryComputerSource(), flood_history=remember if history else None)
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


def is_built_up(lc, grid: Grid) -> tuple[bool, float]:
    """Built-up share within 250 m, and whether this point works as a baseline site.

    Open water is skipped on the centre pixel: a point in the lagoon is not an address.
    WorldCover 0 means outside the tile, and must not dilute the share.
    """
    share = share_within(lc.astype(int) == WORLDCOVER_BUILT_UP, ring_mask(grid, 250), lc > 0)
    centre = int(lc[grid.side // 2, grid.side // 2])
    return bool(share is not None and share >= BUILT_UP_MIN and centre != WORLDCOVER_WATER), share or 0.0


def sample_built_up(n: int, seed: int = BASELINE_SEED) -> list[tuple[float, float, float]]:
    """n random (lat, lon, built-up %) points inside built-up Lagos. Deterministic."""
    rng = np.random.default_rng(seed)
    catalog = _catalog()
    lat0, lat1, lon0, lon1 = LAGOS_BBOX
    picked: list[tuple[float, float, float]] = []
    tries = 0
    while len(picked) < n and tries < 30 * n:
        tries += 1
        lat, lon = float(rng.uniform(lat0, lat1)), float(rng.uniform(lon0, lon1))
        try:
            got = landcover(catalog, Site(lat, lon), PROBE)
        except Exception:
            continue
        if got is None:
            continue
        ok, share = is_built_up(got[0], PROBE)
        if ok:
            picked.append((lat, lon, 100 * share))
    print(f"    {len(picked)} points from {tries} draws "
          f"({100 * len(picked) / max(tries, 1):.0f}% of the box is built-up land)")
    return picked


def percentile_of(x: float, sample: list[float]) -> float:
    """Where x sits inside sample, 0-100, counting ties as half."""
    below = sum(1 for s in sample if s < x)
    ties = sum(1 for s in sample if s == x)
    return 100 * (below + 0.5 * ties) / len(sample)


def print_baseline(flood: list[Row], base: list[Row]) -> None:
    scores = sorted(r.score for r in base if r.score is not None)
    floods = [r for r in flood if r.score is not None]
    if not scores or not floods:
        print("\n  No baseline comparison: nothing scored.")
        return
    q = np.percentile(scores, [0, 25, 50, 75, 100])
    print(f"\n  BASELINE  n={len(scores)} random built-up points in Lagos (seed {BASELINE_SEED})")
    print(f"    min {q[0]:.1f}   p25 {q[1]:.1f}   median {q[2]:.1f}   p75 {q[3]:.1f}   max {q[4]:.1f}")

    print("\n    flood site                       score   percentile in the baseline")
    for r in floods:
        print(f"    {r.label[:30]:<30s} {r.score:>6.1f}   {percentile_of(r.score, scores):>5.1f}")

    pairs = len(floods) * len(scores)
    wins = sum(1.0 if b > f.score else 0.5 if b == f.score else 0.0 for f in floods for b in scores)
    auc = 1 - wins / pairs
    verdict = "no better than chance" if abs(auc - 0.5) < 0.1 else (
        "flood sites rank above random" if auc > 0.5 else "WORSE than chance")
    print(f"\n    A random built-up point outscores a flood site in {100 * wins / pairs:.1f}% "
          f"of {pairs} pairs.")
    print(f"    AUC (flood ranked above random) = {auc:.3f}  -- {verdict}")

    rain = [float(r.values["m_max_daily_rain"]) for r in base
            if r.values and r.values.get("m_max_daily_rain", "-") != "-"]
    if rain:
        print(f"\n    m_max_daily_rain across the baseline: {min(rain):.0f}-{max(rain):.0f} mm, "
              f"median {np.median(rain):.0f}, spread {max(rain) - min(rain):.0f} mm.")
        print("    Open-Meteo is a ~10 km grid, so one storm gives the whole city the same "
              "number: it separates years, not addresses.")


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "label", "expected", "geocode_quality", "score", "tier",
                    "confidence", *SCORED, "missing", "error"])
        for r in rows:
            w.writerow([r.site_id, r.label, r.expected, r.quality, "" if r.score is None else r.score,
                        r.tier, r.confidence, *[(r.values or {}).get(m, "") for m in SCORED],
                        r.missing, r.error])


def print_history(rows: list[Row], base: list[Row]) -> None:
    """The circularity check. The flood sites were *chosen* from news reports, so finding
    reports for them proves nothing. The informative number is how often a random address
    also turns up a street-level report."""
    runs = [(r, HISTORY[r.site_id]) for r in rows + base if r.site_id in HISTORY]
    if not runs:
        return
    searches = sum(h.searches for _, h in runs)
    cached = sum(1 for _, h in runs if h.from_cache)
    cost = sum(h.cost_usd for _, h in runs if not h.from_cache)
    print(f"\n  LOCAL FLOOD HISTORY  {len(runs)} places, {cached} served from cache, "
          f"{searches} searches, est ${cost:.4f}")

    dropped = sum(len(h.dropped) for _, h in runs)
    print(f"    {sum(len(h.reports) for _, h in runs)} reports kept, {dropped} dropped "
          "(uncited url, unparseable date, or outside the window)")
    for _, h in runs:
        for d in h.dropped[:2]:
            print(f"      dropped: {d[:110]}")

    def street_level(h) -> bool:
        return any(r.match_level.value == "street" for r in h.reports)

    for name, group in (("flood sites", [h for r, h in runs if r.expected == "flood"]),
                        ("random baseline", [h for r, h in runs if r.expected == "baseline"])):
        if not group:
            continue
        any_level = sum(1 for h in group if h.reports)
        street = sum(1 for h in group if street_level(h))
        print(f"\n    {name}: {any_level}/{len(group)} have any report, "
              f"{street}/{len(group)} have a STREET-level report")
    print("\n    The flood sites came from news, so their reports are circular by "
          "construction.")
    print("    Only the baseline street-level rate says whether this signal is selective.")


def print_table(rows: list[Row]) -> None:
    head = f"  {'site':<30s} {'exp':<6s} {'score':>6s} {'tier':<9s}"
    head += "".join(f"{HEADS[m]:>8s}" for m in SCORED)
    print(f"\n{head}")
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        if r.error:
            print(f"  {r.label[:30]:<30s} {r.expected:<6s} {'ERROR':>6s} {r.error[:46]}")
            continue
        mark = "~" if r.quality == "approximate" else " "
        line = f" {mark}{r.label[:30]:<30s} {r.expected:<6s} {r.score:>6.1f} {r.tier:<9s}"
        line += "".join(f"{r.values[m]:>8s}" for m in SCORED)
        print(line)

    for r in (r for r in rows if r.quality == "approximate"):
        print(f"  ~ {r.label} is left out of the summary until it has a real pin")
    scored = [r for r in rows if r.score is not None and r.quality != "approximate"]
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
    ap.add_argument("--baseline", type=int, nargs="?", const=30, default=0, metavar="N",
                    help="also score N random built-up points in Lagos (default 30, fixed seed)")
    ap.add_argument("--baseline-out", type=Path, default=Path("out/baseline.csv"))
    ap.add_argument("--history", action="store_true",
                    help="also look up local flood history from news (Meta web search, costs money)")
    args = ap.parse_args()

    sites = read_sites(args.sites)
    if not sites:
        sys.exit(f"No sites in {args.sites}")
    rows = []
    for i, site in enumerate(sites, 1):
        print(f"  [{i}/{len(sites)}] {site['label']} ...", flush=True)
        rows.append(run_site(site, args.history))

    print_table(rows)

    base: list[Row] = []
    if args.baseline:
        print(f"\n  Sampling {args.baseline} random built-up points in Lagos ...")
        points = sample_built_up(args.baseline)
        for i, (lat, lon, _share) in enumerate(points, 1):
            print(f"  [baseline {i}/{len(points)}] {lat:.4f},{lon:.4f} ...", flush=True)
            base.append(run_site({"site_id": f"bg{i:03d}", "label": f"{lat:.4f},{lon:.4f}",
                                  "expected": "baseline", "lat": str(lat), "lon": str(lon),
                                  "wet_date": ""}, args.history))
        write_csv(args.baseline_out, base)
        print_baseline([r for r in rows if r.expected == "flood" and r.quality != "approximate"], base)

    if args.history:
        print_history(rows, base)

    write_csv(args.out, rows)
    print(f"\n  Saved {args.out.resolve()}"
          + (f" and {args.baseline_out.resolve()}" if base else ""))


if __name__ == "__main__":
    main()
