#!/usr/bin/env python3
"""Run a real FloodLens assessment for one site and print the metrics and score.

    python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --label "Lagos Island"
    python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --explain

Open data only (Planetary Computer, Open-Meteo) unless --explain, which also calls
Muse Spark through the agent and needs MODEL_API_KEY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.agent.loop import explain
from app.evidence import evidence_pngs
from app.geo.sources import PlanetaryComputerSource
from app.geo.water import radar_water
from app.pipeline import assess
from app.schemas import AssessmentRequest
from app.scoring import FEATURES


def save_evidence(layers, out: Path) -> dict[str, bytes]:
    """Write the same PNGs the agent looks at through view_evidence."""
    out.mkdir(parents=True, exist_ok=True)
    pngs = evidence_pngs(layers)
    for name, png in pngs.items():
        (out / f"{name}.png").write_bytes(png)
    return pngs


def _stats(name: str, a: np.ndarray | None) -> None:
    if a is None:
        print(f"    {name:14s} missing")
        return
    f = a[np.isfinite(a)]
    if f.size == 0:
        print(f"    {name:14s} all no-data")
        return
    p = np.percentile(f, [1, 10, 50, 90, 99])
    print(f"    {name:14s} n={f.size:6d}  zeros={100 * (f == 0).mean():5.1f}%  "
          f"p1/p10/p50/p90/p99 = {' / '.join(f'{v:.2f}' for v in p)}")


def debug_report(layers) -> None:
    print("\n  DEBUG: raw layers")
    print(f"    item ids: {layers.item_ids}")
    if layers.optical:
        o = layers.optical
        print(f"    optical {o.item_id}  tile cloud {100 * o.tile_cloud:.1f}%  "
              f"SCL classes {dict(zip(*[x.tolist() for x in np.unique(o.scl, return_counts=True)], strict=True))}")
    _stats("jrc_occurrence", layers.jrc_occurrence)
    _stats("dem_m", layers.dem)
    if layers.radar:
        from app.geo.water import to_db
        vv = layers.radar.vv
        _stats("radar_vv_db", np.where(vv > 0, to_db(vv), np.nan))
        _, t = radar_water(vv)
        print(f"    radar {layers.radar.item_id}  Otsu threshold after clipping: {t:.1f} dB")
    for note in layers.notes:
        print(f"    note: {note}")


def print_run(run) -> None:
    r = run.report
    print("\n  AGENT STEPS")
    for s in run.steps:
        tok = f"{s.input_tokens:>6d} in {s.output_tokens:>5d} out" if s.input_tokens or s.output_tokens else " " * 19
        print(f"    {s.n:>2d} {s.kind:<6s} {s.name:<22s} {s.ms:>6d} ms  {tok}  {s.detail}")
    print(f"\n  {len(run.steps)} steps, {run.calls} model calls, "
          f"{run.input_tokens} in + {run.output_tokens} out tokens, est ${run.cost_usd:.4f}")
    if run.errors:
        print("  verifier errors:")
        for e in run.errors:
            print(f"    - {e}")

    print(f"\n  REPORT  (explained: {str(r.explained).lower()})")
    print(f"\n  {r.summary}")
    print("\n  Drivers:")
    for d in r.drivers:
        print(f"    {d.metric_id:24s} {d.why}")
    print(f"\n  Action: {r.action.value}")
    print(f"  Rationale: {r.rationale}")
    if r.caveats:
        print("  Caveats:")
        for c in r.caveats:
            print(f"    - {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--label")
    ap.add_argument("--out", type=Path, default=Path("out/assess"))
    ap.add_argument("--debug", action="store_true", help="print raw layer statistics")
    ap.add_argument("--explain", action="store_true",
                    help="have Muse Spark explain the score (calls the Meta API, needs MODEL_API_KEY)")
    args = ap.parse_args()

    source = PlanetaryComputerSource()

    class Keep:  # keep the layers so we can save evidence images too
        layers = None

        def fetch(self, site):
            Keep.layers = source.fetch(site)
            return Keep.layers

    result = assess(AssessmentRequest(lat=args.lat, lon=args.lon, label=args.label), Keep())
    save_evidence(Keep.layers, args.out)
    if args.debug:
        debug_report(Keep.layers)

    print(f"\n{args.label or ''} ({args.lat}, {args.lon})")
    for m in result.metrics:
        print(f"  {m.label:55s} {m.display():>8s} {m.unit:3s}  [{m.source}]")
    weights = {f.metric_id: f.weight for f in FEATURES}
    print("\n  Score contributions:")
    for mid, pts in result.score.contributions.items():
        print(f"    {mid:24s} {pts:5.1f}  (weight {weights[mid]:.2f})")
    print(f"\n  SCORE {result.score.score}/100  TIER {result.score.tier.value}  confidence {result.score.confidence}")
    if result.score.missing:
        print(f"  missing: {', '.join(result.score.missing)}")
    for note in result.notes:
        print(f"  note: {note}")
    (args.out / "assessment.json").write_text(result.model_dump_json(indent=2))

    if args.explain:
        run = explain(result, Keep.layers)
        print_run(run)
        (args.out / "report.json").write_text(run.model_dump_json(indent=2))

    print(f"\n  Evidence images and assessment.json in {args.out.resolve()}")
    print(json.dumps({"tier": result.score.tier.value, "score": result.score.score}))


if __name__ == "__main__":
    main()
