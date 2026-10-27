#!/usr/bin/env python3
"""Run a real FloodLens assessment for one site and print the metrics and score.

    python scripts/assess_site.py --lat 6.4541 --lon 3.3947 --label "Lagos Island"

Uses only open data (Planetary Computer, Open-Meteo); no Meta API calls yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.geo.raster import stretch
from app.geo.sources import PlanetaryComputerSource
from app.geo.water import clear_mask, optical_water, radar_water
from app.pipeline import assess
from app.schemas import AssessmentRequest
from app.scoring import FEATURES


def save_evidence(layers, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if layers.optical:
        o = layers.optical
        clear = clear_mask(o.scl)
        rgb = np.dstack([stretch(b, 1, 99, clear) for b in (o.red, o.green, o.blue)])
        Image.fromarray(rgb).resize((800, 800), Image.NEAREST).save(out / "optical.png")
        Image.fromarray((optical_water(o.scl, o.green, o.nir) * 255).astype(np.uint8)).resize(
            (800, 800), Image.NEAREST).save(out / "water_now.png")
    if layers.radar:
        mask, _ = radar_water(layers.radar.vv)
        Image.fromarray((mask * 255).astype(np.uint8)).resize((800, 800), Image.NEAREST).save(out / "water_wet.png")


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


def debug_report(layers, out: Path) -> None:
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
    if layers.jrc_occurrence is not None:
        occ = np.nan_to_num(layers.jrc_occurrence)
        Image.fromarray((np.clip(occ, 0, 100) * 2.55).astype(np.uint8)).resize(
            (800, 800), Image.NEAREST).save(out / "jrc_occurrence.png")
    for note in layers.notes:
        print(f"    note: {note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--label")
    ap.add_argument("--out", type=Path, default=Path("out/assess"))
    ap.add_argument("--debug", action="store_true", help="print raw layer statistics")
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
        debug_report(Keep.layers, args.out)

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
    print(f"\n  Evidence images and assessment.json in {args.out.resolve()}")
    print(json.dumps({"tier": result.score.tier.value, "score": result.score.score}))


if __name__ == "__main__":
    main()
