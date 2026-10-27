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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--label")
    ap.add_argument("--out", type=Path, default=Path("out/assess"))
    args = ap.parse_args()

    source = PlanetaryComputerSource()

    class Keep:  # keep the layers so we can save evidence images too
        layers = None

        def fetch(self, site):
            Keep.layers = source.fetch(site)
            return Keep.layers

    result = assess(AssessmentRequest(lat=args.lat, lon=args.lon, label=args.label), Keep())
    save_evidence(Keep.layers, args.out)

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
