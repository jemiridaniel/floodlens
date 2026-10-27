#!/usr/bin/env python3
"""Test SAM 3.1 on site photos: which prompts find drains, water marks and door steps?

    export MODEL_API_KEY=...
    python scripts/sam_photos.py photos/*.jpg

Writes one overlay per photo and prompt to out/photos/, plus results.json.
Cost: $0.0025 per photo per prompt (5 photos x 8 prompts = $0.10).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from PIL import Image, ImageOps

from app.config import META_BASE_URL, get_settings
from app.sam.client import overlay, segment

PROMPTS = ["standing water", "puddle", "water stain", "drain", "gutter", "door", "step", "road"]
MAX_SIDE = 1280


def load(path: Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # respect phone rotation
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(img)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--prompt", action="append", help="override the default prompts (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("out/photos"))
    args = ap.parse_args()
    key = get_settings().model_api_key
    if not key:
        raise SystemExit("Set MODEL_API_KEY first.")
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or PROMPTS

    rows = []
    async with AsyncOpenAI(base_url=META_BASE_URL, api_key=key) as api:
        for path in args.photos:
            img = load(path)
            print(f"\n{path.name} ({img.shape[1]}x{img.shape[0]})")
            for prompt in prompts:
                try:
                    seg = await segment(img, prompt, api=api)
                except Exception as exc:
                    print(f"  {prompt:16s} ERROR {exc}")
                    rows.append({"photo": path.name, "prompt": prompt, "error": str(exc)[:200]})
                    continue
                print(f"  {prompt:16s} instances={seg.instances:2d}  cover={100 * seg.coverage:5.1f}%")
                rows.append({"photo": path.name, "prompt": prompt, "instances": seg.instances,
                             "coverage_pct": round(100 * seg.coverage, 1)})
                if seg.instances:
                    Image.fromarray(overlay(img, seg.mask)).save(
                        args.out / f"{path.stem}__{prompt.replace(' ', '_')}.jpg", quality=85)
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    found = sorted({r["prompt"] for r in rows if r.get("instances")})
    print(f"\nPrompts that found something: {', '.join(found) or 'none'}")
    print(f"Overlays in {args.out.resolve()} (open them to check the masks are right)")


if __name__ == "__main__":
    asyncio.run(main())
