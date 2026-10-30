"""Site photos: strip location metadata, store, and segment with SAM.

Two rules here, both deliberate:

1. **EXIF is stripped before anything else.** Phone photos carry GPS coordinates, and a
   loan file should not quietly accumulate the location of a borrower's home. The pixels
   are copied into a fresh image, so nothing but the picture survives. Orientation is
   applied first, or stripping it would leave photos sideways.
2. **Findings never touch the score.** SAM on a doorway tells you where the water reached
   on one day from one angle; turning that into a number needs a labelled photo test we
   have not run. Until then the masks are shown as evidence and weigh nothing.
"""
from __future__ import annotations

import asyncio
import io
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

UPLOADS = Path(os.environ.get("FLOODLENS_UPLOAD_DIR",
                              Path(__file__).resolve().parents[2] / "data" / "uploads"))
MAX_PHOTOS = 5
MAX_EDGE_PX = 1600  # plenty for SAM, and keeps the upload directory small
PROMPTS = ("standing water", "puddle", "door", "road", "drain")
# design tokens: water reads blue, built things and drains amber
MASK_COLOURS = {"standing water": (63, 169, 245), "puddle": (63, 169, 245),
                "door": (242, 163, 58), "road": (242, 163, 58), "drain": (242, 163, 58)}


@dataclass
class Finding:
    prompt: str
    coverage: float  # share of the frame, 0-1
    instances: int
    status: str
    overlay: str = ""  # evidence name of the overlay PNG, if one was made


@dataclass
class Photo:
    name: str  # stored filename, also the evidence name
    width: int
    height: int
    findings: list[Finding] = field(default_factory=list)
    error: str = ""


def strip_exif(data: bytes, max_edge: int = MAX_EDGE_PX) -> Image.Image:
    """The picture, and nothing else: no EXIF, no GPS, orientation already applied."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # honour the orientation tag before dropping it
    img = img.convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    # Rebuild from raw pixels: nothing but the picture crosses over, no info dict at all.
    return Image.frombytes("RGB", img.size, img.tobytes())


def save_photo(data: bytes, directory: Path | None = None) -> tuple[str, Image.Image]:
    """Strip, store under a fresh name, and return (name, image)."""
    directory = UPLOADS if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    image = strip_exif(data)
    name = f"{uuid.uuid4().hex[:12]}.png"
    image.save(directory / name, format="PNG")  # PNG carries no EXIF container at all
    return name, image


def overlay(base: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int],
            alpha: float = 0.55) -> np.ndarray:
    out = base.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(colour) * alpha
    return out.astype(np.uint8)


def to_png(image: Image.Image | np.ndarray) -> bytes:
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def segment_photo(name: str, image: Image.Image, prompts=PROMPTS,
                        segment=None) -> tuple[Photo, dict[str, bytes]]:
    """Run every prompt over one photo. Returns the findings and any overlay PNGs.

    A prompt that fails is recorded and skipped: one bad segmentation must not cost the
    officer the rest of the photo's evidence.
    """
    if segment is None:
        from .sam.client import segment as segment  # noqa: PLW0127  (lazy: needs the API key)
    arr = np.asarray(image)
    photo = Photo(name=name, width=image.width, height=image.height)
    pngs: dict[str, bytes] = {}
    for prompt in prompts:
        try:
            got = await segment(image, prompt)
        except Exception as exc:
            photo.findings.append(Finding(prompt=prompt, coverage=0.0, instances=0,
                                          status=f"failed: {exc.__class__.__name__}"))
            continue
        finding = Finding(prompt=prompt, coverage=got.coverage, instances=got.instances,
                          status=got.status)
        if got.instances and got.mask.any():
            key = f"{Path(name).stem}-{prompt.replace(' ', '-')}"
            pngs[key] = to_png(overlay(arr, got.mask, MASK_COLOURS.get(prompt, (242, 163, 58))))
            finding.overlay = key
        photo.findings.append(finding)
    return photo, pngs


async def segment_all(photos: list[tuple[str, Image.Image]], segment=None
                      ) -> tuple[list[Photo], dict[str, bytes]]:
    results = await asyncio.gather(*(segment_photo(n, im, segment=segment) for n, im in photos))
    pngs: dict[str, bytes] = {}
    out: list[Photo] = []
    for photo, made in results:
        out.append(photo)
        pngs.update(made)
    return out, pngs
