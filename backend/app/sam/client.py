"""SAM 3.1 on Meta Model API: segment one noun phrase in one image.

Used for site photos and the 5 km water context map. Day 0 testing showed SAM
does not resolve water or buildings in 2 km, 10 m satellite tiles, so it is not
used to measure water at the site.
"""
from __future__ import annotations

import base64
import io
from contextlib import aclosing
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import decode_mask_to_raster, image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI
from PIL import Image

from ..config import META_BASE_URL, get_settings

REQUEST_TIMEOUT_S = 120  # a segmentation that stalls must not hold up an assessment


@dataclass
class Segmentation:
    prompt: str
    mask: np.ndarray  # union of all instances, full image size
    instances: int
    status: str

    @property
    def coverage(self) -> float:
        return float(self.mask.mean())


def png_data_url(image: np.ndarray | Image.Image) -> str:
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def _events(stream):
    """Close each output-text lane for the parser (same shim as Meta's SAM cookbook)."""
    done: set = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (data.get("type") == "response.content_part.done" and lane not in done
                and isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)):
            done.add(lane)
            yield {"type": "response.output_text.done", "item_id": data.get("item_id"),
                   "output_index": data.get("output_index"),
                   "content_index": data.get("content_index"), "text": part["text"]}


def union_mask(records, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Place each instance mask at its box on a full-size canvas and OR them together."""
    canvas = np.zeros(shape, dtype=bool)
    boxes = {r.object_id: r for r in records if r.kind == "box"}
    n = 0
    for rec in records:
        if rec.kind != "mask":
            continue
        m = rec.mask
        crop = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8)[: m.width * m.height]
        crop = crop.reshape(m.height, m.width).astype(bool)
        box = boxes.get(rec.object_id)
        left, top = (int(box.left), int(box.top)) if box else (0, 0)
        h, w = min(m.height, shape[0] - top), min(m.width, shape[1] - left)
        if h > 0 and w > 0:
            canvas[top:top + h, left:left + w] |= crop[:h, :w]
        n += 1
    return canvas, n


async def segment(image: np.ndarray | Image.Image, prompt: str, api: AsyncOpenAI | None = None) -> Segmentation:
    settings = get_settings()
    own = api is None
    if own:
        if not settings.model_api_key:
            raise RuntimeError("MODEL_API_KEY is not set")
        api = AsyncOpenAI(base_url=META_BASE_URL, api_key=settings.model_api_key,
                          timeout=REQUEST_TIMEOUT_S, max_retries=1)
    arr = np.asarray(image)
    try:
        stream = await api.responses.create(
            model=settings.sam_model, stream=True,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": png_data_url(image)},
            ]}],
        )
        async with stream, aclosing(_events(stream)) as events:
            result = await parse_responses_stream(events, image_segmentation_format()).final_result()
    finally:
        if own:
            await api.close()
    mask, n = union_mask(result.records, arr.shape[:2])
    return Segmentation(prompt=prompt, mask=mask, instances=n, status=result.outcome.status)


def overlay(base: np.ndarray, mask: np.ndarray, colour=(63, 169, 245), alpha: float = 0.55) -> np.ndarray:
    out = base.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(colour) * alpha
    return out.astype(np.uint8)
