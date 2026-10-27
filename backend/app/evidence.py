"""Render the images an assessment rests on: one PNG per named view.

The agent looks at these through view_evidence, and scripts/assess_site.py writes
the same bytes to disk, so what Spark sees is exactly what a reviewer sees.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .geo.raster import stretch
from .geo.sources import SiteLayers
from .geo.water import clear_mask, optical_water, radar_water

VIEW_HELP = {
    "optical": "True-colour Sentinel-2 on the clearest recent date, 2 km across, site at the centre.",
    "water_now": "White = water detected on that clear date, before the history check.",
    "water_history": "JRC 1984-present: brighter = water present in more years.",
    "water_wet": "White = water detected by Sentinel-1 radar on the wettest recent date.",
}
PNG_SIZE = 800  # the grid is 200 px; upscale so the model sees crisp pixel edges


def _png(a: np.ndarray) -> bytes:
    img = Image.fromarray(a).resize((PNG_SIZE, PNG_SIZE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(mask: np.ndarray) -> bytes:
    return _png((mask * 255).astype(np.uint8))


def evidence_pngs(layers: SiteLayers) -> dict[str, bytes]:
    """Every view this site has data for, name -> PNG bytes. Missing layers are skipped."""
    out: dict[str, bytes] = {}
    if layers.optical is not None:
        o = layers.optical
        clear = clear_mask(o.scl)
        rgb = np.dstack([stretch(b, 1, 99, clear) for b in (o.red, o.green, o.blue)])
        out["optical"] = _png(rgb)
        out["water_now"] = _mask_png(optical_water(o.scl, o.green, o.nir))
    if layers.jrc_occurrence is not None:
        occ = np.clip(np.nan_to_num(layers.jrc_occurrence), 0, 100)
        out["water_history"] = _png((occ * 2.55).astype(np.uint8))
    if layers.radar is not None:
        out["water_wet"] = _mask_png(radar_water(layers.radar.vv)[0])
    return out


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
