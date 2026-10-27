"""Read a square window of a cloud-optimised GeoTIFF around a site."""
from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from .aoi import Grid, Site, bbox


def read_window(
    href: str,
    site: Site,
    grid: Grid,
    resampling: Resampling = Resampling.bilinear,
    band: int = 1,
) -> np.ndarray:
    """Read one band clipped to the grid around the site, resampled to grid.side x grid.side.

    Pixels outside the source are 0. The result is float32.
    """
    box = bbox(site, grid.size_m)
    with rasterio.open(href) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x0, y0 = tf.transform(box[0], box[1])
        x1, y1 = tf.transform(box[2], box[3])
        win = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), src.transform)
        return src.read(
            band,
            window=win,
            out_shape=(grid.side, grid.side),
            resampling=resampling,
            boundless=True,
            fill_value=0,
        ).astype("float32")


def stretch(a: np.ndarray, lo: float = 1, hi: float = 99, valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile stretch to uint8, computed over valid (e.g. cloud-free) non-zero pixels only."""
    sel = a > 0
    if valid is not None:
        sel &= valid
    if not sel.any():
        return np.zeros(a.shape, dtype=np.uint8)
    p0, p1 = np.percentile(a[sel], [lo, hi])
    return (np.clip((a - p0) / max(p1 - p0, 1e-6), 0, 1) * 255).astype(np.uint8)
