"""Read a square window of a cloud-optimised GeoTIFF around a site."""
from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from .aoi import Grid, Site, bbox

# GDAL has no request timeout by default, so a stalled read of a remote COG blocks
# forever. That is worse than a failure: sources.try_twice only retries exceptions, and a
# hang never raises one, so one bad socket stops a whole validation run. (It did: a 30-site
# baseline sat for 25 minutes on one Azure blob connection at 0% CPU.) With these set, a
# stall becomes an exception, the retry gets its chance, and a layer that really is gone
# ends up a named missing group at low confidence.
GDAL_HTTP_OPTIONS = {
    "GDAL_HTTP_TIMEOUT": "30",         # seconds for the whole request
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


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
    with rasterio.Env(**GDAL_HTTP_OPTIONS), rasterio.open(href) as src:
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
