"""Water detection and site metrics. Pure numpy: no network, easy to test.

Lessons from the Day 0 SAM test (see README):
- Cloud is the norm over Lagos, so optical water uses Sentinel-2's own scene
  classification (SCL) and only cloud-free pixels count.
- A fixed radar threshold of -18 dB under-counted water (8.5% vs 44% optical),
  so radar water uses an Otsu threshold fitted to each scene, kept within a
  physically plausible range.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Sentinel-2 L2A scene classification codes
SCL_NODATA, SCL_SATURATED, SCL_SHADOW, SCL_CLOUD_SHADOW = 0, 1, 2, 3
SCL_WATER = 6
SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS = 8, 9, 10
SCL_UNUSABLE = (SCL_NODATA, SCL_SATURATED, SCL_CLOUD_SHADOW, SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS)

RADAR_DB_RANGE = (-24.0, -14.0)  # open water in Sentinel-1 VV sits below roughly -15 dB


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters normalised difference water index, (G - NIR) / (G + NIR)."""
    g = green.astype("float32")
    n = nir.astype("float32")
    return (g - n) / np.maximum(g + n, 1e-6)


CLOUD_BUFFER_PX = 10  # 100 m at 10 m: cloud edges and shadows are often mislabelled as water


def clear_mask(scl: np.ndarray, buffer_px: int = 0) -> np.ndarray:
    """Pixels Sentinel-2 classified as usable (not cloud, shadow, cirrus or no data),
    optionally also dropping a buffer of buffer_px around every unusable pixel."""
    bad = np.isin(scl.astype(int), SCL_UNUSABLE)
    if buffer_px > 0 and bad.any():
        bad = ndimage.binary_dilation(bad, iterations=buffer_px)
    return ~bad


def optical_water(scl: np.ndarray, green: np.ndarray, nir: np.ndarray, ndwi_min: float = 0.1,
                  buffer_px: int = CLOUD_BUFFER_PX) -> np.ndarray:
    """Water from Sentinel-2: SCL water class or NDWI above ndwi_min, on pixels clear of
    cloud and shadow by at least buffer_px. (In the first Lagos run, cloud shadow over
    the city was labelled water 72 m from a site that is 500+ m from any real water.)"""
    usable = clear_mask(scl, buffer_px)
    return ((scl.astype(int) == SCL_WATER) | (ndwi(green, nir) > ndwi_min)) & usable


# Cloud for REPORTING only: it captions the picture so a reader can judge what they are
# looking at. It never touches a scored metric -- optical_water still uses clear_mask with
# its 100 m buffer, and the scoring is frozen.
#
# SCL alone is not enough. Over the bright built-up surfaces that cover most of Lagos it
# labelled two obvious cloud blobs "not vegetated": 0.03% cloud on a picture visibly 2%
# clouded. So SCL's cloud classes are OR'd with a brightness test. Cloud is bright in both
# the visible and the near infrared and close to spectrally flat; a bright roof is neither
# that bright nor that flat. The thresholds were chosen against the Okota 2026-01-24 scene:
# they catch 100% of the visible cloud at 3.1% of the frame, where lowering blue to 0.30
# reaches 6.6% and starts sweeping in concrete.
SCL_CLOUD_CLASSES = (SCL_CLOUD_SHADOW, SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS)
CLOUD_BLUE_MIN = 0.40  # surface reflectance
CLOUD_NIR_MIN = 0.30
CLOUD_FLATNESS_MAX = 0.25  # |blue - red| / red
REFLECTANCE_SCALE = 10_000  # Sentinel-2 L2A digital numbers


def cloud_mask(scl: np.ndarray, blue: np.ndarray, red: np.ndarray,
               nir: np.ndarray) -> np.ndarray:
    """Cloud, cloud shadow and cirrus, as a reader would see them in the picture."""
    flagged = np.isin(scl.astype(int), SCL_CLOUD_CLASSES)
    b, r, n = (band.astype("float32") / REFLECTANCE_SCALE for band in (blue, red, nir))
    flat = np.abs(b - r) / np.maximum(r, 1e-6)
    bright = (b > CLOUD_BLUE_MIN) & (n > CLOUD_NIR_MIN) & (flat < CLOUD_FLATNESS_MAX)
    return flagged | bright


def cloud_fraction(scl: np.ndarray, blue: np.ndarray, red: np.ndarray, nir: np.ndarray,
                   area: np.ndarray | None = None) -> float:
    """Share of `area` (default: the whole frame) under cloud, 0-1."""
    mask = cloud_mask(scl, blue, red, nir)
    return float(mask.mean() if area is None else mask[area].mean() if area.any() else 0.0)


# When the brightness test finds most of the cloud, the scene classifier has missed it --
# and a classifier that misses cloud also misses that cloud's shadow, which no brightness
# test can find because shadow is dark. The figure is then a floor, not a measurement, and
# the report should say "partly cloudy" rather than quote a number that looks exact.
SCL_SHARE_FOR_CONFIDENCE = 0.75


def cloud_report(scl: np.ndarray, blue: np.ndarray, red: np.ndarray, nir: np.ndarray,
                 area: np.ndarray | None = None) -> tuple[float, bool]:
    """(share under cloud, is the share trustworthy).

    Cloud shadow is counted when Sentinel-2's own classifier flags it. It cannot be
    counted when the classifier missed the cloud overhead, so that case is reported as
    uncertain instead of quietly under-stating.
    """
    flagged = np.isin(scl.astype(int), SCL_CLOUD_CLASSES)
    total = cloud_mask(scl, blue, red, nir)
    if area is not None:
        if not area.any():
            return 0.0, True
        flagged, total = flagged[area], total[area]
    share = float(total.mean())
    if share == 0:
        return 0.0, True
    return share, bool(flagged.sum() >= SCL_SHARE_FOR_CONFIDENCE * total.sum())


def to_db(linear: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(linear.astype("float32"), 1e-6))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for a 1-D sample (maximises between-class variance)."""
    values = values[np.isfinite(values)]
    if values.size < 2 or values.min() == values.max():
        return float(values.mean()) if values.size else float("nan")
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = ((hist * centres).sum() - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


def radar_water(vv_linear: np.ndarray) -> tuple[np.ndarray, float]:
    """Water from Sentinel-1 VV backscatter. Returns (mask, threshold_db)."""
    valid = vv_linear > 0
    db = to_db(vv_linear)
    t = otsu_threshold(db[valid]) if valid.any() else RADAR_DB_RANGE[0]
    t = float(np.clip(t, *RADAR_DB_RANGE))
    return (db < t) & valid, t


def share_within(mask: np.ndarray, ring: np.ndarray, valid: np.ndarray | None = None) -> float | None:
    """Share (0-1) of the ring covered by mask, counting only valid pixels. None if nothing valid."""
    area = ring if valid is None else ring & valid
    n = int(area.sum())
    return None if n == 0 else float(mask[area].sum() / n)


def mean_within(values: np.ndarray, ring: np.ndarray, valid: np.ndarray | None = None) -> float | None:
    area = ring if valid is None else ring & valid
    return None if not area.any() else float(values[area].mean())


def distance_to_water_m(water: np.ndarray, res_m: float) -> float | None:
    """Distance from the grid centre to the nearest water pixel, in metres. None if no water."""
    if not water.any():
        return None
    dist = ndimage.distance_transform_edt(~water) * res_m
    c = water.shape[0] // 2
    return float(dist[c, c])


def ground_level_m(dem: np.ndarray, area: np.ndarray, pct: float = 10) -> float | None:
    """Estimated ground level in an area of a surface model.

    Copernicus GLO-30 is a surface model: in dense cities it partly measures roofs.
    A low percentile of the cells around the site is a better ground estimate than
    the single cell at the site.
    """
    vals = dem[area & np.isfinite(dem) & (dem != 0)]
    return None if vals.size == 0 else float(np.percentile(vals, pct))


def low_spot_m(dem: np.ndarray, around_site: np.ndarray, neighbourhood: np.ndarray) -> float | None:
    """Ground level at the site minus the median ground level around it, in metres.

    Negative means the site sits in a dip, where rain collects even with no river or
    lagoon nearby. This is the pluvial counterpart to height_above_water_m: Lagos floods
    that have no water body in view still have somewhere for the water to pool.
    """
    near = ground_level_m(dem, around_site)
    around = ground_level_m(dem, neighbourhood, pct=50)
    return None if near is None or around is None else near - around


def height_above_water_m(dem: np.ndarray, water: np.ndarray, around_site: np.ndarray | None = None) -> float | None:
    """Ground level near the site minus the median surface of nearby water. None if no water."""
    if not water.any():
        return None
    if around_site is None:
        c = dem.shape[0] // 2
        ground = float(dem[c, c])
    else:
        ground = ground_level_m(dem, around_site & ~water)
        if ground is None:
            return 0.0  # the site area is all water
    return ground - float(np.median(dem[water]))
