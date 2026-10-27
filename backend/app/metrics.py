"""Turn raw site layers into named, sourced metrics. Every number in a report comes from here."""
from __future__ import annotations

import numpy as np

from .geo import water as w
from .geo.aoi import ring_mask
from .geo.sources import SiteLayers
from .schemas import Metric

JRC_PERMANENT = 50  # % occurrence above which a pixel counts as regular water
GROUND_RADIUS_M = 100  # area around the site used to estimate ground level
NEIGHBOURHOOD_M = 1000  # the 2 km grid's inscribed circle: the site's surroundings
WORLDCOVER_BUILT_UP = 50  # ESA WorldCover class codes
WORLDCOVER_WETLAND = (90, 95)  # herbaceous wetland, mangroves


def compute_metrics(layers: SiteLayers) -> list[Metric]:
    g = layers.grid
    r250 = ring_mask(g, 250)
    r500 = ring_mask(g, 500)
    out: list[Metric] = []

    # History is the check on a single optical date: tall buildings cast shadows that
    # Sentinel-2's classifier labels water (14.6% within 250 m of Lagos Island, where JRC
    # has never seen water). None when there is no usable history to check against.
    ever_water = None
    if layers.jrc_occurrence is not None and bool(np.isfinite(layers.jrc_occurrence).any()):
        ever_water = np.nan_to_num(layers.jrc_occurrence) > 0

    water_now = None
    if layers.optical is not None:
        o = layers.optical
        water_now = w.optical_water(o.scl, o.green, o.nir)
        clear = w.clear_mask(o.scl, w.CLOUD_BUFFER_PX)
        confirmed = water_now if ever_water is None else water_now & ever_water
        share = w.share_within(confirmed, r250, clear)
        if share is not None:
            if ever_water is None:
                label = "Water within 250 m on a clear date (unconfirmed)"
                source = f"Sentinel-2 {o.date}"
            else:
                label = "Water within 250 m on a clear date, confirmed by history"
                source = f"Sentinel-2 {o.date} + JRC Global Surface Water"
            out.append(Metric(id="m_water_now_250", label=label, value=100 * share, unit="%", source=source))
        out.append(Metric(id="m_tile_cloud", label="Cloud over the site image", value=100 * o.tile_cloud,
                          unit="%", source=f"Sentinel-2 {o.date}", decimals=0))

    if layers.radar is not None:
        rd = layers.radar
        wet, t_db = w.radar_water(rd.vv)
        share = w.share_within(wet, r250, rd.vv > 0)
        if share is not None:
            out.append(Metric(id="m_wet_water_250", label="Area under water within 250 m on a wet date",
                              value=100 * share, unit="%",
                              source=f"Sentinel-1 {rd.date}, threshold {t_db:.1f} dB"))

    regular_water = None
    if layers.jrc_occurrence is not None:
        occ = layers.jrc_occurrence
        valid = np.isfinite(occ)
        mean = w.mean_within(np.nan_to_num(occ), r500, valid)
        if mean is not None:
            out.append(Metric(id="m_water_history_500", label="How often water was present within 500 m, since 1984",
                              value=mean, unit="%", source="JRC Global Surface Water"))
        regular_water = np.nan_to_num(occ) >= JRC_PERMANENT

    # Distance and height use regular water from 37 years of JRC history. A single optical
    # image is only the fallback: one date's water can be shadow, or a temporary puddle.
    if regular_water is not None and regular_water.any():
        ref_water, ref_source = regular_water, "JRC Global Surface Water"
    else:
        ref_water, ref_source = water_now, "Sentinel-2 (no JRC water in tile)"
    if ref_water is not None:
        d = w.distance_to_water_m(ref_water, g.res_m)
        if d is None:
            # No water anywhere in view is a measurement, not a gap. The nearest regular
            # water is at least the distance from the site to the edge of what we looked
            # at, so report that. Dropping the metric instead would rescale its weight
            # onto the other features, which is how four verified floods scored Low.
            # Height above water is the one thing we genuinely cannot measure here.
            edge_m = g.size_m / 2
            seen = "JRC Global Surface Water and Sentinel-2" if regular_water is not None else "Sentinel-2"
            out.append(Metric(id="m_dist_water", label=f"No regular water within {edge_m / 1000:g} km",
                              value=edge_m, unit="m", decimals=0,
                              source=f"{seen}: no water in the {g.size_m / 1000:g} km grid"))
        else:
            out.append(Metric(id="m_dist_water", label="Distance to nearest regular water", value=d, unit="m",
                              source=ref_source, decimals=0))
            if layers.dem is not None:
                h = w.height_above_water_m(layers.dem, ref_water, around_site=ring_mask(g, GROUND_RADIUS_M))
                if h is not None:
                    out.append(Metric(id="m_height_above_water", label="Ground height above nearest water", value=h,
                                      unit="m",
                                      source=f"Copernicus DEM GLO-30, low point within {GROUND_RADIUS_M} m"))

    # Pluvial risk: where rain collects, and whether it can soak away.
    if layers.dem is not None:
        low = w.low_spot_m(layers.dem, ring_mask(g, GROUND_RADIUS_M), ring_mask(g, NEIGHBOURHOOD_M))
        if low is not None:
            out.append(Metric(
                id="m_local_low_spot", value=low, unit="m",
                label=f"Ground at the site against the median within {NEIGHBOURHOOD_M / 1000:g} km "
                      "(negative = a dip)",
                source=f"Copernicus DEM GLO-30, {GROUND_RADIUS_M} m low point vs the surrounding median"))

    if layers.landcover is not None:
        lc = layers.landcover.astype(int)
        known = lc > 0  # 0 is outside the tile, not a class
        built = w.share_within(lc == WORLDCOVER_BUILT_UP, r500, known)
        if built is not None:
            out.append(Metric(id="m_built_up_500", label="Built-up land within 500 m", value=100 * built,
                              unit="%", source="ESA WorldCover 10 m", decimals=0))
        wetland = w.share_within(np.isin(lc, WORLDCOVER_WETLAND), r500, known)
        if wetland is not None:
            out.append(Metric(id="m_wetland_500", label="Wetland or mangrove within 500 m", value=100 * wetland,
                              unit="%", source="ESA WorldCover 10 m"))

    if layers.daily_rain_mm:
        day, mm = max(layers.daily_rain_mm, key=lambda x: x[1])
        out.append(Metric(id="m_max_daily_rain", label=f"Heaviest daily rainfall (on {day})", value=mm,
                          unit="mm", source="Open-Meteo archive", decimals=0))
    return out
