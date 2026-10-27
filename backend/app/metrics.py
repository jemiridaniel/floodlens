"""Turn raw site layers into named, sourced metrics. Every number in a report comes from here."""
from __future__ import annotations

import numpy as np

from .geo import water as w
from .geo.aoi import ring_mask
from .geo.sources import SiteLayers
from .schemas import Metric

JRC_PERMANENT = 50  # % occurrence above which a pixel counts as regular water
GROUND_RADIUS_M = 100  # area around the site used to estimate ground level


def compute_metrics(layers: SiteLayers) -> list[Metric]:
    g = layers.grid
    r250 = ring_mask(g, 250)
    r500 = ring_mask(g, 500)
    out: list[Metric] = []

    water_now = None
    if layers.optical is not None:
        o = layers.optical
        water_now = w.optical_water(o.scl, o.green, o.nir)
        clear = w.clear_mask(o.scl)
        share = w.share_within(water_now, r250, clear)
        if share is not None:
            out.append(Metric(id="m_water_now_250", label="Water within 250 m, clear dry-season image",
                              value=100 * share, unit="%", source=f"Sentinel-2 {o.date}"))
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

    # Distance and height use the best water map available: regular (JRC) water, plus what is wet now.
    ref_water = None
    for m in (regular_water, water_now):
        if m is not None:
            ref_water = m if ref_water is None else ref_water | m
    if ref_water is not None:
        d = w.distance_to_water_m(ref_water, g.res_m)
        if d is not None:
            out.append(Metric(id="m_dist_water", label="Distance to nearest regular water", value=d, unit="m",
                              source="JRC + Sentinel-2", decimals=0))
        if layers.dem is not None:
            h = w.height_above_water_m(layers.dem, ref_water, around_site=ring_mask(g, GROUND_RADIUS_M))
            if h is not None:
                out.append(Metric(id="m_height_above_water", label="Ground height above nearest water", value=h,
                                  unit="m", source=f"Copernicus DEM GLO-30, low point within {GROUND_RADIUS_M} m"))

    if layers.daily_rain_mm:
        day, mm = max(layers.daily_rain_mm, key=lambda x: x[1])
        out.append(Metric(id="m_max_daily_rain", label=f"Heaviest daily rainfall (on {day})", value=mm,
                          unit="mm", source="Open-Meteo archive", decimals=0))
    return out
