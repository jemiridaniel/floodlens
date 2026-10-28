"""Fetch the open data layers for a site (network). Each layer is optional: cloud,
missing scenes or outages leave it as None, and the score works with what it has."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np
from rasterio.enums import Resampling

from ..config import OPEN_METEO_ARCHIVE, STAC_URL
from .aoi import Grid, Site, bbox
from .raster import read_window
from .water import clear_mask

GRID = Grid(size_m=2000, res_m=10)  # 200 x 200 px at Sentinel-2 native resolution

# Nothing here may block forever. GDAL covers the raster reads (geo/raster.py), but the
# STAC searches and Planetary Computer's SAS signing go through requests, which has no
# default timeout at all: planetary_computer calls session.get(url, headers=...) with none,
# so one stalled token request hung a whole validation run for an hour at 0% CPU.
STAC_TIMEOUT_S = 30
STAC_RETRIES = 2
REQUESTS_TIMEOUT_S = 30


def bound_requests_timeout(seconds: float = REQUESTS_TIMEOUT_S) -> None:
    """Give every requests call a default timeout, once, process-wide.

    This is the only way to bound planetary_computer's signing call: it builds its own
    Session per call and passes no timeout, so there is nothing to configure. Callers that
    set their own timeout keep it.
    """
    import requests

    if getattr(requests.Session.request, "_floodlens_bounded", False):
        return
    original = requests.Session.request

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", seconds)
        return original(self, method, url, **kwargs)

    request._floodlens_bounded = True
    requests.Session.request = request


@dataclass
class Optical:
    item_id: str
    date: str
    tile_cloud: float  # 0-1, over this grid
    scl: np.ndarray
    blue: np.ndarray
    green: np.ndarray
    red: np.ndarray
    nir: np.ndarray


@dataclass
class Radar:
    item_id: str
    date: str
    vv: np.ndarray  # linear backscatter


@dataclass
class SiteLayers:
    site: Site
    grid: Grid
    optical: Optical | None = None
    radar: Radar | None = None
    jrc_occurrence: np.ndarray | None = None  # 0-100 % of years with water, NaN = no data
    dem: np.ndarray | None = None  # metres
    landcover: np.ndarray | None = None  # ESA WorldCover class codes, 0 = no data
    daily_rain_mm: list[tuple[str, float]] = field(default_factory=list)
    item_ids: dict[str, str] = field(default_factory=dict)  # layer -> STAC item id, for evidence
    notes: list[str] = field(default_factory=list)


class LayerSource(Protocol):
    def fetch(self, site: Site, wet_date: str | None = None) -> SiteLayers: ...


def _catalog():
    import planetary_computer
    import pystac_client
    from pystac_client.stac_api_io import StacApiIO

    bound_requests_timeout()  # covers planetary_computer's SAS signing call
    # max_retries rides on the StacApiIO; the timeout must go to Client.open, which
    # otherwise overwrites stac_io.timeout with its own default of None.
    stac_io = StacApiIO(max_retries=STAC_RETRIES)
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace,
                                     stac_io=stac_io, timeout=STAC_TIMEOUT_S)


def _search(catalog, collection: str, site: Site, dates: str | None = None, query: dict | None = None):
    kwargs = {"collections": [collection], "bbox": bbox(site, GRID.size_m), "max_items": 50}
    if dates:
        kwargs["datetime"] = dates
    if query:
        kwargs["query"] = query
    return list(catalog.search(**kwargs).items())


def clearest_optical(catalog, site: Site, dates: str, max_checks: int = 12) -> Optical | None:
    """The Sentinel-2 scene with the least cloud over this grid (not over the whole scene)."""
    items = _search(catalog, "sentinel-2-l2a", site, dates, {"eo:cloud_cover": {"lt": 60}})
    if not items:
        return None
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    small = Grid(GRID.size_m, GRID.size_m / 64)
    best = None
    for item in items[:max_checks]:
        scl = read_window(item.assets["SCL"].href, site, small, Resampling.nearest)
        cloud = 1 - float(clear_mask(scl).mean())
        if best is None or cloud < best[1]:
            best = (item, cloud)
        if cloud < 0.01:
            break
    item, cloud = best

    def band(name: str, rs=Resampling.bilinear) -> np.ndarray:
        return read_window(item.assets[name].href, site, GRID, rs)

    return Optical(
        item_id=item.id,
        date=item.properties["datetime"][:10],
        tile_cloud=round(cloud, 3),
        scl=band("SCL", Resampling.nearest),
        blue=band("B02"),
        green=band("B03"),
        red=band("B04"),
        nir=band("B08"),
    )


RADAR_WINDOW_DAYS = 12  # two Sentinel-1 revisit cycles: a named flood date is usually reachable


def radar_near(catalog, site: Site, day: str, window_days: int = RADAR_WINDOW_DAYS) -> Radar | None:
    """First Sentinel-1 RTC scene on or after `day` (within window_days)."""
    d0 = dt.date.fromisoformat(day)
    dates = f"{d0.isoformat()}/{(d0 + dt.timedelta(days=window_days)).isoformat()}"
    items = _search(catalog, "sentinel-1-rtc", site, dates)
    if not items:
        return None
    item = min(items, key=lambda i: i.properties["datetime"])
    return Radar(item_id=item.id, date=item.properties["datetime"][:10],
                 vv=read_window(item.assets["vv"].href, site, GRID))


def covering_item(items, site: Site):
    """The tiled item whose footprint contains the site (search can also return neighbours)."""
    for item in items:
        b = item.bbox
        if b and b[0] <= site.lon <= b[2] and b[1] <= site.lat <= b[3]:
            return item
    return items[0] if items else None


def jrc_occurrence(catalog, site: Site) -> tuple[np.ndarray, str] | None:
    item = covering_item(_search(catalog, "jrc-gsw", site), site)
    if item is None:
        return None
    occ = read_window(item.assets["occurrence"].href, site, GRID, Resampling.nearest)
    occ[occ > 100] = np.nan  # 255 = no data
    return occ, item.id


def dem(catalog, site: Site) -> tuple[np.ndarray, str] | None:
    item = covering_item(_search(catalog, "cop-dem-glo-30", site), site)
    if item is None:
        return None
    return read_window(item.assets["data"].href, site, GRID), item.id


def landcover(catalog, site: Site, grid: Grid = GRID) -> tuple[np.ndarray, str] | None:
    """ESA WorldCover 10 m land cover classes (verified against the STAC catalog: collection
    esa-worldcover, asset "map", classes 50 built-up / 90 herbaceous wetland / 95 mangroves).
    One item per product year, so take the most recent."""
    items = _search(catalog, "esa-worldcover", site)
    items.sort(key=lambda i: i.properties.get("start_datetime") or i.properties.get("datetime") or "",
               reverse=True)
    item = covering_item(items, site)
    if item is None:
        return None
    return read_window(item.assets["map"].href, site, grid, Resampling.nearest), item.id


def daily_rain(site: Site, start: str, end: str, client: httpx.Client | None = None) -> list[tuple[str, float]]:
    """Daily precipitation (mm) from the Open-Meteo historical archive."""
    params = {
        "latitude": site.lat, "longitude": site.lon,
        "start_date": start, "end_date": end,
        "daily": "precipitation_sum", "timezone": "UTC",
    }
    c = client or httpx.Client(timeout=30)
    try:
        r = c.get(OPEN_METEO_ARCHIVE, params=params)
        r.raise_for_status()
        daily = r.json()["daily"]
    finally:
        if client is None:
            c.close()
    return [(d, float(p)) for d, p in zip(daily["time"], daily["precipitation_sum"], strict=True) if p is not None]


def wettest_days(rain: list[tuple[str, float]], n: int = 3, min_gap_days: int = 10) -> list[tuple[str, float]]:
    """Top-n rainiest days, at least min_gap_days apart (so one storm counts once)."""
    picked: list[tuple[str, float]] = []
    for day, mm in sorted(rain, key=lambda x: -x[1]):
        d = dt.date.fromisoformat(day)
        if all(abs((d - dt.date.fromisoformat(p)).days) >= min_gap_days for p, _ in picked):
            picked.append((day, mm))
        if len(picked) == n:
            break
    return picked


def try_twice(label: str, fn, notes: list[str]):
    """Run fn, retrying once. Planetary Computer fails transiently, and a layer lost to a
    blip silently rescales the score, so it is worth one more go before giving up."""
    for attempt in (1, 2):
        try:
            return fn()
        except Exception as exc:
            if attempt == 2:
                notes.append(f"{label} unavailable after 2 tries: {exc.__class__.__name__}: {exc}")
    return None


class PlanetaryComputerSource:
    """Real data from Microsoft Planetary Computer + Open-Meteo."""

    def __init__(self, optical_dates: str | None = None, rain_years: int = 5):
        self.optical_dates = optical_dates
        self.rain_years = rain_years

    def fetch(self, site: Site, wet_date: str | None = None) -> SiteLayers:
        layers = SiteLayers(site=site, grid=GRID)
        catalog = _catalog()
        today = dt.date.today()

        optical_dates = self.optical_dates or f"{today.year - 1}-11-01/{today.isoformat()}"
        layers.optical = try_twice("optical",
                                   lambda: clearest_optical(catalog, site, optical_dates), layers.notes)
        for name, fn in (("jrc_occurrence", jrc_occurrence), ("dem", dem), ("landcover", landcover)):
            got = try_twice(name, lambda fn=fn: fn(catalog, site), layers.notes)
            if got is not None:
                setattr(layers, name, got[0])
                layers.item_ids[name] = got[1]

        start = today.replace(year=today.year - self.rain_years).isoformat()
        end = (today - dt.timedelta(days=7)).isoformat()  # archive lags a few days
        layers.daily_rain_mm = try_twice("rainfall", lambda: daily_rain(site, start, end),
                                         layers.notes) or []

        # A date the site is known to have flooded beats inferring one from rainfall:
        # heavy rain somewhere in the grid is not the same as water at the site.
        for day, _ in [(wet_date, None)] if wet_date else wettest_days(layers.daily_rain_mm):
            layers.radar = try_twice("radar", lambda day=day: radar_near(catalog, site, day),
                                     layers.notes)
            if layers.radar:
                break
        if wet_date and layers.radar is None:
            layers.notes.append(f"no Sentinel-1 scene within {RADAR_WINDOW_DAYS} days of {wet_date}")
        return layers
