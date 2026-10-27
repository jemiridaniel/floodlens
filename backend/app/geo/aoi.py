"""Area of interest around a collateral site: bounding boxes and assessment rings."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RING_RADII_M = (250, 500)


@dataclass(frozen=True)
class Site:
    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not (-90 <= self.lat <= 90 and -180 <= self.lon <= 180):
            raise ValueError(f"Invalid coordinates: {self.lat}, {self.lon}")


@dataclass(frozen=True)
class Grid:
    """A square pixel grid centred on the site."""

    size_m: float
    res_m: float

    @property
    def side(self) -> int:
        return int(round(self.size_m / self.res_m))


def bbox(site: Site, size_m: float) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) of a size_m x size_m square centred on the site."""
    half = size_m / 2
    dlat = half / 111_320
    dlon = half / (111_320 * math.cos(math.radians(site.lat)))
    return site.lon - dlon, site.lat - dlat, site.lon + dlon, site.lat + dlat


def distance_m(a: Site, b: Site) -> float:
    """Great-circle distance between two sites, in metres."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat, dlon = lat2 - lat1, math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(h)))


def distance_grid(grid: Grid) -> np.ndarray:
    """Distance in metres from the site (grid centre) to each pixel centre."""
    n = grid.side
    c = (np.arange(n) + 0.5 - n / 2) * grid.res_m
    xx, yy = np.meshgrid(c, c)
    return np.hypot(xx, yy)


def ring_mask(grid: Grid, radius_m: float) -> np.ndarray:
    """Boolean disc of the given radius around the site."""
    return distance_grid(grid) <= radius_m
