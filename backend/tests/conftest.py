"""Synthetic site layers: a 2 km grid with a lagoon along the southern edge."""
from __future__ import annotations

import numpy as np
import pytest

from app.geo.aoi import Site
from app.geo.sources import GRID, Optical, Radar, SiteLayers
from app.geo.water import SCL_WATER

WORLDCOVER_URBAN, WORLDCOVER_WATER, WORLDCOVER_WETLAND = 50, 80, 90


def make_layers(water_rows: int = 70, flooded_rows: int = 30, site_height: float = 2.0) -> SiteLayers:
    n = GRID.side  # 200 px, site at row/col 100
    scl = np.full((n, n), 5, dtype="float32")  # 5 = not vegetated (urban)
    scl[n - water_rows:, :] = SCL_WATER
    green = np.full((n, n), 900, "float32")
    nir = np.full((n, n), 2500, "float32")
    green[n - water_rows:, :], nir[n - water_rows:, :] = 1200, 300

    vv = np.full((n, n), 0.15, "float32")  # land, about -8 dB
    vv[n - water_rows - flooded_rows:, :] = 0.005  # water, about -23 dB

    occ = np.zeros((n, n), "float32")
    occ[n - water_rows:, :] = 95

    dem = np.full((n, n), site_height, "float32")
    dem[n - water_rows:, :] = 0.0

    # Built-up city, with a wetland fringe along the shore and the lagoon beyond it.
    lc = np.full((n, n), WORLDCOVER_URBAN, "float32")
    lc[n - water_rows:, :] = WORLDCOVER_WATER
    lc[n - water_rows - 5:n - water_rows, :] = WORLDCOVER_WETLAND

    return SiteLayers(
        site=Site(6.4541, 3.3947), grid=GRID,
        optical=Optical("S2_TEST", "2026-01-24", 0.12, 0.02, scl, green, green, green, nir),
        radar=Radar("S1_TEST", "2025-07-22", vv),
        jrc_occurrence=occ, dem=dem, landcover=lc,
        daily_rain_mm=[("2025-06-10", 140.0), ("2025-06-11", 20.0), ("2024-07-02", 95.0)],
    )


class FakeSource:
    def __init__(self, layers: SiteLayers):
        self.layers = layers
        self.asked: list[tuple[Site, str | None]] = []

    def fetch(self, site: Site, wet_date: str | None = None) -> SiteLayers:
        self.asked.append((site, wet_date))
        return self.layers


@pytest.fixture
def layers() -> SiteLayers:
    return make_layers()
