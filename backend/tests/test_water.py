import numpy as np
import pytest

from app.geo import water as w
from app.geo.aoi import Grid, ring_mask


def test_ndwi_sign():
    assert w.ndwi(np.array([1200.0]), np.array([300.0]))[0] > 0.5
    assert w.ndwi(np.array([900.0]), np.array([2500.0]))[0] < 0


def test_clear_mask_drops_cloud_and_shadow():
    scl = np.array([0, 3, 4, 5, 6, 8, 9, 10])
    assert w.clear_mask(scl).tolist() == [False, False, True, True, True, False, False, False]


def test_optical_water_ignores_cloudy_bright_pixels():
    scl = np.array([6, 9, 5])
    green, nir = np.array([1, 1200, 1200.0]), np.array([1, 300, 300.0])
    assert w.optical_water(scl, green, nir).tolist() == [True, False, True]


def test_otsu_splits_bimodal():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-22, 1, 5000), rng.normal(-8, 1, 5000)])
    assert -19 < w.otsu_threshold(x) < -11


def test_radar_threshold_clipped_to_plausible_range():
    vv = np.full((10, 10), 0.2, "float32")
    vv[:, :5] = 0.15  # no real water: Otsu would split land from land
    mask, t = w.radar_water(vv)
    assert w.RADAR_DB_RANGE[0] <= t <= w.RADAR_DB_RANGE[1]
    assert not mask.any()


def test_ring_and_share():
    g = Grid(2000, 10)
    r = ring_mask(g, 250)
    assert r.sum() == pytest.approx(np.pi * 25**2, rel=0.03)
    water = np.zeros((g.side, g.side), bool)
    water[: g.side // 2] = True
    assert w.share_within(water, r) == pytest.approx(0.5, abs=0.02)
    assert w.share_within(water, r, np.zeros_like(r)) is None


def test_distance_and_height():
    water = np.zeros((200, 200), bool)
    water[150:] = True  # 50 px = 500 m south of the centre
    assert w.distance_to_water_m(water, 10) == pytest.approx(500, abs=10)
    dem = np.full((200, 200), 3.0)
    dem[150:] = 0.5
    assert w.height_above_water_m(dem, water) == pytest.approx(2.5)
    assert w.distance_to_water_m(np.zeros((5, 5), bool), 10) is None


def test_ground_level_ignores_rooftops():
    rng = np.random.default_rng(1)
    dem = np.full((200, 200), 2.0) + rng.normal(0, 0.1, (200, 200))
    roofs = rng.random((200, 200)) < 0.6
    dem[roofs] += 12  # 60% of cells are 12 m roofs
    water = np.zeros((200, 200), bool)
    water[180:] = True
    dem[water] = 0.0
    around = ring_mask(Grid(2000, 10), 100)
    h = w.height_above_water_m(dem, water, around_site=around)
    assert h == pytest.approx(2.0, abs=0.5)  # ground, not roof height
    assert w.height_above_water_m(dem, water, around_site=water) == 0.0  # site area all water
