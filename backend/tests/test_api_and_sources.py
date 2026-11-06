import types

import httpx
import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from pyproj import Transformer
from rasterio.transform import from_origin

from app.geo.aoi import Grid, Site
from app.geo.raster import read_window, stretch
from app.geo.sources import daily_rain, wettest_days
from app.main import app
from app.sam import client as sam


@pytest.fixture
def api():
    with TestClient(app) as client:
        yield client


def test_health(api):
    assert api.get("/health").json()["status"] == "ok"


def test_the_browser_app_is_allowed_and_nothing_else(api):
    """The Next.js dev server talks to this API; no other origin may."""
    allowed = api.get("/health", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    other = api.get("/health", headers={"Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in other.headers


def test_wettest_days_spaced_apart():
    rain = [("2025-06-10", 140.0), ("2025-06-11", 130.0), ("2024-07-02", 95.0), ("2023-05-01", 60.0)]
    assert [d for d, _ in wettest_days(rain, n=3)] == ["2025-06-10", "2024-07-02", "2023-05-01"]


def test_daily_rain_parses_open_meteo():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["daily"] == "precipitation_sum"
        return httpx.Response(200, json={"daily": {"time": ["2025-06-10", "2025-06-11"],
                                                   "precipitation_sum": [140.2, None]}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        assert daily_rain(Site(6.45, 3.39), "2025-06-10", "2025-06-11", client=c) == [("2025-06-10", 140.2)]


def test_read_window_from_utm_geotiff(tmp_path):
    x, y = Transformer.from_crs(4326, 32631, always_xy=True).transform(3.3947, 6.4541)
    path = tmp_path / "band.tif"
    data = (np.arange(1000 * 1000) % 5000).reshape(1000, 1000).astype("uint16") + 1
    with rasterio.open(path, "w", driver="GTiff", width=1000, height=1000, count=1, dtype="uint16",
                       crs="EPSG:32631", transform=from_origin(x - 5000, y + 5000, 10, 10)) as dst:
        dst.write(data, 1)
    band = read_window(str(path), Site(6.4541, 3.3947), Grid(2000, 10))
    assert band.shape == (200, 200) and band.min() > 0
    assert stretch(band).dtype == np.uint8


def test_sam_union_mask_places_crops_at_boxes(monkeypatch):
    monkeypatch.setattr(sam, "decode_mask_to_raster", lambda m: bytes([1, 1, 0, 0, 1, 1]))
    R = types.SimpleNamespace
    records = [R(kind="box", object_id="0", left=5, top=4, right=8, bottom=6),
               R(kind="mask", object_id="0", mask=R(width=3, height=2))]
    canvas, n = sam.union_mask(records, (10, 10))
    assert n == 1 and canvas.sum() == 4 and canvas[4, 5] and canvas[5, 7] and not canvas[4, 7]
    records[0].left = records[0].top = 9  # clipped at the edge
    assert sam.union_mask(records, (10, 10))[0].sum() == 1


def test_covering_item_picks_tile_containing_site():
    from app.geo.sources import covering_item
    R = types.SimpleNamespace
    near = R(id="neighbour", bbox=[10, 0, 20, 10])
    home = R(id="home", bbox=[0, 0, 10, 10])
    assert covering_item([near, home], Site(6.45, 3.39)).id == "home"
    assert covering_item([], Site(6.45, 3.39)) is None


def test_cloud_cover_is_the_whole_tile_not_the_window():
    """The window around a site is often clear inside a cloudy pass, so the 64x64 window
    estimate reads 0% and misleads. The reported figure is the scene's own eo:cloud_cover."""
    from app.geo.sources import tile_cloud_fraction

    class Item:
        def __init__(self, cover):
            self.properties = {"eo:cloud_cover": cover}

    assert tile_cloud_fraction(Item(0)) == 0.0
    assert tile_cloud_fraction(Item(37.4)) == pytest.approx(0.374)
    assert tile_cloud_fraction(Item(100)) == 1.0
    assert tile_cloud_fraction(Item(None)) == 0.0  # missing metadata is not "cloudy"
    assert tile_cloud_fraction(Item(-3)) == 0.0 and tile_cloud_fraction(Item(140)) == 1.0


def test_the_cloud_metric_reports_the_tile_and_says_so():
    from app.metrics import compute_metrics

    from .conftest import make_layers
    layers = make_layers()
    layers.optical.tile_cloud, layers.optical.grid_cloud = 0.374, 0.0
    metric = next(m for m in compute_metrics(layers) if m.id == "m_tile_cloud")
    assert metric.value == pytest.approx(37.4)
    assert metric.label == "Cloud over the satellite tile"
    assert "whole tile" in metric.source
    assert metric.display() == "37"  # a reader sees 37%, not the window's 0%


def test_the_window_estimate_still_picks_the_scene():
    """Selection must stay on the window: a 40% cloudy tile can be clear over the site."""
    from app.geo.sources import Optical
    o = Optical("S2", "2026-01-24", tile_cloud=0.4, grid_cloud=0.0, image_cloud=0.03,
                scl=None, blue=None, green=None, red=None, nir=None)
    assert o.grid_cloud < o.tile_cloud
    # three different questions, three different answers, and the caption uses the picture
    assert o.image_cloud != o.tile_cloud


# ---------------------------------------------- cloud over the pictured window
def _tile(cloud_px=None, bright_roofs=False):
    """A 200 x 200 synthetic tile. Optionally puts cloud OUTSIDE the 500 m measurement
    ring but inside the 2 km picture, which is the case that caught us out."""
    import numpy as np
    n = 200
    scl = np.full((n, n), 5, "float32")  # not vegetated: dense urban
    blue = np.full((n, n), 2100, "float32")  # 0.21 reflectance, a normal Lagos roofscape
    red = np.full((n, n), 2600, "float32")
    nir = np.full((n, n), 3000, "float32")
    if bright_roofs:  # bright concrete: bright in the visible, not flat, not that bright
        blue[20:40, 20:40], red[20:40, 20:40], nir[20:40, 20:40] = 3400, 4600, 3200
    if cloud_px is not None:
        ys, xs = cloud_px
        blue[ys, xs], red[ys, xs], nir[ys, xs] = 4900, 5600, 6000  # bright and flat
    return scl, blue, red, nir


def test_cloud_outside_the_rings_still_counts_for_the_caption():

    from app.geo.aoi import Grid, ring_mask
    from app.geo.water import cloud_fraction

    grid = Grid(size_m=2000, res_m=10)
    corner = (slice(0, 40), slice(0, 40))  # top-left of the picture, ~1.2 km from centre
    scl, blue, red, nir = _tile(cloud_px=corner)

    whole = cloud_fraction(scl, blue, red, nir)
    within_500 = cloud_fraction(scl, blue, red, nir, ring_mask(grid, 500))
    assert whole == pytest.approx(1600 / 40000, abs=0.002)  # 4% of the picture
    assert within_500 == 0.0  # and none of it is inside the measured ring
    assert whole > within_500, "the caption must not inherit the ring's answer"


def test_bright_roofs_are_not_called_cloud():
    """Half of Lagos is brighter than a naive threshold; concrete must not read as cloud."""
    from app.geo.water import cloud_fraction
    scl, blue, red, nir = _tile(bright_roofs=True)
    assert cloud_fraction(scl, blue, red, nir) == 0.0


def test_cloud_the_scene_classifier_missed_is_still_counted():
    """SCL called two obvious cloud blobs 'not vegetated' over bright Lagos rooftops."""
    import numpy as np

    from app.geo.water import cloud_fraction
    scl, blue, red, nir = _tile(cloud_px=(slice(80, 120), slice(80, 120)))
    assert not np.isin(scl.astype(int), (3, 8, 9, 10)).any()  # SCL sees nothing
    assert cloud_fraction(scl, blue, red, nir) == pytest.approx(0.04, abs=0.002)


def test_scl_cloud_classes_are_counted_even_when_not_bright():

    from app.geo.water import cloud_fraction
    scl, blue, red, nir = _tile()
    scl[0:20, 0:20] = 3   # cloud shadow: dark, so only SCL can find it
    scl[20:30, 0:20] = 10  # cirrus: thin, so only SCL can find it
    assert cloud_fraction(scl, blue, red, nir) == pytest.approx(600 / 40000, abs=0.002)
