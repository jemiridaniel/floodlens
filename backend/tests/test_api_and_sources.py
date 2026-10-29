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
