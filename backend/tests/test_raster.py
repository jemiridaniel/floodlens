"""Remote reads must be bounded. Offline: no COG is ever opened."""
from __future__ import annotations

import pytest
import rasterio

from app.geo.aoi import Grid, Site
from app.geo.raster import GDAL_HTTP_OPTIONS, read_window


class Opened(Exception):
    """Raised instead of really opening a COG, to end the test early."""


@pytest.fixture
def env_spy(monkeypatch):
    """Records what rasterio.Env was given, and whether it was entered before any open."""
    seen: dict = {"entered": False, "options": None}

    class FakeEnv:
        def __init__(self, **options):
            seen["options"] = options

        def __enter__(self):
            seen["entered"] = True
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(href, *a, **kw):
        if not seen["entered"]:
            raise AssertionError("rasterio.Env must be entered before the COG is opened")
        raise Opened(href)

    monkeypatch.setattr(rasterio, "Env", FakeEnv)
    monkeypatch.setattr(rasterio, "open", fake_open)
    return seen


def test_a_remote_read_is_wrapped_in_a_timeout(env_spy):
    """Without this, a hung socket blocks forever: GDAL sets no request timeout itself,
    and sources.try_twice can only retry something that actually raises."""
    with pytest.raises(Opened):
        read_window("https://example.invalid/scene.tif", Site(6.5, 3.3), Grid(2000, 10))
    assert env_spy["entered"] is True
    assert env_spy["options"] == GDAL_HTTP_OPTIONS


def test_the_timeouts_are_set_and_finite():
    assert 0 < int(GDAL_HTTP_OPTIONS["GDAL_HTTP_TIMEOUT"]) <= 120
    assert 0 < int(GDAL_HTTP_OPTIONS["GDAL_HTTP_CONNECTTIMEOUT"]) <= 60
    assert int(GDAL_HTTP_OPTIONS["GDAL_HTTP_MAX_RETRY"]) >= 1
    assert all(isinstance(v, str) for v in GDAL_HTTP_OPTIONS.values())  # GDAL wants strings


def test_a_stalled_read_becomes_a_retry_then_a_named_missing_group(monkeypatch):
    """The whole point: a hang that now raises is retried, and a layer that stays gone is
    reported rather than silently rescaled."""
    from app.geo.sources import try_twice
    from app.scoring import score

    notes: list[str] = []
    attempts = []

    def timing_out():
        attempts.append(1)
        raise OSError("GDAL timeout after 30s")

    assert try_twice("landcover", timing_out, notes) is None
    assert len(attempts) == 2  # tried again before giving up
    assert "after 2 tries" in notes[0]

    # with landcover and the DEM gone, the terrain group is empty and must be named
    s = score([])
    assert "terrain and surface" in s.missing_groups and s.confidence == "low"
