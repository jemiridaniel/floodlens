"""The validation set is data, so check it like data: offline, no network.

A wrong pin or a missing wet_date quietly weakens every future weight-tuning run, so
these are contract tests on data/validation_sites.csv itself.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from app.geo.sources import PlanetaryComputerSource
from app.pipeline import assess
from app.schemas import AssessmentRequest
from app.scoring import FEATURES

from .conftest import FakeSource, make_layers

SITES_CSV = Path(__file__).resolve().parents[2] / "data" / "validation_sites.csv"
LAGOS = {"lat": (6.3, 6.8), "lon": (3.0, 3.7)}  # anything outside this is a geocoding slip
COLUMNS = {"site_id", "label", "lat", "lon", "expected", "wet_date", "evidence_url", "geocode", "notes"}


@pytest.fixture(scope="module")
def sites() -> list[dict[str, str]]:
    with SITES_CSV.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("site_id")]


def test_csv_has_the_columns_validate_py_reads(sites):
    with SITES_CSV.open(newline="") as f:
        assert set(csv.DictReader(f).fieldnames) == COLUMNS
    assert sites, "the validation set is empty"


def test_site_ids_are_unique(sites):
    ids = [s["site_id"] for s in sites]
    assert len(ids) == len(set(ids))


def test_coordinates_parse_and_land_in_lagos(sites):
    for s in sites:
        lat, lon = float(s["lat"]), float(s["lon"])
        assert LAGOS["lat"][0] <= lat <= LAGOS["lat"][1], f"{s['site_id']} lat {lat}"
        assert LAGOS["lon"][0] <= lon <= LAGOS["lon"][1], f"{s['site_id']} lon {lon}"


def test_expected_is_flood_or_dry(sites):
    assert {s["expected"] for s in sites} <= {"flood", "dry"}


def test_flood_sites_carry_a_wet_date_and_evidence(sites):
    for s in (s for s in sites if s["expected"] == "flood"):
        assert s["evidence_url"].startswith("https://"), s["site_id"]
        assert dt.date.fromisoformat(s["wet_date"])  # the flood date we point radar at


def test_dry_sites_need_no_wet_date(sites):
    for s in (s for s in sites if s["expected"] == "dry"):
        assert not s["wet_date"], f"{s['site_id']}: a dry site has no flood date to look at"


def test_every_row_is_a_valid_assessment_request(sites):
    for s in sites:
        req = AssessmentRequest(lat=float(s["lat"]), lon=float(s["lon"]), label=s["label"],
                                wet_date=s["wet_date"] or None)
        assert req.wet_date == (s["wet_date"] or None)


def test_wet_date_reaches_the_source(layers):
    """pipeline.assess passes it straight through; the source decides what to do with it."""
    source = FakeSource(layers)
    assess(AssessmentRequest(lat=6.4541, lon=3.3947, wet_date="2026-06-30"), source)
    assert source.asked[-1][1] == "2026-06-30"

    assess(AssessmentRequest(lat=6.4541, lon=3.3947), FakeSource(make_layers()))  # still optional
    source = FakeSource(make_layers())
    assess(AssessmentRequest(lat=6.4541, lon=3.3947), source)
    assert source.asked[-1][1] is None


def test_a_malformed_wet_date_is_rejected():
    for bad in ("30 Jun 2026", "2026-6-30", "2026/06/30", "yesterday"):
        with pytest.raises(ValueError):
            AssessmentRequest(lat=6.4541, lon=3.3947, wet_date=bad)


def test_a_given_wet_date_replaces_the_rainfall_guess(monkeypatch):
    """With wet_date set, the radar search uses it and the wettest-rainfall days are not tried."""
    asked: list[str] = []
    monkeypatch.setattr("app.geo.sources._catalog", lambda: object())
    monkeypatch.setattr("app.geo.sources.clearest_optical", lambda *a, **k: None)
    monkeypatch.setattr("app.geo.sources.jrc_occurrence", lambda *a, **k: None)
    monkeypatch.setattr("app.geo.sources.dem", lambda *a, **k: None)
    monkeypatch.setattr("app.geo.sources.daily_rain",
                        lambda *a, **k: [("2026-07-15", 200.0), ("2026-06-30", 10.0)])
    monkeypatch.setattr("app.geo.sources.radar_near",
                        lambda catalog, site, day, **k: asked.append(day) or None)

    from app.geo.aoi import Site
    source = PlanetaryComputerSource()
    layers = source.fetch(Site(6.5026016, 3.3051669), wet_date="2026-06-30")
    assert asked == ["2026-06-30"]  # not 2026-07-15, the rainiest day
    assert any("no Sentinel-1 scene" in n for n in layers.notes)

    asked.clear()
    source.fetch(Site(6.5026016, 3.3051669))
    # no wet_date: back to the rainfall guess, rainiest first, then the next as a fallback
    assert asked == ["2026-07-15", "2026-06-30"]


def test_validate_has_a_column_head_for_every_scored_metric():
    """validate.py prints one column per FEATURE, so a new feature needs a heading."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "_validate", Path(__file__).resolve().parents[1] / "scripts" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass needs the module importable while it runs
    spec.loader.exec_module(mod)
    assert mod.SCORED == [f.metric_id for f in FEATURES]
    assert set(mod.HEADS) == set(mod.SCORED)


def test_radar_window_is_wide_enough_for_a_named_flood_date():
    """6 days missed the Lekki flood date; two revisit cycles is the point of 12."""
    from app.geo.sources import RADAR_WINDOW_DAYS
    assert RADAR_WINDOW_DAYS == 12


def test_worldcover_picks_the_newest_product_year(monkeypatch):
    """Planetary Computer has one esa-worldcover item per year per tile (2020 v100,
    2021 v200). Taking the first search hit would be arbitrary."""
    from app.geo import sources
    from app.geo.aoi import Site

    class Item:
        def __init__(self, item_id, when):
            self.id, self.bbox = item_id, (3.0, 6.0, 4.0, 7.0)
            self.properties = {"start_datetime": when}
            self.assets = {"map": type("A", (), {"href": f"href://{item_id}"})()}

    items = [Item("ESA_WorldCover_10m_2020_v100", "2020-01-01T00:00:00Z"),
             Item("ESA_WorldCover_10m_2021_v200", "2021-01-01T00:00:00Z")]
    monkeypatch.setattr(sources, "_search", lambda *a, **k: items)
    read: list[str] = []
    monkeypatch.setattr(sources, "read_window", lambda href, *a, **k: read.append(href) or "grid")

    got = sources.landcover(object(), Site(6.5, 3.3))
    assert got == ("grid", "ESA_WorldCover_10m_2021_v200")
    assert read == ["href://ESA_WorldCover_10m_2021_v200"]
