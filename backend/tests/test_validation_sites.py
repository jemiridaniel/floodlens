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


def test_validate_reports_the_five_scored_metrics():
    assert [f.metric_id for f in FEATURES] == [
        "m_water_history_500", "m_wet_water_250", "m_height_above_water", "m_dist_water", "m_max_daily_rain"]
