"""The assessment job and its live event stream. Offline: no network, no API keys."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import jobs as J
from app.main import HOOKS, _jobs, app, limiter
from app.schemas import AgentRun, FloodHistory, FloodReport, Report

from .conftest import FakeSource, make_layers


# ------------------------------------------------------------------- fakes
def fake_history(reports=None, cost=0.03):
    kept = reports if reports is not None else [
        FloodReport(place_named="Ago Palace Way", match_level="street", event_date="2026-06-30",
                    publisher="Vanguard", url="https://example.test/a", quote="knee deep",
                    text_verified="true")]

    def lookup(site, address=None, **kw):
        lookup.address = address
        return FloodHistory(place="Ago Palace Way, Lagos", reports=kept, searched=True,
                            cost_usd=cost, searches=4, text_checked=True)

    lookup.address = None
    return lookup


def fake_explain(assessment, layers=None, **kw):
    return AgentRun(
        report=Report(summary="A summary.", action=assessment.allowed_actions[0],
                      rationale="Because.", caveats=[], explained=True),
        steps=[], calls=2, input_tokens=100, output_tokens=50, cost_usd=0.002)


async def fake_segment(image, prompt):
    mask = np.zeros(np.asarray(image).shape[:2], dtype=bool)
    if prompt in ("standing water", "door"):
        mask[:4, :4] = True
    return type("Seg", (), {"mask": mask, "instances": int(mask.any()),
                            "status": "completed", "coverage": float(mask.mean())})()


def photo_bytes(colour=(120, 140, 160), size=(32, 24), exif=True) -> bytes:
    img = Image.new("RGB", size, colour)
    buf = io.BytesIO()
    if exif:
        ex = Image.Exif()
        ex[271] = "FloodPhone"  # Make
        ex[306] = "2026:06:30 09:15:00"  # DateTime
        ex[34853] = {1: "N"}  # GPSInfo: the reason stripping matters
        img.save(buf, format="JPEG", exif=ex)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOODLENS_DEMO_CACHE", raising=False)
    limiter._hits.clear()  # the limiter is module state: one test must not spend another's quota
    HOOKS.clear()
    HOOKS.update(source=FakeSource(make_layers()), history_lookup=fake_history(),
                 explain=fake_explain, segment=fake_segment, uploads=tmp_path / "uploads")
    _jobs.clear()
    with TestClient(app) as client:
        yield client
    HOOKS.clear()


def start(api, **fields):
    data = {"lat": 6.5026, "lon": 3.3052, **fields}
    r = api.post("/assessments", data=data)
    assert r.status_code == 202, r.text
    return r.json()["id"]


def drain(api, assessment_id):
    """Read the whole event stream, returning the parsed events."""
    events = []
    with api.stream("GET", f"/assessments/{assessment_id}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
            if events and events[-1].get("type") == "done":
                break
    return events


# ------------------------------------------------------------------- the flow
def test_the_job_runs_every_stage_in_order(api):
    events = drain(api, start(api, label="Test shop"))
    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    done = {e["step"]: e for e in events if e["type"] == "step" and e["status"] != "running"}
    assert set(done) == set(J.STEP_ORDER)

    # the area screen and the flood history run together, so either may land first, but
    # both must be finished before the score, and the rest are strictly sequential
    finished = [e["step"] for e in events if e["type"] == "step" and e["status"] != "running"]
    assert set(finished[:2]) == {J.AREA, J.HISTORY}
    assert finished[2:] == [J.PHOTOS, J.SCORE, J.EXPLAIN, J.VERIFY]
    starts = [e["step"] for e in events if e["type"] == "step" and e["status"] == "running"]
    assert starts[:2] == [J.AREA, J.HISTORY]  # both started before either finished
    assert done["Area screen"]["status"] == "done"
    assert done["Flood history"]["status"] == "done"
    assert done["Site photos"]["status"] == "skipped"  # none uploaded
    assert done["Score"]["status"] == "done" and "/100" in done["Score"]["detail"]
    assert done["Verify"]["status"] == "done" and done["Verify"]["detail"] == "accepted"


def test_every_step_carries_duration_and_cumulative_cost(api):
    events = drain(api, start(api))
    costs = [e["cost_usd"] for e in events if e["type"] == "step"]
    assert costs == sorted(costs), "cumulative cost must never go down"
    assert costs[-1] == pytest.approx(0.032)  # 0.03 search + 0.002 agent
    assert all("ms" in e for e in events if e["type"] == "step")
    assert any(e["ms"] >= 0 for e in events if e["type"] == "step" and e["status"] == "done")


def test_the_stream_replays_for_a_late_subscriber(api):
    """A phone that reconnects mid-assessment must not miss the steps it slept through."""
    assessment_id = start(api)
    first = drain(api, assessment_id)
    again = drain(api, assessment_id)
    assert [e["seq"] for e in again] == [e["seq"] for e in first]
    assert again[-1]["type"] == "done"


def test_the_full_result_groups_metrics_by_evidence_type(api):
    assessment_id = start(api, label="Test shop", address="12 Ago Palace Way")
    drain(api, assessment_id)
    body = api.get(f"/assessments/{assessment_id}").json()

    assert body["status"] == "done"
    assert {g["name"] for g in body["groups"]} == {"local history", "coastal and riverine",
                                                   "terrain and surface", "rainfall"}
    assert all(g["present"] for g in body["groups"])
    assert body["assessment"]["score"]["tier"] in {"Low", "Moderate", "High", "Severe"}
    assert body["reports"][0]["match_level"] == "street"
    assert body["report"]["summary"] == "A summary."
    assert body["cost_usd"] == pytest.approx(0.032)
    assert [s["step"] for s in body["steps"]] == list(J.STEP_ORDER)


def test_the_typed_address_reaches_the_search(api):
    """It widens the search; Place.typed keeps it out of the match-level test."""
    drain(api, start(api, address="12 Ago Palace Way, Okota"))
    assert HOOKS["history_lookup"].address == "12 Ago Palace Way, Okota"


def test_evidence_images_are_served(api):
    assessment_id = start(api)
    drain(api, assessment_id)
    urls = api.get(f"/assessments/{assessment_id}").json()["evidence_urls"]
    assert {"optical", "water_now", "water_history"} <= set(urls)
    r = api.get(f"/assessments/{assessment_id}/evidence/optical.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert api.get(f"/assessments/{assessment_id}/evidence/nope.png").status_code == 404


# ------------------------------------------------------------------- photos
def test_photos_are_stripped_of_exif_and_segmented(api, tmp_path):
    r = api.post("/assessments", data={"lat": 6.5026, "lon": 3.3052},
                 files=[("photos", ("site.jpg", photo_bytes(), "image/jpeg")),
                        ("photos", ("door.jpg", photo_bytes((90, 90, 90)), "image/jpeg"))])
    assert r.status_code == 202
    assessment_id = r.json()["id"]
    events = drain(api, assessment_id)
    photos_step = next(e for e in events if e.get("step") == "Site photos" and e["status"] == "done")
    assert "no effect on the score" in photos_step["detail"]

    body = api.get(f"/assessments/{assessment_id}").json()
    assert len(body["photos"]) == 2
    prompts = [f["prompt"] for f in body["photos"][0]["findings"]]
    assert prompts == ["standing water", "puddle", "door", "road", "drain"]
    hits = [f for f in body["photos"][0]["findings"] if f["instances"]]
    assert {f["prompt"] for f in hits} == {"standing water", "door"}
    assert all(f["overlay_url"] for f in hits)

    # the stored file carries no EXIF at all
    stored = list((tmp_path / "uploads").glob("*.png"))
    assert len(stored) == 2
    with Image.open(stored[0]) as im:
        assert not dict(im.getexif())
    # and no photo finding is in the score
    assert "m_photo" not in json.dumps(body["assessment"]["metrics"])


def test_exif_is_stripped_and_orientation_applied():
    from app.photos import strip_exif
    img = Image.new("RGB", (40, 20), (10, 20, 30))
    ex = Image.Exif()
    ex[274] = 6  # rotate 90 CW
    ex[34853] = {1: "N"}  # GPS
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=ex)

    clean = strip_exif(buf.getvalue())
    assert not dict(clean.getexif())
    assert clean.size == (20, 40)  # orientation honoured before the tag was dropped
    assert "exif" not in clean.info


def test_too_many_photos_is_rejected(api):
    files = [("photos", (f"p{i}.jpg", photo_bytes(), "image/jpeg")) for i in range(6)]
    r = api.post("/assessments", data={"lat": 6.5026, "lon": 3.3052}, files=files)
    assert r.status_code == 413


# ------------------------------------------------------------------- failures
def test_a_failed_search_does_not_sink_the_assessment(api):
    def boom(site, address=None, **kw):
        raise RuntimeError("search is down")

    HOOKS["history_lookup"] = boom
    assessment_id = start(api)
    events = drain(api, assessment_id)
    assert events[-1]["status"] == "done"  # the assessment still finishes
    history = next(e for e in events if e.get("step") == "Flood history" and e["status"] != "running")
    assert history["status"] == "failed" and "search is down" in history["detail"]

    body = api.get(f"/assessments/{assessment_id}").json()
    missing = body["assessment"]["score"]["missing_groups"]
    assert "local history" in missing  # not counted as "no floods here"
    assert body["assessment"]["score"]["confidence"] == "low"
    assert not next(g for g in body["groups"] if g["name"] == "local history")["present"]


def test_a_failed_area_screen_fails_the_assessment(api):
    class Broken:
        def fetch(self, site, wet_date=None):
            raise RuntimeError("planetary computer is down")

    HOOKS["source"] = Broken()
    events = drain(api, start(api))
    assert events[-1]["type"] == "done" and events[-1]["status"] == "failed"
    assert "area screen" in events[-1]["error"]


def test_bad_coordinates_and_missing_id(api):
    assert api.post("/assessments", data={"lat": 120, "lon": 3}).status_code == 422
    assert api.get("/assessments/nope").status_code == 404
    assert api.get("/assessments/nope/evidence/optical.png").status_code == 404


# ------------------------------------------------------------------- demo cache
def test_a_cached_site_replays_instantly(api, tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODLENS_DEMO_CACHE", "1")
    monkeypatch.setattr(J, "DEMO_CACHE", tmp_path / "demo")

    first = start(api)
    drain(api, first)
    saved = api.get(f"/assessments/{first}").json()
    assert (tmp_path / "demo").exists(), "a finished run should seed the demo cache"

    # a second request for the same point must not run the pipeline at all
    HOOKS["source"] = None
    HOOKS["history_lookup"] = lambda *a, **k: pytest.fail("the cache should have answered")
    second = start(api)
    events = drain(api, second)
    assert events[-1]["status"] == "done"
    body = api.get(f"/assessments/{second}").json()
    assert body["from_cache"] is True
    assert body["id"] == second and body["id"] != first
    assert body["assessment"]["score"] == saved["assessment"]["score"]
    assert [s["step"] for s in body["steps"]] == list(J.STEP_ORDER)
    assert api.get(f"/assessments/{second}/evidence/optical.png").status_code == 200


def test_the_demo_cache_is_off_by_default(api, tmp_path, monkeypatch):
    monkeypatch.setattr(J, "DEMO_CACHE", tmp_path / "demo")
    drain(api, start(api))
    assert not (tmp_path / "demo").exists()
    assert api.get("/health").json()["demo_cache"] == "off"


def test_a_site_with_no_street_reports_says_so(api):
    """The Lekki Phase 1 case: the search ran, found nothing at street level, and the
    report must say so in those words. The UI keys its banner off has_street_reports."""
    from app.agent.loop import NO_STREET_REPORTS
    HOOKS["history_lookup"] = fake_history(reports=[])
    assessment_id = start(api, label="Lekki Phase 1 shop")
    drain(api, assessment_id)
    body = api.get(f"/assessments/{assessment_id}").json()

    a = body["assessment"]
    assert a["has_street_reports"] is False
    assert a["street_flood_events"] == 0
    assert body["reports"] == []
    # the search ran and found nothing, which is a measurement: the group is still present
    assert "local history" not in a["score"]["missing_groups"]
    assert a["score"]["tier_floor"] is None
    assert NO_STREET_REPORTS == ("No street-level flood reports found. "
                                 "This does not mean the site is safe.")


def test_an_area_only_report_is_not_a_street_report(api):
    """Area matches are listed as context but must not switch off the sentence."""
    from app.schemas import FloodReport
    HOOKS["history_lookup"] = fake_history(reports=[
        FloodReport(place_named="Oshodi/Isolo", match_level="area", event_date="2026-06-30",
                    publisher="Punch", url="https://example.test/b", quote="district flooded",
                    text_verified="true")])
    assessment_id = start(api)
    drain(api, assessment_id)
    body = api.get(f"/assessments/{assessment_id}").json()
    assert body["assessment"]["has_street_reports"] is False  # the sentence still shows
    assert len(body["reports"]) == 1  # and the report is still listed as context


def test_two_street_events_raise_the_tier_through_the_api(api):
    """The evidence rule, end to end."""
    from app.schemas import FloodReport
    HOOKS["history_lookup"] = fake_history(reports=[
        FloodReport(place_named="Ago Palace Way", match_level="street", event_date=d,
                    publisher="P", url=f"https://example.test/{d}", quote="flooded",
                    text_verified="true")
        for d in ("2026-06-30", "2025-09-16")])
    assessment_id = start(api)
    drain(api, assessment_id)
    a = api.get(f"/assessments/{assessment_id}").json()["assessment"]
    assert a["street_flood_events"] == 2
    # the invariant: two street-level events can never leave the tier below High. On this
    # fixture the weighted score reaches High on its own, so tier_floor stays None -- it
    # is set only when the rule actually had to raise the tier.
    assert a["score"]["tier"] in ("High", "Severe")

    from app.schemas import Tier
    from app.scoring import apply_tier_floor
    raised, why = apply_tier_floor(Tier.low, a["street_flood_events"])
    assert raised == Tier.high and "street-level" in why


def test_a_demo_copy_from_older_scoring_rules_is_ignored(api, tmp_path, monkeypatch):
    """A cached demo must never replay a verdict the current code would not produce."""
    monkeypatch.setenv("FLOODLENS_DEMO_CACHE", "1")
    monkeypatch.setattr(J, "DEMO_CACHE", tmp_path / "demo")

    drain(api, start(api))
    folder = next((tmp_path / "demo").iterdir())
    assert (folder / "scoring.txt").read_text() == J.scoring_fingerprint()
    assert J.load_demo(6.5026, 3.3052, tmp_path / "demo") is not None

    (folder / "scoring.txt").write_text("weights-from-last-week")
    assert J.load_demo(6.5026, 3.3052, tmp_path / "demo") is None  # stale: re-run instead

    folder_without_stamp = tmp_path / "demo" / "9.9999_9.9999"
    folder_without_stamp.mkdir()
    (folder_without_stamp / "view.json").write_text("{}")
    assert J.load_demo(9.9999, 9.9999, tmp_path / "demo") is None


def test_the_fingerprint_moves_with_the_scoring_rules(monkeypatch):
    from app import scoring
    before = J.scoring_fingerprint()
    monkeypatch.setattr(scoring, "TIER_FLOOR_STREET_EVENTS", 3)
    assert J.scoring_fingerprint() != before


def test_a_stage_that_hangs_costs_one_stage_not_the_assessment(api, monkeypatch):
    """A slow search must never hold an officer at 'Working on it' indefinitely."""
    import asyncio

    monkeypatch.setitem(J.STAGE_LIMIT_S, J.HISTORY, 1)

    def never_returns(site, address=None, **kw):
        import time
        time.sleep(30)

    HOOKS["history_lookup"] = never_returns
    assessment_id = start(api)
    events = drain(api, assessment_id)

    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    history = next(e for e in events if e.get("step") == J.HISTORY and e["status"] != "running")
    assert history["status"] == "failed" and "gave up after 1s" in history["detail"]

    body = api.get(f"/assessments/{assessment_id}").json()
    assert body["assessment"] is not None  # the area screen still produced a score
    assert "local history" in body["assessment"]["score"]["missing_groups"]
    assert body["assessment"]["score"]["confidence"] == "low"
    assert asyncio  # the stage ceiling is asyncio.wait_for, not a thread join


def test_a_failed_explain_still_leaves_a_usable_report(api):
    """The score is valid whatever happened to the agent. An officer must never be shown
    a tier with no summary, no action and no badge."""
    def boom(assessment, layers=None, **kw):
        raise RuntimeError("the model is down")

    HOOKS["explain"] = boom
    assessment_id = start(api)
    events = drain(api, assessment_id)
    assert events[-1]["status"] == "done"

    body = api.get(f"/assessments/{assessment_id}").json()
    report = body["report"]
    assert report is not None
    assert report["explained"] is False  # and the badge says so
    assert report["action"] in body["assessment"]["allowed_actions"]
    assert report["summary"]
    assert any("the model is down" in c for c in report["caveats"])
    verify_step = next(s for s in body["steps"] if s["step"] == "Verify")
    assert verify_step["status"] in ("skipped", "failed")


def test_an_explain_that_overruns_still_leaves_a_report(api, monkeypatch):
    monkeypatch.setitem(J.STAGE_LIMIT_S, J.EXPLAIN, 1)

    def slow(assessment, layers=None, **kw):
        import time
        time.sleep(20)

    HOOKS["explain"] = slow
    assessment_id = start(api)
    events = drain(api, assessment_id)
    assert events[-1]["status"] == "done"

    body = api.get(f"/assessments/{assessment_id}").json()
    assert body["report"] is not None and body["report"]["explained"] is False
    explain_step = next(s for s in body["steps"] if s["step"] == "Explain")
    assert explain_step["status"] == "failed" and "gave up after 1s" in explain_step["detail"]
    assert "gave up after 1s" in " ".join(body["report"]["caveats"])


# ------------------------------------------------------------------ rate limiting
def test_new_assessments_are_capped_per_address(api, monkeypatch):
    """A public demo spends real money per uncached run."""
    monkeypatch.setattr(limiter, "limit", 2)

    for _ in range(2):
        assert api.post("/assessments", data={"lat": 6.5026, "lon": 3.3052}).status_code == 202
    refused = api.post("/assessments", data={"lat": 6.5026, "lon": 3.3052})
    assert refused.status_code == 429
    assert "Retry-After" in refused.headers
    detail = refused.json()["detail"]
    assert "2 new assessments in an hour" in detail
    assert "saved example sites" in detail  # tells the visitor what still works


def test_cached_sites_are_never_rate_limited(api, tmp_path, monkeypatch):
    """The demo must keep working once a visitor has used their quota."""
    monkeypatch.setenv("FLOODLENS_DEMO_CACHE", "1")
    monkeypatch.setattr(J, "DEMO_CACHE", tmp_path / "demo")

    drain(api, start(api))  # one real run, which seeds the cache
    monkeypatch.setattr(limiter, "limit", 0)  # quota now entirely used up
    assert api.post("/assessments", data={"lat": 6.9, "lon": 3.9}).status_code == 429
    assert api.post("/assessments", data={"lat": 6.5026, "lon": 3.3052}).status_code == 202


def test_the_limit_is_per_address_and_rolls_off():
    from app.limits import RateLimit
    limit = RateLimit(limit=2, window_s=100)
    for _ in range(2):
        limit.record("1.1.1.1", now=1000)
    assert limit.check("1.1.1.1", now=1000)[0] is False
    assert limit.check("2.2.2.2", now=1000)[0] is True  # a different caller is unaffected
    assert limit.check("1.1.1.1", now=1101)[0] is True  # the window has rolled off
    assert limit.remaining("1.1.1.1", now=1101) == 2


def test_the_caller_is_read_from_the_proxy_header():
    from app.limits import client_ip

    class Req:
        def __init__(self, headers, host=None):
            self.headers = headers
            self.client = type("C", (), {"host": host})() if host else None

    assert client_ip(Req({"x-forwarded-for": "9.9.9.9, 10.0.0.1"})) == "9.9.9.9"
    assert client_ip(Req({}, host="5.5.5.5")) == "5.5.5.5"
    assert client_ip(Req({})) == "unknown"


def test_the_remaining_quota_is_reported(api, monkeypatch):
    monkeypatch.setattr(limiter, "limit", 3)
    assert api.get("/limits").json() == {"new_assessments_per_hour": 3, "remaining": 3}
    api.post("/assessments", data={"lat": 6.9, "lon": 3.9})
    assert api.get("/limits").json()["remaining"] == 2


def test_the_fingerprint_moves_when_the_measuring_code_changes(tmp_path, monkeypatch):
    """Scoring is not the only thing that decides a result. The cloud figure was fixed in
    metrics.py and the cached demos kept serving the old one, because the weights had not
    moved."""
    import app.jobs as jobs_module
    before = J.scoring_fingerprint()
    real = (Path(jobs_module.__file__).resolve().parent / "metrics.py").read_bytes()

    fake = tmp_path / "app"
    for name in J.MEASURING_MODULES:
        (fake / name).parent.mkdir(parents=True, exist_ok=True)
        (fake / name).write_bytes(real if name != "metrics.py" else real + b"\n# changed\n")
    monkeypatch.setattr(jobs_module, "__file__", str(fake / "jobs.py"))
    assert J.scoring_fingerprint() != before
