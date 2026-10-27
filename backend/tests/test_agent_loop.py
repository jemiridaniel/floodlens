"""The agent loop, offline: a canned Meta Responses API in place of the real client.

Covers the three paths that matter: a clean pass, a verifier rejection then a good
retry, and two rejections falling back to a metrics-only report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from app.agent import loop
from app.pipeline import assess
from app.schemas import AssessmentRequest
from app.scoring import ALLOWED_ACTIONS

from .conftest import FakeSource, make_layers

# ---------------------------------------------------------------- canned client


@dataclass
class FakeCall:
    name: str
    arguments: str = "{}"
    call_id: str = "call_x"
    type: str = "function_call"


@dataclass
class FakeUsage:
    input_tokens: int = 1200
    output_tokens: int = 300
    input_tokens_details: dict = field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens_details: dict = field(default_factory=lambda: {"reasoning_tokens": 0})


@dataclass
class FakeResponse:
    output: list = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeClient:
    """Returns one scripted response per model call; records what it was sent."""

    def __init__(self, *turns: FakeResponse, repeat: bool = False):
        self.turns = list(turns)
        self.repeat = repeat
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.turns) > 1 or not self.repeat:
            if not self.turns:
                raise AssertionError("the loop made more model calls than the test scripted")
            return self.turns.pop(0)
        return self.turns[0]


def call(tool: str, usage: FakeUsage | None = None, **args) -> FakeResponse:
    """One model turn that calls one tool. (Named `tool` so a tool arg can be `name`.)"""
    return FakeResponse(output=[FakeCall(name=tool, arguments=json.dumps(args))],
                        usage=usage or FakeUsage())


# Only numbers the metrics or the score state: 300 m, 2.0 m, 74.6, 50.0%, 25.0 points.
GOOD = {
    "summary": "The site sits 300 m from regular water and just 2.0 m above it, "
               "so the risk tier is High at 74.6 out of 100.",
    "drivers": [
        {"metric_id": "m_wet_water_250", "why": "Radar saw 50.0% of the 250 m ring under water on a wet date."},
        {"metric_id": "m_height_above_water", "why": "The ground is only 2.0 m above the nearby water surface."},
    ],
    "action": "approve_with_insurance",
    "rationale": "Half the ring floods on a wet date, which alone adds 25.0 points.",
    "caveats": ["Radar can miss water trapped between buildings."],
}
BAD_NUMBER = {**GOOD, "summary": "The site is 42 m from water and floods often."}


@pytest.fixture
def assessment():
    return assess(AssessmentRequest(lat=6.4541, lon=3.3947, label="Synthetic"), FakeSource(make_layers()))


def test_clean_pass(assessment, layers):
    client = FakeClient(
        call("get_metrics"),
        call("view_evidence", name="optical"),
        call("submit_report", **GOOD),
    )
    run = loop.explain(assessment, layers, client=client)

    assert run.report.explained is True
    assert run.report.action.value == "approve_with_insurance"
    assert run.errors == []
    assert run.calls == 3
    assert [s.kind for s in run.steps] == ["model", "tool", "model", "tool", "model", "verify"]
    assert run.steps[-1].detail == "accepted"
    assert run.input_tokens == 3600 and run.output_tokens == 900  # 3 calls x the canned usage
    assert run.cost_usd == pytest.approx(loop.cost_usd(run.model, 3600, 900, 0))
    assert run.cost_usd > 0


def test_verifier_failure_then_successful_retry(assessment, layers):
    client = FakeClient(
        call("submit_report", **BAD_NUMBER),
        call("submit_report", **GOOD),
    )
    run = loop.explain(assessment, layers, client=client)

    assert run.report.explained is True
    assert run.report.summary == GOOD["summary"]
    assert run.errors == []  # the retry succeeded; the rejection is kept in the steps
    rejected = [s for s in run.steps if s.kind == "verify" and "42" in s.detail]
    assert len(rejected) == 1
    # the errors were handed back to the model as the tool result
    sent_back = json.dumps(client.requests[-1]["input"])
    assert "REJECTED by the verifier" in sent_back and "42" in sent_back


def test_two_failures_fall_back_to_metrics_only(assessment, layers):
    client = FakeClient(call("submit_report", **BAD_NUMBER), call("submit_report", **BAD_NUMBER))
    run = loop.explain(assessment, layers, client=client)

    assert run.report.explained is False
    assert run.calls == 2
    assert any("42" in e for e in run.errors)
    assert "74.6" in run.report.summary and run.report.action in ALLOWED_ACTIONS[assessment.score.tier]
    assert any("did not pass" in c or "failed the verifier" in c for c in run.report.caveats)


def test_fallback_report_passes_the_verifier(assessment):
    """The fallback is written from the metrics, so it must never trip the verifier itself."""
    from app.agent.verifier import verify
    report = loop.fallback_report(assessment, "testing")
    verdict = verify(report.text(), assessment.score.tier.value, report.action.value,
                     assessment.metrics, assessment.score, loop.allowed_numbers(assessment))
    assert verdict.ok, verdict.errors


def test_images_are_only_sent_in_user_messages(assessment, layers):
    """Meta's API rejects images outside user messages, so view_evidence returns text
    and the PNG follows as its own user message."""
    client = FakeClient(call("view_evidence", name="water_history"), call("submit_report", **GOOD))
    loop.explain(assessment, layers, client=client)

    items = client.requests[-1]["input"]
    with_image = [i for i in items if any(c.get("type") == "input_image" for c in i.get("content", []))]
    assert len(with_image) == 1
    assert with_image[0]["role"] == "user" and with_image[0]["type"] == "message"
    tool_results = [i for i in items if i.get("type") == "function_call_output"]
    assert tool_results and all("data:image" not in i["output"] for i in tool_results)


def test_a_tier_claim_that_contradicts_the_score_is_rejected(assessment, layers):
    wrong_tier = {**GOOD, "summary": "The risk tier is Severe for this site."}
    client = FakeClient(call("submit_report", **wrong_tier), call("submit_report", **wrong_tier))
    run = loop.explain(assessment, layers, client=client)
    assert run.report.explained is False
    assert any("Severe" in e and "High" in e for e in run.errors)


def test_confidence_and_rainfall_words_are_not_tier_claims(assessment):
    from app.schemas import Tier
    assert loop.stated_tier("Confidence is low and rainfall was moderate.", Tier.high) == "High"
    assert loop.stated_tier("The risk tier is Severe.", Tier.high) == "Severe"


def test_disallowed_action_is_rejected(assessment, layers):
    plain_approve = {**GOOD, "action": "approve"}  # not allowed at tier High
    client = FakeClient(call("submit_report", **plain_approve), call("submit_report", **plain_approve))
    run = loop.explain(assessment, layers, client=client)
    assert run.report.explained is False
    assert any("not allowed" in e for e in run.errors)


def test_unknown_driver_metric_is_rejected(assessment, layers):
    made_up = {**GOOD, "drivers": [{"metric_id": "m_invented", "why": "Looks bad."}]}
    client = FakeClient(call("submit_report", **made_up), call("submit_report", **GOOD))
    run = loop.explain(assessment, layers, client=client)
    assert run.report.explained is True
    assert any("m_invented" in s.detail for s in run.steps if s.kind == "verify")


def test_model_calls_are_capped(assessment, layers):
    """The model never calls a tool: the cap ends the run, not an infinite loop."""
    client = FakeClient(FakeResponse(output=[]), repeat=True)
    run = loop.explain(assessment, layers, client=client, max_calls=4)
    assert run.calls == 4
    assert run.report.explained is False
    assert len(client.requests) == 4


def test_model_outage_falls_back_without_raising(assessment, layers):
    class Broken:
        def __init__(self):
            self.responses = SimpleNamespace(create=self._boom)

        def _boom(self, **kwargs):
            raise RuntimeError("502 upstream")

    run = loop.explain(assessment, layers, client=Broken())
    assert run.report.explained is False
    assert any("502 upstream" in e for e in run.errors)


def test_tools_offer_only_available_evidence(assessment):
    """No radar, no water_wet view: the enum the model sees matches what exists."""
    no_radar = make_layers()
    no_radar.radar = None
    client = FakeClient(call("submit_report", **GOOD))
    loop.explain(assessment, no_radar, client=client)
    tools = {t["name"]: t for t in client.requests[0]["tools"]}
    assert set(tools) == {"get_metrics", "view_evidence", "submit_report"}
    assert tools["view_evidence"]["parameters"]["properties"]["name"]["enum"] == [
        "optical", "water_history", "water_now"]
    assert tools["submit_report"]["parameters"]["properties"]["action"]["enum"] == [
        a.value for a in assessment.allowed_actions]


def test_prices_come_from_the_table_for_the_model_used():
    m = 1_000_000
    assert loop.cost_usd("muse-spark-1.3", m, m) == pytest.approx(1.25 + 4.25)
    assert loop.cost_usd("muse-spark-1.3-contributor", m, m) == pytest.approx(0.10 + 0.20)
    # cached input is a subset of input_tokens, billed at the cached rate
    assert loop.cost_usd("muse-spark-1.3", m, 0, cached_tokens=m) == pytest.approx(0.15)
    assert loop.cost_usd("muse-spark-1.3", m, 0, cached_tokens=400_000) == pytest.approx(0.6 * 1.25 + 0.4 * 0.15)
    assert loop.cost_usd("muse-spark-1.3-contributor", m, 0, cached_tokens=m) == pytest.approx(0.002)
    assert loop.cost_usd("some-unlisted-model", m, m) == 0.0  # logged, never guessed


def test_reasoning_effort_defaults_to_medium_and_follows_the_env(assessment, layers, monkeypatch):
    monkeypatch.delenv("FLOODLENS_REASONING_EFFORT", raising=False)
    client = FakeClient(call("submit_report", **GOOD))
    run = loop.explain(assessment, layers, client=client)
    assert client.requests[0]["reasoning"] == {"effort": "medium"}
    assert run.reasoning_effort == "medium"

    monkeypatch.setenv("FLOODLENS_REASONING_EFFORT", "low")
    client = FakeClient(call("submit_report", **GOOD))
    run = loop.explain(assessment, layers, client=client)
    assert client.requests[0]["reasoning"] == {"effort": "low"}
    assert run.reasoning_effort == "low"


def test_cached_and_reasoning_tokens_are_recorded_and_priced(assessment, layers):
    usage = FakeUsage(input_tokens=2000, output_tokens=800,
                      input_tokens_details={"cached_tokens": 1500},
                      output_tokens_details={"reasoning_tokens": 650})
    client = FakeClient(call("submit_report", usage=usage, **GOOD))
    run = loop.explain(assessment, layers, client=client)
    assert (run.input_tokens, run.cached_tokens) == (2000, 1500)
    assert (run.output_tokens, run.reasoning_tokens) == (800, 650)
    assert run.cost_usd == pytest.approx(loop.cost_usd(run.model, 2000, 800, 1500))
    assert run.steps[0].cached_tokens == 1500 and run.steps[0].reasoning_tokens == 650


def test_a_report_may_quote_a_date_the_metrics_state(assessment, layers):
    """m_max_daily_rain's label carries its date; the verifier allows that date only."""
    dated = {**GOOD, "rationale": "The heaviest day on record here was 2025-06-10."}
    invented = {**GOOD, "rationale": "The site flooded on 2019-08-01."}
    run = loop.explain(assessment, layers, client=FakeClient(call("submit_report", **dated)))
    assert run.report.explained is True

    client = FakeClient(call("submit_report", **invented), call("submit_report", **invented))
    run = loop.explain(assessment, layers, client=client)
    assert run.report.explained is False
    assert any("2019-08-01" in e for e in run.errors)


def test_submit_report_asks_the_api_to_enforce_the_schema(assessment, layers):
    client = FakeClient(call("submit_report", **GOOD))
    loop.explain(assessment, layers, client=client)
    tools = {t["name"]: t for t in client.requests[0]["tools"]}
    assert tools["submit_report"]["strict"] is True
    # the read-only tools are not strict: they take no free-form content worth constraining
    assert "strict" not in tools["get_metrics"]


def _strict_violations(node, path="root", root=True):
    """The four rules from dev.meta.ai/docs/structured-output."""
    bad = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False:
                bad.append(f"{path}: additionalProperties is not false")
            if set(node.get("properties", {})) != set(node.get("required", [])):
                bad.append(f"{path}: required does not list every property")
        bad += [f"{path}: {k} is not supported" for k in ("allOf", "oneOf") if k in node]
        if root:
            bad += [f"root: top-level {k}" for k in ("anyOf", "oneOf", "allOf", "enum", "not") if k in node]
        for k, v in node.items():
            bad += _strict_violations(v, f"{path}.{k}", root=False)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _strict_violations(v, f"{path}[{i}]", root=False)
    return bad


def test_report_schema_stays_inside_the_strict_subset():
    """A schema outside the subset fails the whole request with an HTTP 400, which would
    only show up on a real call, so check it here."""
    from app.schemas import Action
    assert _strict_violations(loop._report_schema(list(Action))) == []


def test_a_wrong_typed_drivers_field_never_reaches_the_verifier(assessment, layers, monkeypatch):
    """Types are the API's job now. If one slips through anyway the report is malformed,
    not merely wrong, so it is rejected before the verifier ever sees it."""
    verified = []
    real = loop.verify
    monkeypatch.setattr(loop, "verify", lambda *a, **k: verified.append(a) or real(*a, **k))

    wrong_type = {**GOOD, "drivers": "m_dist_water is the main driver"}  # a string, not a list
    client = FakeClient(call("submit_report", **wrong_type), call("submit_report", **GOOD))
    run = loop.explain(assessment, layers, client=client)

    schema_steps = [s for s in run.steps if s.kind == "schema"]
    assert len(schema_steps) == 1
    assert "drivers" in schema_steps[0].detail and "valid list" in schema_steps[0].detail
    assert len(verified) == 1  # only the well-formed retry was verified
    assert verified[0][0] == GOOD["summary"] + "\n" + GOOD["rationale"] + "\n" + "\n".join(
        d["why"] for d in GOOD["drivers"]) + "\n" + "\n".join(GOOD["caveats"])
    assert run.report.explained is True  # the retry still carries the run
