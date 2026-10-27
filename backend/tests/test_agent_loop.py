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


def call(tool: str, **args) -> FakeResponse:
    """One model turn that calls one tool. (Named `tool` so a tool arg can be `name`.)"""
    return FakeResponse(output=[FakeCall(name=tool, arguments=json.dumps(args))])


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
    assert run.cost_usd == pytest.approx(loop.cost_usd(3600, 900))
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
