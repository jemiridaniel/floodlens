"""Muse Spark writes the assessment; the code keeps the numbers.

The loop is deliberately small. The model may read the measured metrics, look at the
evidence images, and submit one report. It cannot compute, choose or change a number,
a tier or an action: `pipeline.assess()` has already done that in code. Every submitted
report goes through `verifier.verify()` before it leaves this module. One rejection buys
a retry carrying the errors; a second falls back to a report written from the metrics
alone and flagged `explained=False`.

Meta Model API notes (dev.meta.ai/docs):
- Responses API, so tools are declared flat (`{"type": "function", "name": ...}`) and the
  turn history is a list of input items we keep ourselves (no `previous_response_id`).
- `tool_choice` only supports "auto", so `submit_report` cannot be forced. The loop
  re-asks when the model answers in prose, and MAX_MODEL_CALLS is what ends it.
- Images are allowed in user messages only, so `view_evidence` returns a text
  acknowledgement and the PNG follows as a separate user message.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from ..config import META_BASE_URL, get_settings
from ..evidence import VIEW_HELP, data_url, evidence_pngs
from ..geo.sources import SiteLayers
from ..schemas import Action, AgentRun, AgentStep, Assessment, Driver, Report, Tier
from .verifier import numbers, verify

log = logging.getLogger(__name__)

MAX_MODEL_CALLS = 8
MAX_VERIFY_ATTEMPTS = 2  # one submission, then one retry carrying the verifier's errors
DETAIL_CHARS = 160  # steps are for display: keep each result short

# Estimated USD per 1M tokens for muse-spark-1.3. Update from the Meta billing page;
# the per-assessment cost is logged so the $150 credit can be tracked.
USD_PER_M_INPUT = 0.35
USD_PER_M_OUTPUT = 1.40

INSTRUCTIONS = """You are FloodLens's analyst. You explain a flood-risk assessment of a \
property used as loan collateral, for a Nigerian microfinance loan officer.

The measurements, the score, the tier and the allowed actions are already computed in code.
You do not and cannot change them. Your job is to explain what they mean.

Rules, enforced by a verifier that will reject your report:
1. Call get_metrics first. Every number you write must be copied from it, exactly as given
   (same digits). Never estimate, convert, round, add or average numbers.
2. Do not state a tier other than the one you are given.
3. Choose an action only from the list you are given.
4. If you want to see the site, call view_evidence. Images are context, not measurements:
   never read a number off an image.
5. Finish by calling submit_report exactly once. Be concrete and brief; write for a loan
   officer, not a scientist. Say plainly what is uncertain.
"""


def _report_schema(allowed: list[Action]) -> dict[str, Any]:
    """Structured output for the report. The action enum is this site's allowed actions,
    so a disallowed action cannot be expressed (the verifier still checks)."""
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Two or three sentences for the loan officer."},
            "drivers": {
                "type": "array",
                "description": "The measurements that drive this score, strongest first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_id": {"type": "string", "description": "An id from get_metrics, e.g. m_dist_water."},
                        "why": {"type": "string", "description": "What this measurement means here."},
                    },
                    "required": ["metric_id", "why"],
                    "additionalProperties": False,
                },
            },
            "action": {"type": "string", "enum": [a.value for a in allowed]},
            "rationale": {"type": "string", "description": "Why that action follows from the measurements."},
            "caveats": {
                "type": "array",
                "description": "What is missing or uncertain, in plain words.",
                "items": {"type": "string"},
            },
        },
        "required": ["summary", "drivers", "action", "rationale", "caveats"],
        "additionalProperties": False,
    }


def _tools(views: list[str], allowed: list[Action]) -> list[dict[str, Any]]:
    """The agent's tools. A new one (segment_photo on a site photo) goes here and in
    _run_tool; nothing else in the loop needs to change."""
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": "get_metrics",
            "description": "The measured metrics for this site: value, unit, source, and the points each "
                           "contributes to the score. Also the score, tier, confidence and allowed actions.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    ]
    if views:
        tools.append({
            "type": "function",
            "name": "view_evidence",
            "description": "Look at one evidence image for this site. "
                           + " ".join(f"'{v}': {VIEW_HELP[v]}" for v in views if v in VIEW_HELP),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": views}},
                "required": ["name"],
                "additionalProperties": False,
            },
        })
    tools.append({
        "type": "function",
        "name": "submit_report",
        "description": "Submit the finished report. Call this exactly once, last.",
        "parameters": _report_schema(allowed),
    })
    return tools


def metrics_payload(assessment: Assessment) -> dict[str, Any]:
    """What get_metrics returns: the measured facts, and nothing else."""
    s = assessment.score
    return {
        "site": {"lat": assessment.lat, "lon": assessment.lon, "label": assessment.label},
        "metrics": [
            {
                "id": m.id, "label": m.label, "value": m.display(), "unit": m.unit, "source": m.source,
                "contribution_points": s.contributions.get(m.id, 0.0),
                "in_score": m.id in s.contributions,
            }
            for m in assessment.metrics
        ],
        "score": s.score,
        "tier": s.tier.value,
        "confidence": s.confidence,
        "missing_measurements": s.missing,
        "allowed_actions": [a.value for a in assessment.allowed_actions],
        "notes": assessment.notes,
    }


def allowed_numbers(assessment: Assessment) -> set[float]:
    """Code-computed numbers the report may quote besides the metric values themselves.

    The score and each metric's contribution, plus every number the metric labels and
    sources already state. Those last ones matter because the verifier reads a date like
    2025-06-10 as the numbers 6 and 10: quoting a date the metrics gave us is not
    inventing a number, while anything the code never produced is still rejected.
    """
    stated = {float(v) for v in assessment.score.contributions.values()}
    stated.add(float(assessment.score.score))
    for m in assessment.metrics:
        stated.update(numbers(f"{m.label} {m.source}"))
    return stated


TIER_CLAIM = "(?:{t})\\b(?:\\s+flood)?\\s+(?:risk|tier)|(?:risk|tier)\\s*(?:is|:)?\\s*(?:{t})\\b"


def stated_tier(text: str, scored: Tier) -> str:
    """The tier the report claims, so the verifier can catch prose contradicting the score.

    The report has no tier field (the agent cannot set the tier), but it can still describe
    the wrong one. Only tier words next to "risk" or "tier" count, so "low confidence" and
    "moderate rainfall" are not read as tier claims.
    """
    named = {t for t in Tier if re.search(TIER_CLAIM.format(t=t.value), text, re.I)}
    return next(iter(named)).value if len(named) == 1 else scored.value


def fallback_report(assessment: Assessment, why: str) -> Report:
    """A report built from the metrics alone, for when the model's never passes the verifier."""
    s = assessment.score
    by_id = {m.id: m for m in assessment.metrics}
    top = sorted(s.contributions.items(), key=lambda kv: -kv[1])[:3]
    drivers = [
        Driver(metric_id=mid, why=f"{by_id[mid].label}: {by_id[mid].display()} {by_id[mid].unit}".strip())
        for mid, _ in top if mid in by_id
    ]
    caveats = [f"No written explanation: {why}", "The measurements and the score below are unaffected."]
    if s.missing:
        caveats.append("Not measured for this site: " + ", ".join(s.missing) + ".")
    return Report(
        summary=f"Flood risk tier {s.tier.value}, scored {s.score} of a possible 100, "
                f"at {s.confidence} confidence.",
        drivers=drivers,
        action=assessment.allowed_actions[0],
        rationale="The score is a fixed weighted formula over the measurements listed above, "
                  "and the action is the most cautious one this tier allows.",
        caveats=caveats,
        explained=False,
    )


def _short(text: str) -> str:
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= DETAIL_CHARS else one_line[:DETAIL_CHARS - 1] + "…"


def _call_item(call: Any) -> dict[str, Any]:
    """Echo a function call back as a plain input item (works for real and mocked clients)."""
    return {"type": "function_call", "name": call.name, "arguments": call.arguments, "call_id": call.call_id}


def _output_item(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _user_text(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _user_image(text: str, png: bytes) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": text},
        {"type": "input_image", "image_url": data_url(png)},
    ]}


def _parse_report(arguments: str, assessment: Assessment) -> tuple[Report | None, list[str]]:
    """Validate submit_report's arguments. Errors here go back to the model like the verifier's."""
    try:
        data = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return None, [f"submit_report arguments are not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, ["submit_report arguments must be a JSON object"]
    data.pop("explained", None)  # the code owns this flag, not the model
    try:
        report = Report(**data)
    except ValidationError as exc:
        return None, [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    known = {m.id for m in assessment.metrics}
    unknown = [d.metric_id for d in report.drivers if d.metric_id not in known]
    return report, [f"driver metric_id {u!r} is not one of the measured metrics" for u in sorted(set(unknown))]


def _run_tool(name: str, args: dict[str, Any], assessment: Assessment,
              pngs: dict[str, bytes]) -> tuple[str, str, bytes | None]:
    """Run one non-submitting tool. Returns (output for the model, step detail, image or None)."""
    if name == "get_metrics":
        payload = metrics_payload(assessment)
        return json.dumps(payload), f"{len(payload['metrics'])} metrics, score {payload['score']}", None
    if name == "view_evidence":
        view = args.get("name")
        if view not in pngs:
            return (f"No evidence image named {view!r}. Available: {', '.join(sorted(pngs)) or 'none'}.",
                    f"unknown view {view!r}", None)
        return f"The {view} image follows in the next message.", f"{view} ({len(pngs[view])} B PNG)", pngs[view]
    return f"No tool named {name!r}.", f"unknown tool {name!r}", None


def explain(assessment: Assessment, layers: SiteLayers | None = None, client: Any | None = None,
            max_calls: int = MAX_MODEL_CALLS) -> AgentRun:
    """Have Muse Spark explain an already-scored assessment. Never raises on model failure:
    a run that cannot produce a verified report comes back with explained=False."""
    pngs = evidence_pngs(layers) if layers is not None else {}
    own = client is None
    if own:
        settings = get_settings()
        if not settings.model_api_key:
            raise RuntimeError("MODEL_API_KEY is not set")
        client = OpenAI(base_url=META_BASE_URL, api_key=settings.model_api_key)
    try:
        run = _run(client, assessment, pngs, max_calls)
    finally:
        if own:
            client.close()
    log.info("assessment %s: explained=%s in %d model calls, %d in + %d out tokens, est $%.4f",
             assessment.id, run.report.explained, run.calls, run.input_tokens, run.output_tokens, run.cost_usd)
    return run


def _run(client: Any, assessment: Assessment, pngs: dict[str, bytes], max_calls: int) -> AgentRun:
    settings = get_settings()
    tools = _tools(sorted(pngs), assessment.allowed_actions)
    extra_allowed = allowed_numbers(assessment)
    steps: list[AgentStep] = []
    items: list[Any] = [_user_text(
        f"Assess the site in get_metrics ({assessment.label or 'unlabelled'}). "
        "Read the metrics, look at any evidence you need, then submit the report."
    )]
    run = AgentRun(report=fallback_report(assessment, "the model made no usable call"))
    rejections = 0

    def add_step(kind: str, name: str, detail: str, ms: int, usage: Any = None) -> None:
        steps.append(AgentStep(
            n=len(steps) + 1, kind=kind, name=name, detail=_short(detail), ms=ms,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        ))

    def finish(report: Report, errors: list[str]) -> AgentRun:
        run.report = report
        run.steps = steps
        run.errors = errors
        run.input_tokens = sum(s.input_tokens for s in steps)
        run.output_tokens = sum(s.output_tokens for s in steps)
        run.cost_usd = cost_usd(run.input_tokens, run.output_tokens)
        return run

    errors: list[str] = []
    while run.calls < max_calls:
        t0 = time.monotonic()
        try:
            resp = client.responses.create(
                model=settings.spark_model, instructions=INSTRUCTIONS, input=items,
                tools=tools, parallel_tool_calls=False,  # one call per turn keeps progress legible
            )
        except Exception as exc:  # a model outage must not sink the assessment
            add_step("model", settings.spark_model, f"{exc.__class__.__name__}: {exc}",
                     int(1000 * (time.monotonic() - t0)))
            return finish(fallback_report(assessment, f"the model call failed ({exc.__class__.__name__})"),
                          [f"{exc.__class__.__name__}: {exc}"])
        run.calls += 1
        calls = [i for i in (getattr(resp, "output", None) or []) if getattr(i, "type", "") == "function_call"]
        add_step("model", settings.spark_model,
                 f"{len(calls)} tool call(s): {', '.join(c.name for c in calls) or 'none'}",
                 int(1000 * (time.monotonic() - t0)), getattr(resp, "usage", None))

        if not calls:
            # tool_choice is "auto" only, so submit_report cannot be forced: ask again.
            items.append(_user_text("Use the tools. Call get_metrics if you have not, then submit_report."))
            continue

        for call in calls:
            items.append(_call_item(call))
            if call.name != "submit_report":
                t1 = time.monotonic()
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                output, detail, png = _run_tool(call.name, args, assessment, pngs)
                items.append(_output_item(call.call_id, output))
                if png is not None:
                    items.append(_user_image(f"Evidence image: {args.get('name')}.", png))
                add_step("tool", call.name, detail, int(1000 * (time.monotonic() - t1)))
                continue

            t1 = time.monotonic()
            report, errors = _parse_report(call.arguments, assessment)
            if report is not None and not errors:
                verdict = verify(report.text(), stated_tier(report.text(), assessment.score.tier),
                                 report.action.value, assessment.metrics, assessment.score, extra_allowed)
                errors = verdict.errors
            ok = report is not None and not errors
            add_step("verify", "verifier", "accepted" if ok else "; ".join(errors),
                     int(1000 * (time.monotonic() - t1)))
            if ok:
                return finish(report, [])
            rejections += 1
            if rejections >= MAX_VERIFY_ATTEMPTS:
                return finish(fallback_report(assessment, "the model's report failed the verifier twice"), errors)
            items.append(_output_item(call.call_id, "REJECTED by the verifier. Fix exactly these problems and "
                                                   "call submit_report again:\n- " + "\n- ".join(errors)))

    return finish(fallback_report(assessment, f"the model did not finish within {max_calls} calls"), errors)


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens / 1e6 * USD_PER_M_INPUT + output_tokens / 1e6 * USD_PER_M_OUTPUT, 4)
