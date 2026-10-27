"""Check an LLM-written report against the measured metrics before anyone sees it.

Rules:
1. A YYYY-MM-DD date is checked as one token: it must appear in some metric's label or
   source. Dates are removed before rule 2, so a date can never smuggle its parts past
   it as measured numbers.
2. Every other number in the text must match a metric value (at its display precision,
   or within a small tolerance), the score, or a small allowlist (ring radii, years).
3. The stated tier must equal the scorer's tier.
4. The recommended action must be allowed for that tier.
5. Any required sentence must appear verbatim. This is how the "no street-level flood
   reports" caveat is made non-optional rather than left to the model's judgement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Action, Metric, Score, Tier
from ..scoring import ALLOWED_ACTIONS

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:\.\d+)?")
DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
ALWAYS_OK = {250.0, 500.0, 100.0, 1984.0}


@dataclass
class Verdict:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _numbers(text: str) -> list[float]:
    out = []
    for tok in NUMBER.findall(text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _matches(x: float, allowed: list[tuple[float, int]]) -> bool:
    for value, decimals in allowed:
        tol = max(0.5 * 10 ** (-decimals), abs(value) * 0.01)
        if abs(x - value) <= tol + 1e-9:
            return True
    return False


def verify(text: str, tier: str, action: str, metrics: list[Metric], score: Score,
           extra_allowed: set[float] | None = None, required: list[str] | None = None) -> Verdict:
    errors: list[str] = []
    flat = " ".join(text.split())
    for phrase in required or []:
        if " ".join(phrase.split()) not in flat:
            errors.append(f"the report must contain this sentence exactly: {phrase!r}")
    allowed = [(m.value, m.decimals) for m in metrics] + [(score.score, 1)]
    ok_set = ALWAYS_OK | (extra_allowed or set())

    # Dates first, as whole tokens. A metric's own date may be quoted; any other date is
    # invented. Matched dates are then cut out of the text, so the number rule never sees
    # "2025-06-14" as the measurements 6 and 14.
    measured_dates = {d for m in metrics for d in DATE.findall(f"{m.label} {m.source}")}
    for d in DATE.findall(text):
        if d not in measured_dates:
            errors.append(f"date {d} is not in any measured metric")

    for x in _numbers(DATE.sub(" ", text)):
        if x in ok_set or (1900 <= x <= 2100 and x.is_integer()):
            continue
        if not _matches(x, allowed):
            errors.append(f"number {x:g} does not match any measured value")

    try:
        stated = Tier(tier)
    except ValueError:
        errors.append(f"unknown tier {tier!r}")
        stated = None
    if stated is not None and stated != score.tier:
        errors.append(f"tier {stated.value} does not match scored tier {score.tier.value}")

    try:
        act = Action(action)
    except ValueError:
        errors.append(f"unknown action {action!r}")
        act = None
    if act is not None and act not in ALLOWED_ACTIONS[score.tier]:
        errors.append(f"action {act.value} is not allowed for tier {score.tier.value}")

    return Verdict(ok=not errors, errors=errors)
