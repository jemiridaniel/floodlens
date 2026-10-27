"""Check an LLM-written report against the measured metrics before anyone sees it.

Rules:
1. Every number in the text must match a metric value (at its display precision,
   or within a small tolerance), the score, or a small allowlist (ring radii, years).
2. The stated tier must equal the scorer's tier.
3. The recommended action must be allowed for that tier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Action, Metric, Score, Tier
from ..scoring import ALLOWED_ACTIONS

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:\.\d+)?")
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
           extra_allowed: set[float] | None = None) -> Verdict:
    errors: list[str] = []
    allowed = [(m.value, m.decimals) for m in metrics] + [(score.score, 1)]
    ok_set = ALWAYS_OK | (extra_allowed or set())
    for x in _numbers(text):
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
