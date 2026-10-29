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
6. When an evidence rule raised the tier above the weighted score, the report may not
   write as though the score earned it. "Scores 42.5 and is High tier" is false: 42.5 is
   Moderate, and the tier came from two verified street-level floods. A reader who
   believes the score produced the tier will mistrust the number or the tier, and both
   are right.
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


def numbers(text: str) -> list[float]:
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


SENTENCE = re.compile(r"(?<=[.!?])\s+")


# Phrases that show a sentence is attributing the tier to the rule rather than the score.
FLOOR_LANGUAGE = re.compile(r"held at|evidence rule|raised to|floor", re.I)


def floored_tier_errors(text: str, score: Score) -> list[str]:
    """Catch text that credits the weighted score with a tier the evidence rule set.

    A sentence naming both the score and the floored tier is only wrong if it reads as
    though one produced the other. "Weighted score 42.5 (Moderate), held at High by the
    evidence rule" names both and is exactly right, so a sentence that also names the
    lower tier the score really gives, or that says where the tier came from, is left
    alone. Only "scores 42.5 and is High tier" is the error.
    """
    if not score.tier_floor or score.weighted_tier is None:
        return []
    tier = re.compile(rf"\b{score.tier.value}\b", re.I)
    weighted = re.compile(rf"\b{score.weighted_tier.value}\b", re.I)
    figure = re.compile(rf"(?<![\d.]){re.escape(f'{score.score:g}')}(?![\d])")
    errors = []
    for sentence in SENTENCE.split(" ".join(text.split())):
        if not (tier.search(sentence) and figure.search(sentence)):
            continue
        if weighted.search(sentence) or FLOOR_LANGUAGE.search(sentence):
            continue  # the sentence says where the tier came from: that is the right form
        errors.append(
            f"this sentence credits the score with the tier, but {score.score:g} is "
            f"{score.weighted_tier.value} and the tier was held at {score.tier.value} by "
            f"the evidence rule: {sentence.strip()!r}")
    return errors


def verify(text: str, tier: str, action: str, metrics: list[Metric], score: Score,
           extra_allowed: set[float] | None = None, required: list[str] | None = None) -> Verdict:
    errors: list[str] = []
    flat = " ".join(text.split())
    for phrase in required or []:
        if " ".join(phrase.split()) not in flat:
            errors.append(f"the report must contain this sentence exactly: {phrase!r}")
    errors += floored_tier_errors(text, score)
    allowed = [(m.value, m.decimals) for m in metrics] + [(score.score, 1)]
    ok_set = ALWAYS_OK | (extra_allowed or set())

    # Dates first, as whole tokens. A metric's own date may be quoted; any other date is
    # invented. Matched dates are then cut out of the text, so the number rule never sees
    # "2025-06-14" as the measurements 6 and 14.
    measured_dates = {d for m in metrics for d in DATE.findall(f"{m.label} {m.source}")}
    for d in DATE.findall(text):
        if d not in measured_dates:
            errors.append(f"date {d} is not in any measured metric")

    for x in numbers(DATE.sub(" ", text)):
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
