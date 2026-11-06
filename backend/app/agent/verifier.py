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
6. Where no street-level flood report was found, the report may not write as though the
   site has a clean flood history. We did not look at a flood register; we ran a news
   search that covers some parts of Lagos far better than others. "History is clean" is a
   claim about the world; "no reports found" is a claim about our search, and only the
   second one is true.
7. When an evidence rule raised the tier above the weighted score, the report may not
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

# Wordings that assert the site has no flood history, rather than reporting what the
# search found. Deliberately narrow: "no street-level flood reports found" is the
# required sentence and must not match any of these.
CLEAN_HISTORY_CLAIMS = (
    (r"\bhistory\s+(?:is|was|looks|appears|remains)\s+clean\b", "says the history is clean"),
    (r"\bclean\s+(?:flood\s+)?(?:history|record)\b", "calls the record clean"),
    (r"\bno\s+(?:flood|flooding)\s+history\b", "asserts no flood history"),
    (r"\bno\s+history\s+of\s+(?:flood|flooding)\b", "asserts no history of flooding"),
    (r"\bnever\s+(?:been\s+)?flooded\b", "asserts the site never flooded"),
    (r"\bno\s+(?:past|previous|prior|recorded|known)\s+flood", "asserts no past flooding"),
    (r"\bno\s+record\s+of\s+(?:flood|flooding)\b", "asserts no record of flooding"),
    (r"\bhistory\s+shows?\s+no\b", "says the history shows nothing"),
    (r"\b(?:free\s+of|unaffected\s+by)\s+flood", "calls the site free of flooding"),
    (r"\bnot\s+flooded\s+(?:before|in\s+the\s+past)\b", "asserts it has not flooded"),
)


def clean_history_errors(text: str) -> list[str]:
    """Catch a report claiming the site has no flood history when none was established.

    A news search that finds nothing says something about the search, not about the site:
    Lagos coverage is uneven, and the places least covered are not the places least
    flooded. The report may say what the search found and must not go beyond it.
    """
    flat = " ".join(text.split())
    errors = []
    for pattern, what in CLEAN_HISTORY_CLAIMS:
        found = re.search(pattern, flat, re.I)
        if found:
            errors.append(
                f"no street-level flood report was found for this site, so the report cannot "
                f"claim a clean history: this {what} ({found.group(0)!r}). Say what the search "
                f"found, not what the site has never done.")
    return errors


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
           extra_allowed: set[float] | None = None, required: list[str] | None = None,
           allowed_actions: list[Action] | None = None,
           has_street_reports: bool = True) -> Verdict:
    errors: list[str] = []
    flat = " ".join(text.split())
    for phrase in required or []:
        if " ".join(phrase.split()) not in flat:
            errors.append(f"the report must contain this sentence exactly: {phrase!r}")
    errors += floored_tier_errors(text, score)
    if not has_street_reports:
        errors += clean_history_errors(text)
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
    permitted = ALLOWED_ACTIONS[score.tier] if allowed_actions is None else allowed_actions
    if act is not None and act not in permitted:
        errors.append(f"action {act.value} is not allowed here; permitted: "
                      + ", ".join(a.value for a in permitted))

    return Verdict(ok=not errors, errors=errors)
