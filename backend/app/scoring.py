"""Fixed, documented flood-risk formula. Code decides the score; the LLM only explains it.

Each feature is mapped to 0-1 risk by a simple ramp, weighted, and summed to 0-100.
Missing features are dropped and the remaining weights rescaled; confidence falls
with the share of weight that was missing. Weights are hand-set for now and get
tuned on known flooded / dry validation sites (data/validation_sites.csv).
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import Action, Metric, Score, Tier


@dataclass(frozen=True)
class Feature:
    metric_id: str
    weight: float
    zero_at: float  # value with no added risk
    full_at: float  # value with full risk (may be below zero_at: lower = riskier)

    def risk(self, value: float) -> float:
        span = self.full_at - self.zero_at
        return min(max((value - self.zero_at) / span, 0.0), 1.0)


FEATURES: tuple[Feature, ...] = (
    Feature("m_water_history_500", 0.25, zero_at=0, full_at=30),
    Feature("m_wet_water_250", 0.25, zero_at=0, full_at=30),
    Feature("m_height_above_water", 0.20, zero_at=10, full_at=0.5),
    Feature("m_dist_water", 0.15, zero_at=1500, full_at=100),
    Feature("m_max_daily_rain", 0.15, zero_at=60, full_at=200),
)

TIER_CUTS = ((75, Tier.severe), (50, Tier.high), (25, Tier.moderate), (0, Tier.low))

ALLOWED_ACTIONS: dict[Tier, list[Action]] = {
    Tier.low: [Action.approve],
    Tier.moderate: [Action.approve_note, Action.approve_with_insurance],
    Tier.high: [Action.approve_with_insurance, Action.adjust_terms],
    Tier.severe: [Action.decline, Action.adjust_terms],
}


def tier_for(score: float) -> Tier:
    for cut, tier in TIER_CUTS:
        if score >= cut:
            return tier
    return Tier.low


def score(metrics: list[Metric]) -> Score:
    by_id = {m.id: m for m in metrics}
    present = [f for f in FEATURES if f.metric_id in by_id]
    missing = [f.metric_id for f in FEATURES if f.metric_id not in by_id]
    total_w = sum(f.weight for f in present)
    if total_w == 0:
        return Score(score=0, tier=Tier.low, confidence="low", contributions={}, missing=missing)

    contributions = {
        f.metric_id: round(100 * f.weight / total_w * f.risk(by_id[f.metric_id].value), 1) for f in present
    }
    value = round(sum(contributions.values()), 1)
    covered = total_w / sum(f.weight for f in FEATURES)
    confidence = "high" if covered >= 0.85 else "medium" if covered >= 0.6 else "low"
    return Score(score=value, tier=tier_for(value), confidence=confidence,
                 contributions=contributions, missing=missing)
