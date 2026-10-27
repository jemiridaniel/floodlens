"""Fixed, documented flood-risk formula. Code decides the score; the LLM only explains it.

Each feature is mapped to 0-1 risk by a simple ramp, weighted, and summed to 0-100.
Missing features are dropped and the remaining weights rescaled; confidence falls
with the share of weight that was missing.

The weights below are REASONED PRIORS, not fitted values. Nothing here is tuned to the
validation set: four sites cannot support fitting eight weights, and tuning on them would
turn the one honest test we have into a training set. They encode one finding and one
piece of physical reasoning.

The finding: the first validation run scored four flood-verified Lagos sites at 0.0, 1.2,
2.1 and 30.9 out of 100. Every feature then in the formula measured proximity to a body
of water, so the formula could only see riverine and coastal flooding. Three of the four
sites have no water at all within 2 km; they flood because rain cannot drain away. That is
pluvial flooding, and it was worth 0 weight.

The reasoning behind the split:
- Pluvial 0.45. Rain floods a site when the water has somewhere to collect (a local dip,
  m_local_low_spot, the strongest single signal at 0.20) and nowhere to soak away
  (impervious ground, m_built_up_500, 0.15). Wetland or mangrove within 500 m,
  m_wetland_500 at 0.10, marks ground that was drained or reclaimed and still sits low.
- Riverine and coastal 0.45, the same share, split across the four water-body features.
  This is the better-measured half -- 37 years of JRC history is stronger evidence than a
  land-cover proxy -- but it describes a kind of flooding that most inland Lagos
  collateral is not exposed to, so it does not deserve a larger share.
- Rainfall 0.10. The trigger for both, but weakly site-specific: one storm covers the
  whole city, so it separates years, not addresses.

Re-check all of this against data/validation_sites.csv (scripts/validate.py) once the set
has dry sites and hand-placed pins. Until then these are priors, and the right fix for a
bad score is a better measurement, not a bigger weight.
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
        r = min(max((value - self.zero_at) / span, 0.0), 1.0)
        return 0.0 if r == 0 else r  # a descending ramp (full_at < zero_at) yields -0.0


FEATURES: tuple[Feature, ...] = (
    # Pluvial: rain with somewhere to collect and nowhere to drain. 0.45 total.
    # A 3 m dip is a serious sink; 20-90% built-up spans "some open ground left" to
    # "effectively sealed"; any wetland in a 500 m urban ring is already a signal, so
    # 20% is treated as full risk.
    Feature("m_local_low_spot", 0.20, zero_at=0, full_at=-3),
    Feature("m_built_up_500", 0.15, zero_at=20, full_at=90),
    Feature("m_wetland_500", 0.10, zero_at=0, full_at=20),
    # Riverine and coastal: water arriving from a body of water. 0.45 total.
    Feature("m_water_history_500", 0.15, zero_at=0, full_at=30),
    Feature("m_wet_water_250", 0.10, zero_at=0, full_at=30),
    Feature("m_dist_water", 0.10, zero_at=1500, full_at=100),
    Feature("m_height_above_water", 0.10, zero_at=10, full_at=0.5),
    # The trigger for both, but it rains on the whole city at once. 0.10.
    Feature("m_max_daily_rain", 0.10, zero_at=60, full_at=200),
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
