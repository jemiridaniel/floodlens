"""Fixed, documented flood-risk formula. Code decides the score; the LLM only explains it.

Each feature is mapped to 0-1 risk by a simple ramp, weighted, and summed to 0-100.
Missing features are dropped and the remaining weights rescaled; confidence falls
with the share of weight that was missing.

The weights below are REASONED PRIORS, not fitted values, and they are weighted by what
each group is actually known to measure.

What we learned the hard way. The satellite features alone put four flood-verified Lagos
sites at the 43rd to 70th percentile of 30 random built-up points in the same city: AUC
0.578, no better than chance. Nothing in the imagery separates a street that floods from
one that does not, because in Lagos the difference is drainage, and drainage is not
visible from orbit. So the satellite half no longer claims to grade a street. It is an
AREA SCREEN for coastal and riverine exposure, which is the thing it can honestly see.

The three groups:

1. LOCAL HISTORY, 0.40. m_reported_floods, from news reports of flooding at this street,
   estate or district, each one verified against the search engine's own citations. This
   is the only direct evidence we have about a specific address, so it carries the most
   weight. It is deliberately a POSITIVE signal only: reports found raise the score, but
   none found does not vouch for a site, because Lagos news covers some neighbourhoods far
   better than others. A site with no reports scores 0 here and leans on the area screen.
   Note the circularity risk when reading validation: sites chosen from news will be found
   in news. The honest test is how often a RANDOM address also turns up street-level
   reports.

2. AREA SCREEN, 0.50, what open satellite data can honestly say about a neighbourhood:
   - Coastal and riverine exposure, 0.25. Water that arrives from a body of water. The
     best-measured thing here (37 years of JRC history), and the one FloodLens should be
     trusted on.
   - Terrain and surface, 0.25. A local dip collects rain (m_local_low_spot) and sealed
     ground stops it soaking away (m_built_up_500); wetland within 500 m marks drained or
     reclaimed low ground. Plausible pluvial proxies that the baseline showed do not
     discriminate between addresses on their own, so they screen an area and no more.

3. RAINFALL, 0.10. The trigger for both, but Open-Meteo is a ~10 km grid: one storm hands
   the whole city nearly the same number (50-79 mm across the baseline). It separates
   years, not addresses, and is weighted accordingly.

The right fix for a bad score is a better measurement, not a bigger weight. Re-check with
scripts/validate.py --baseline once data/validation_sites.csv has dry sites and real pins.
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
    # 1. Local history, 0.40. Weighted events: 1.0 street, 0.6 estate, 0.3 area. Three
    # street-level floods in five years is a street that floods most rainy seasons.
    Feature("m_reported_floods", 0.40, zero_at=0, full_at=3),
    # 2a. Area screen, coastal and riverine exposure, 0.25.
    Feature("m_water_history_500", 0.10, zero_at=0, full_at=30),
    Feature("m_wet_water_250", 0.05, zero_at=0, full_at=30),
    Feature("m_dist_water", 0.05, zero_at=1500, full_at=100),
    Feature("m_height_above_water", 0.05, zero_at=10, full_at=0.5),
    # 2b. Area screen, terrain and surface, 0.25. A 3 m dip is a serious sink; 20-90%
    # built-up spans "open ground left" to "sealed"; any wetland in a 500 m urban ring
    # is already a signal, so 20% is full risk.
    Feature("m_local_low_spot", 0.12, zero_at=0, full_at=-3),
    Feature("m_built_up_500", 0.08, zero_at=20, full_at=90),
    Feature("m_wetland_500", 0.05, zero_at=0, full_at=20),
    # 3. Rainfall, 0.10.
    Feature("m_max_daily_rain", 0.10, zero_at=60, full_at=200),
)

# Evidence groups. Losing one entirely is different from losing a metric: the score then
# rests on kinds of evidence that were never meant to stand alone, so it is reported and
# confidence is capped rather than the weight being quietly rescaled onto the rest.
GROUPS: dict[str, tuple[str, ...]] = {
    "local history": ("m_reported_floods",),
    "coastal and riverine": ("m_water_history_500", "m_wet_water_250", "m_dist_water",
                             "m_height_above_water"),
    "terrain and surface": ("m_local_low_spot", "m_built_up_500", "m_wetland_500"),
    "rainfall": ("m_max_daily_rain",),
}
AREA_SCREEN = ("coastal and riverine", "terrain and surface")  # the two halves of the 0.50

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
        return Score(score=0, tier=Tier.low, confidence="low", contributions={}, missing=missing,
                     missing_groups=list(GROUPS))

    contributions = {
        f.metric_id: round(100 * f.weight / total_w * f.risk(by_id[f.metric_id].value), 1) for f in present
    }
    value = round(sum(contributions.values()), 1)
    covered = total_w / sum(f.weight for f in FEATURES)
    confidence = "high" if covered >= 0.85 else "medium" if covered >= 0.6 else "low"
    # A whole kind of evidence missing is not the same as a gap inside one: the remaining
    # weight is carrying work it was never weighted to carry, so say so and cap confidence.
    gone = [g for g, ids in GROUPS.items() if all(i in missing for i in ids)]
    if gone:
        confidence = "low"
    return Score(score=value, tier=tier_for(value), confidence=confidence,
                 contributions=contributions, missing=missing, missing_groups=gone)
