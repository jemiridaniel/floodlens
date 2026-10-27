from app.agent.verifier import verify
from app.schemas import Metric, Score, Tier

METRICS = [
    Metric(id="m_wet_water_250", label="", value=14.27, unit="%", source=""),
    Metric(id="m_dist_water", label="", value=312.0, unit="m", source="", decimals=0),
]
SCORE = Score(score=58.4, tier=Tier.high, confidence="high", contributions={}, missing=[])


def test_accepts_grounded_report():
    text = ("High risk (58.4/100): 14.3% of the 250 m ring was under water on a wet date in 2025, "
            "and the site is 312 m from regular water.")
    v = verify(text, "High", "approve_with_insurance", METRICS, SCORE)
    assert v.ok, v.errors


def test_rejects_invented_number():
    v = verify("Water reached 1.2 m inside the shop.", "High", "adjust_terms", METRICS, SCORE)
    assert not v.ok and "1.2" in v.errors[0]


def test_rejects_wrong_tier_and_disallowed_action():
    v = verify("Looks fine.", "Low", "approve", METRICS, SCORE)
    assert not v.ok
    assert any("tier Low" in e for e in v.errors)
    assert any("not allowed" in e for e in v.errors)


def test_rejects_unknown_values():
    v = verify("ok", "Medium", "yolo", METRICS, SCORE)
    assert len(v.errors) == 2
