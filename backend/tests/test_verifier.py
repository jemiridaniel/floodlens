from app.agent.verifier import verify
from app.schemas import Metric, Score, Tier

METRICS = [
    Metric(id="m_wet_water_250", label="Area under water within 250 m on a wet date", value=14.27, unit="%",
           source="Sentinel-1 2025-06-16, threshold -14.0 dB"),
    Metric(id="m_dist_water", label="", value=312.0, unit="m", source="", decimals=0),
    Metric(id="m_max_daily_rain", label="Heaviest daily rainfall (on 2025-06-14)", value=75.0, unit="mm",
           source="Open-Meteo archive", decimals=0),
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


def test_accepts_a_date_the_metrics_state():
    """The rainfall metric's label carries its date, so the report may quote it."""
    v = verify("The heaviest day was 75 mm on 2025-06-14, and the radar date was 2025-06-16.",
               "High", "approve_with_insurance", METRICS, SCORE)
    assert v.ok, v.errors


def test_rejects_an_invented_date():
    v = verify("The site flooded on 2024-03-11.", "High", "adjust_terms", METRICS, SCORE)
    assert not v.ok
    assert any("2024-03-11" in e for e in v.errors)


def test_a_date_does_not_launder_its_parts_as_numbers():
    """2025-06-14 must not make 6 or 14 quotable."""
    v = verify("Water covered 14% of the ring on 2025-06-14.", "High", "adjust_terms", METRICS, SCORE)
    assert not v.ok
    assert any("14" in e and "does not match" in e for e in v.errors)


def test_a_number_in_a_source_is_still_not_quotable():
    """A source reads 'threshold -14.0 dB'; that does not make 14% a measured value."""
    v = verify("About 14% of the ring was wet.", "High", "adjust_terms", METRICS, SCORE)
    assert not v.ok and any("14" in e for e in v.errors)
    assert verify("Threshold was -14.0 dB.", "High", "adjust_terms", METRICS, SCORE).errors
