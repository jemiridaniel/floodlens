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


def test_a_required_sentence_must_be_present_word_for_word():
    missing = verify("All quiet here.", "High", "approve_with_insurance", METRICS, SCORE,
                     required=["No street-level flood reports found. This does not mean the site is safe."])
    assert not missing.ok
    assert any("must contain this sentence exactly" in e for e in missing.errors)

    present = verify("All quiet here. No street-level flood reports found. This does not mean the "
                     "site is safe.", "High", "approve_with_insurance", METRICS, SCORE,
                     required=["No street-level flood reports found. This does not mean the site is safe."])
    assert present.ok, present.errors


def test_a_required_sentence_tolerates_line_wrapping_only():
    phrase = "No street-level flood reports found. This does not mean the site is safe."
    wrapped = verify("No street-level flood reports found.\n   This does not mean the site is safe.",
                     "High", "approve_with_insurance", METRICS, SCORE, required=[phrase])
    assert wrapped.ok, wrapped.errors

    reworded = verify("No street-level flood reports were found; that is not a guarantee.",
                      "High", "approve_with_insurance", METRICS, SCORE, required=[phrase])
    assert not reworded.ok  # paraphrase is not enough


# ------------------------------------------- the floored tier may not be credited to the score
FLOORED = Score(score=42.5, tier=Tier.high, confidence="high", contributions={}, missing=[],
                tier_floor="2 distinct street-level flood events in the last 5 years",
                weighted_tier=Tier.moderate)


def test_a_report_may_not_say_the_score_earned_a_floored_tier():
    """42.5 is Moderate. The tier came from two verified floods, not from the arithmetic."""
    v = verify("This site scores 42.5 and is High tier.", "High", "approve_with_insurance",
               METRICS, FLOORED)
    assert not v.ok
    assert any("credits the score with the tier" in e for e in v.errors)
    assert any("42.5 is Moderate" in e for e in v.errors)


def test_the_two_may_be_stated_in_separate_sentences():
    good = ("The weighted score is 42.5, which on its own is Moderate. The tier is held at "
            "High because flooding was reported on this street twice.")
    assert verify(good, "High", "approve_with_insurance", METRICS, FLOORED).ok


def test_the_rule_is_silent_when_no_floor_was_applied():
    plain = Score(score=58.4, tier=Tier.high, confidence="high", contributions={}, missing=[])
    assert verify("This site scores 58.4 and is High tier.", "High", "approve_with_insurance",
                  METRICS, plain).ok


def test_the_rule_only_fires_on_the_floored_tier_word():
    """Mentioning the score beside a different tier word is not the error we are catching."""
    v = verify("The weighted score of 42.5 sits in the Moderate band.", "High",
               "approve_with_insurance", METRICS, FLOORED)
    assert v.ok, v.errors


def test_the_prescribed_sentence_is_accepted():
    """The one form the agent is told to use must not be the one the rule rejects."""
    prescribed = ("Weighted score 42.5 (Moderate), held at High by the evidence rule: "
                  "2 distinct street-level flood events in the last 5 years.")
    v = verify(prescribed, "High", "approve_with_insurance", METRICS, FLOORED,
               extra_allowed={2.0, 5.0})
    assert v.ok, v.errors


def test_naming_where_the_tier_came_from_is_enough():
    for ok in ("The tier is held at High even though the weighted score is 42.5.",
               "Scoring 42.5, this site is raised to High by the evidence rule.",
               "A weighted 42.5 would be Moderate; the floor puts it at High."):
        assert verify(ok, "High", "approve_with_insurance", METRICS, FLOORED).ok, ok


def test_the_bare_claim_is_still_rejected():
    v = verify("This site scores 42.5 and is High tier.", "High", "approve_with_insurance",
               METRICS, FLOORED)
    assert not v.ok and any("credits the score" in e for e in v.errors)
