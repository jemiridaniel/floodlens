import pytest

from app.geo.water import SCL_WATER
from app.metrics import compute_metrics
from app.schemas import Metric, Tier
from app.scoring import FEATURES, score, tier_for

from .conftest import make_layers


def by_id(metrics):
    return {m.id: m for m in metrics}


def test_metrics_from_synthetic_lagoon(layers):
    m = by_id(compute_metrics(layers))
    # lagoon starts 30 px (300 m) south of the site; on the wet date water reaches the site itself
    assert m["m_dist_water"].value == pytest.approx(300, abs=10)
    assert m["m_height_above_water"].value == pytest.approx(2.0)
    assert m["m_water_now_250"].value == 0  # 250 m ring stops short of the lagoon
    assert 40 < m["m_wet_water_250"].value < 60  # flooded half of the ring
    assert m["m_max_daily_rain"].value == 140
    assert "2025-06-10" in m["m_max_daily_rain"].label


def test_wet_low_site_scores_higher_than_dry_high_site():
    wet = score(compute_metrics(make_layers(flooded_rows=30, site_height=1.0)))
    dry = score(compute_metrics(make_layers(water_rows=10, flooded_rows=0, site_height=12.0)))
    assert wet.score > dry.score
    assert wet.tier in (Tier.high, Tier.severe)
    assert dry.tier in (Tier.low, Tier.moderate)


def test_missing_layers_lower_confidence_not_crash(layers):
    layers.radar = None
    layers.dem = None
    s = score(compute_metrics(layers))
    assert set(s.missing) == {"m_wet_water_250", "m_height_above_water"}
    assert s.confidence == "low"
    assert 0 <= s.score <= 100


def test_no_metrics_is_low_confidence_zero():
    s = score([])
    assert s.score == 0 and s.tier == Tier.low and s.confidence == "low"


def test_tier_cuts():
    assert [tier_for(x) for x in (0, 24.9, 25, 49.9, 50, 74.9, 75, 100)] == [
        Tier.low, Tier.low, Tier.moderate, Tier.moderate, Tier.high, Tier.high, Tier.severe, Tier.severe]


def test_weights_sum_to_one():
    assert sum(f.weight for f in FEATURES) == pytest.approx(1.0)


def test_lower_height_is_riskier():
    f = next(f for f in FEATURES if f.metric_id == "m_height_above_water")
    assert f.risk(0.5) == 1.0 and f.risk(10) == 0.0 and 0 < f.risk(5) < 1


def test_full_risk_everywhere_is_100():
    metrics = [Metric(id=f.metric_id, label="", value=f.full_at, unit="", source="") for f in FEATURES]
    assert score(metrics).score == pytest.approx(100, abs=0.5)


def test_distance_prefers_jrc_history_over_one_image(layers):
    import numpy as np
    layers.optical.scl[95:100, 95:100] = 6  # one-date "water" right next to the site
    m = by_id(compute_metrics(layers))
    assert m["m_dist_water"].value == pytest.approx(300, abs=10)
    assert m["m_dist_water"].source == "JRC Global Surface Water"
    layers.jrc_occurrence = np.zeros_like(layers.jrc_occurrence)  # no JRC water: fall back
    assert by_id(compute_metrics(layers))["m_dist_water"].value < 60


def test_shadow_water_with_no_history_is_not_counted(layers):
    """Tall-building shadow beside the site: SCL calls it water, JRC never saw water there."""
    layers.optical.scl[95:100, 95:100] = SCL_WATER  # 25 px of "water" about 50 m from the site
    assert layers.jrc_occurrence[95:100, 95:100].max() == 0

    m = by_id(compute_metrics(layers))["m_water_now_250"]
    assert m.value == 0
    assert m.label == "Water within 250 m on a clear date, confirmed by history"
    assert "JRC" in m.source

    layers.jrc_occurrence = None  # nothing to check against: count it, but say so
    unconfirmed = by_id(compute_metrics(layers))["m_water_now_250"]
    assert unconfirmed.value > 1
    assert unconfirmed.label == "Water within 250 m on a clear date (unconfirmed)"


def test_any_past_water_confirms_optical_water(layers):
    """Occurrence above 0 confirms, even well below the regular-water threshold."""
    layers.optical.scl[95:100, 95:100] = SCL_WATER
    layers.jrc_occurrence[95:100, 95:100] = 5  # seen rarely, far under JRC_PERMANENT
    assert by_id(compute_metrics(layers))["m_water_now_250"].value > 1


def test_all_nodata_history_counts_as_unconfirmed(layers):
    """A JRC window that is entirely no-data must not silently zero the metric."""
    import numpy as np
    layers.optical.scl[95:100, 95:100] = SCL_WATER
    layers.jrc_occurrence = np.full_like(layers.jrc_occurrence, np.nan)
    m = by_id(compute_metrics(layers))["m_water_now_250"]
    assert m.value > 1
    assert m.label.endswith("(unconfirmed)")
