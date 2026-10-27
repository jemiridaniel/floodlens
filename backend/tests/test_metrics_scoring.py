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
    layers.dem = None  # takes height above water and the low spot with it
    s = score(compute_metrics(layers))
    assert set(s.missing) == {"m_wet_water_250", "m_height_above_water", "m_local_low_spot"}
    assert s.confidence == "medium"  # 0.40 of the weight gone
    assert 0 <= s.score <= 100


def test_losing_every_layer_but_rain_is_low_confidence(layers):
    layers.radar = layers.dem = layers.landcover = layers.jrc_occurrence = layers.optical = None
    s = score(compute_metrics(layers))
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


# ---------------------------------------------------------- pluvial features


def test_no_water_in_view_is_a_measurement_not_a_gap(layers):
    """An inland site with no water in the grid: distance saturates at the grid edge and
    says so, and only height above water is dropped."""
    import numpy as np
    layers.jrc_occurrence = np.zeros_like(layers.jrc_occurrence)
    layers.optical.scl[:] = 5  # urban, no optical water anywhere
    layers.optical.green[:], layers.optical.nir[:] = 900, 2500

    m = by_id(compute_metrics(layers))
    assert m["m_dist_water"].value == 1000  # half of the 2 km grid
    assert m["m_dist_water"].label == "No regular water within 1 km"
    assert "no water in the 2 km grid" in m["m_dist_water"].source
    assert "m_height_above_water" not in m
    # the weight stays on the formula instead of being rescaled away
    s = score(compute_metrics(layers))
    assert "m_dist_water" not in s.missing and s.missing == ["m_height_above_water"]


def test_dist_water_is_dropped_only_when_nothing_could_see_water(layers):
    layers.jrc_occurrence = None
    layers.optical = None
    m = by_id(compute_metrics(layers))
    assert "m_dist_water" not in m  # no water layer at all: we cannot claim 1 km either


def test_low_spot_is_negative_in_a_dip(layers):
    """m_local_low_spot: the 10th percentile within 100 m minus the median within 1 km."""
    import numpy as np
    layers.dem = np.full_like(layers.dem, 10.0)
    layers.dem[90:110, 90:110] = 7.5  # a 200 m dip centred on the site
    m = by_id(compute_metrics(layers))
    assert m["m_local_low_spot"].value == pytest.approx(-2.5)
    assert "negative = a dip" in m["m_local_low_spot"].label


def test_low_spot_is_zero_on_flat_ground_and_positive_on_a_rise(layers):
    import numpy as np
    layers.dem = np.full_like(layers.dem, 10.0)
    assert by_id(compute_metrics(layers))["m_local_low_spot"].value == pytest.approx(0.0)
    layers.dem[90:110, 90:110] = 13.0  # the site sits on a rise instead
    assert by_id(compute_metrics(layers))["m_local_low_spot"].value == pytest.approx(3.0)


def test_low_spot_risk_ramp():
    f = next(f for f in FEATURES if f.metric_id == "m_local_low_spot")
    assert f.risk(0) == 0.0 and f.risk(1.5) == 0.0  # a rise adds nothing
    assert f.risk(-3) == 1.0 and f.risk(-6) == 1.0  # saturates at a 3 m dip
    assert f.risk(-1.5) == pytest.approx(0.5)


def test_a_descending_ramp_never_reports_negative_zero():
    f = next(f for f in FEATURES if f.metric_id == "m_local_low_spot")
    assert str(f.risk(0)) == "0.0"
    assert str(score([Metric(id="m_local_low_spot", label="", value=0.0, unit="m",
                             source="")]).contributions["m_local_low_spot"]) == "0.0"


def test_built_up_and_wetland_from_worldcover(layers):
    """Class 50 is built-up, 90 and 95 are wetland and mangrove; 0 is outside the tile."""
    import numpy as np
    n = layers.grid.side
    lc = np.full((n, n), 50, "float32")
    lc[:40, :] = 10   # tree cover, outside the 500 m ring anyway
    lc[120:130, :] = 95  # mangrove inside the ring
    layers.landcover = lc
    m = by_id(compute_metrics(layers))
    assert m["m_built_up_500"].value == pytest.approx(100 - m["m_wetland_500"].value, abs=0.01)
    assert 0 < m["m_wetland_500"].value < 30
    assert m["m_built_up_500"].source == "ESA WorldCover 10 m"


def test_worldcover_no_data_pixels_do_not_count(layers):
    """0 means outside the tile. It must not dilute the built-up share."""
    import numpy as np
    lc = np.full((layers.grid.side, layers.grid.side), 50, "float32")
    lc[:, :100] = 0  # half the grid has no data
    layers.landcover = lc
    assert by_id(compute_metrics(layers))["m_built_up_500"].value == pytest.approx(100.0)


def test_landcover_metrics_are_dropped_without_the_layer(layers):
    layers.landcover = None
    m = by_id(compute_metrics(layers))
    assert "m_built_up_500" not in m and "m_wetland_500" not in m
    assert set(score(compute_metrics(layers)).missing) == {"m_built_up_500", "m_wetland_500"}


def test_pluvial_site_with_no_water_nearby_still_scores(layers):
    """The point of the rebalance: a dip in sealed ground with no water in view is a risk."""
    import numpy as np
    layers.jrc_occurrence = np.zeros_like(layers.jrc_occurrence)
    layers.optical.scl[:] = 5
    layers.optical.green[:], layers.optical.nir[:] = 900, 2500
    layers.radar = None
    layers.dem = np.full_like(layers.dem, 10.0)
    layers.dem[90:110, 90:110] = 7.0  # a 3 m dip
    layers.landcover = np.full_like(layers.landcover, 50)  # fully built up
    s = score(compute_metrics(layers))
    assert s.score > 40, s.contributions  # before the rebalance this site scored near zero
    assert s.tier in (Tier.moderate, Tier.high)


def test_pluvial_and_riverine_carry_equal_weight():
    """The rebalance is a design decision, not an accident of editing."""
    pluvial = {"m_local_low_spot", "m_built_up_500", "m_wetland_500"}
    riverine = {"m_water_history_500", "m_wet_water_250", "m_dist_water", "m_height_above_water"}
    weight = {f.metric_id: f.weight for f in FEATURES}
    assert sum(weight[m] for m in pluvial) == pytest.approx(0.45)
    assert sum(weight[m] for m in riverine) == pytest.approx(0.45)
    assert weight["m_max_daily_rain"] == pytest.approx(0.10)
