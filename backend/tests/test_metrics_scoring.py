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
    # m_reported_floods is never produced offline: it needs the web search
    assert set(s.missing) == {"m_wet_water_250", "m_height_above_water", "m_local_low_spot",
                              "m_reported_floods"}
    assert s.confidence == "low"  # 0.62 of the weight gone
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
    assert "m_dist_water" not in s.missing
    assert set(s.missing) == {"m_height_above_water", "m_reported_floods"}


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
    # the sign is the whole point, so the reader gets a sentence rather than "-2.5 m"
    assert m["m_local_low_spot"].display_value == "2.5 m lower than the surrounding ground"


def test_low_spot_is_zero_on_flat_ground_and_positive_on_a_rise(layers):
    import numpy as np
    layers.dem = np.full_like(layers.dem, 10.0)
    assert by_id(compute_metrics(layers))["m_local_low_spot"].value == pytest.approx(0.0)
    assert by_id(compute_metrics(layers))["m_local_low_spot"].display_value == (
        "level with the surrounding ground")
    layers.dem[90:110, 90:110] = 13.0  # the site sits on a rise instead
    high = by_id(compute_metrics(layers))["m_local_low_spot"]
    assert high.value == pytest.approx(3.0)
    assert high.display_value == "3.0 m higher than the surrounding ground"


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
    assert set(score(compute_metrics(layers)).missing) == {"m_built_up_500", "m_wetland_500",
                                                           "m_reported_floods"}


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


def test_the_three_groups_carry_their_documented_weight():
    """The split is a design decision, not an accident of editing. FROZEN: changing these
    numbers needs new, independent validation sites, not a better-looking table."""
    from app.scoring import AREA_SCREEN, GROUPS
    weight = {f.metric_id: f.weight for f in FEATURES}
    by_group = {g: round(sum(weight[m] for m in ids), 2) for g, ids in GROUPS.items()}
    assert by_group == {"local history": 0.40, "coastal and riverine": 0.25,
                        "terrain and surface": 0.25, "rainfall": 0.10}
    assert round(sum(by_group[g] for g in AREA_SCREEN), 2) == 0.50  # the area screen
    assert set(weight) == {m for ids in GROUPS.values() for m in ids}  # every feature in a group


# ------------------------------------------- evidence groups and layer retries


def test_a_wholly_missing_group_caps_confidence_and_is_named(layers):
    """Losing one metric rescales. Losing a whole kind of evidence must be said out loud."""
    from app.scoring import GROUPS
    layers.landcover = None  # takes out built-up and wetland
    layers.dem = None        # ...and the low spot: the whole terrain and surface group
    s = score(compute_metrics(layers))
    assert "terrain and surface" in s.missing_groups
    assert s.confidence == "low"
    assert set(s.missing_groups) <= set(GROUPS)


def test_a_partly_missing_group_is_not_flagged(layers):
    layers.landcover = None  # built-up and wetland go, the low spot stays
    s = score(compute_metrics(layers))
    assert "terrain and surface" not in s.missing_groups


def test_offline_metrics_always_lack_local_history(layers):
    """compute_metrics never produces m_reported_floods, so the group is always absent
    without a search: an area-screen-only assessment is honestly low confidence."""
    s = score(compute_metrics(layers))
    assert s.missing_groups == ["local history"]
    assert s.confidence == "low"


def test_no_metrics_names_every_group():
    from app.scoring import GROUPS
    assert set(score([]).missing_groups) == set(GROUPS)


def test_a_layer_is_retried_once_before_being_given_up_on():
    from app.geo.sources import try_twice
    notes: list[str] = []
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "layer"

    assert try_twice("optical", flaky, notes) == "layer"
    assert len(calls) == 2 and notes == []  # recovered, and nothing to report

    notes.clear()
    always = []

    def broken():
        always.append(1)
        raise RuntimeError("down")

    assert try_twice("optical", broken, notes) is None
    assert len(always) == 2
    assert notes and "after 2 tries" in notes[0] and "RuntimeError" in notes[0]


# ------------------------------------------------- the street-level tier floor
def test_two_street_events_hold_the_tier_at_high():
    """An evidence rule, not a weight: verified repeat flooding on this street outranks
    any combination of proxies."""
    from app.scoring import TIER_FLOOR, TIER_FLOOR_STREET_EVENTS, apply_tier_floor
    assert TIER_FLOOR_STREET_EVENTS == 2 and TIER_FLOOR == Tier.high

    for start in (Tier.low, Tier.moderate):
        raised, why = apply_tier_floor(start, 2)
        assert raised == Tier.high
        assert "2 distinct street-level flood events" in why and "5 years" in why


def test_the_floor_never_lowers_a_tier():
    from app.scoring import apply_tier_floor
    for tier in (Tier.high, Tier.severe):
        held, why = apply_tier_floor(tier, 5)
        assert held == tier and why is None


def test_one_event_is_not_enough():
    from app.scoring import apply_tier_floor
    for n in (0, 1):
        tier, why = apply_tier_floor(Tier.low, n)
        assert tier == Tier.low and why is None


def test_the_floor_shows_in_the_score_and_names_itself():
    low = score([Metric(id="m_max_daily_rain", label="", value=0, unit="mm", source="")])
    assert low.tier == Tier.low and low.tier_floor is None

    floored = score([Metric(id="m_max_daily_rain", label="", value=0, unit="mm", source="")],
                    street_events=2)
    assert floored.tier == Tier.high
    assert floored.score == low.score  # the score is untouched; only the tier is held
    assert floored.tier_floor and "street-level" in floored.tier_floor


def test_one_flood_reported_by_three_outlets_does_not_reach_the_floor():
    """The floor counts events, not articles, or three outlets covering one storm would
    push every reported street to High."""
    from app.agent.history import distinct_events
    from app.schemas import FloodReport
    same_storm = [FloodReport(place_named="X", match_level="street", event_date=d, publisher=p,
                              url=f"https://example.test/{p}", quote="q")
                  for d, p in (("2026-06-29", "a"), ("2026-06-30", "b"), ("2026-06-30", "c"))]
    assert len(distinct_events(same_storm)) == 1

    two_storms = same_storm + [FloodReport(place_named="X", match_level="street",
                                           event_date="2025-09-16", publisher="d",
                                           url="https://example.test/d", quote="q")]
    assert len(distinct_events(two_storms)) == 2
