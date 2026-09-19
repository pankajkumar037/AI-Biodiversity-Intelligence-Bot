"""Nothing the user did not say may be stamped as something they said."""
from __future__ import annotations

import networkx as nx

from graph_flow.nodes import missing_required
from intake import normalize
from reasoning import causal

MSG = "SOC 0.3%, low rainfall, wheat monoculture, semi-arid Rajasthan"


def test_invented_fields_are_dropped_when_the_message_has_no_word_for_them():
    values = {"soc_percent": 0.3, "cropping_system": "monoculture",
              "soil_moisture_status": "dry", "irrigation": "none",
              "pesticide_use": "low", "biodiversity_trend": "declining"}
    dropped = set(normalize.unsupported_fields(values, MSG))
    assert dropped == {"soil_moisture_status", "irrigation", "pesticide_use",
                       "biodiversity_trend"}


def test_stated_fields_survive_the_guard():
    values = {"pesticide_use": "high", "pollinator_trend": "declining",
              "erosion_observed": True, "irrigation": "drip"}
    msg = "we spray a lot, fewer bees than before, gullies after rain, drip irrigation"
    assert normalize.unsupported_fields(values, msg) == []


def test_the_zone_the_user_named_is_read_directly():
    assert normalize.zone_in_text(MSG) == "semi-arid"
    assert normalize.zone_in_text("hot arid Thar edge") == "arid"
    assert normalize.zone_in_text("humid plantation") == "humid"
    assert normalize.zone_in_text("wheat field") is None


def test_required_slots_gate_the_answer():
    assert missing_required({}) == ["land_use", "soc_percent", "rainfall_mm"]
    assert missing_required({"land_use": "cropland", "soc_percent": 0.3,
                             "climate_zone": "semi-arid"}) == ["cropping_system"]
    assert missing_required({"land_use": "grassland", "soc_percent": 0.4,
                             "rainfall_mm": 350}) == []


def test_irrigation_none_is_not_an_observed_driver():
    assert "irrigation" not in causal.observed_drivers({"irrigation": "none"})
    assert "irrigation" not in causal.observed_drivers({"irrigation": "rainfed"})
    assert "irrigation" in causal.observed_drivers({"irrigation": "drip"})


def test_practices_are_never_root_causes():
    graph = nx.MultiDiGraph()
    for source, target in [("mulching", "SOC"), ("intensive_agriculture", "SOC")]:
        graph.add_edge(source, target, key=f"{source}->{target}", sign=1, effect="increase",
                       condition=None, strength="strong", support_count=3, traversable=True,
                       docs=["d"], units=[], mechanisms=[], weight=1.0)
    found = causal.root_causes(graph, ["SOC"], {"cropping_system": "monoculture"})
    drivers = {item["driver"] for item in found}
    assert "intensive_agriculture" in drivers
    assert "mulching" not in drivers


def test_a_named_crop_implies_cropland_and_is_read_from_the_text():
    assert normalize.crop_in_text(MSG) == "wheat"
    assert normalize.land_use_in_text(MSG) == "cropland"
    assert normalize.land_use_in_text("overgrazed grassland near Jodhpur") == "grassland"
    assert normalize.land_use_in_text("almond orchard, 350 mm") == "orchard"
    assert normalize.land_use_in_text("biodiversity is declining on my land") is None


def test_only_known_constraint_slugs_survive():
    assert "low soil moisture" not in normalize.CONSTRAINT_SLUGS
    assert "leased_land_no_trees" in normalize.CONSTRAINT_SLUGS


def test_stale_fields_from_an_older_build_are_stripped():
    from graph_flow.state import sanitise_profile
    stored = {"audience": {"value": "farmer", "source": "user"},
              "soc_percent": {"value": 0.3, "source": "user"}}
    assert list(sanitise_profile(stored)) == ["soc_percent"]


def test_plainly_stated_facts_are_read_without_the_model():
    facts = normalize.facts_in_text(
        "wheat, heavy pesticide use, and the pollinators are disappearing")
    assert facts == {"pollinator_trend": "declining", "pesticide_use": "high"}
    assert normalize.facts_in_text("we hardly spray; bees are fine") == {}
