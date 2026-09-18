"""Condition matching decides which evidence applies to a site, so it is tested directly."""
from __future__ import annotations

import networkx as nx
import pytest

from reasoning import graph as graph_mod

SEMI_ARID = {"climate_zone": "semi-arid", "land_use": "cropland"}
HUMID = {"climate_zone": "humid", "land_use": "cropland"}
ARID = {"climate_zone": "arid", "land_use": "grassland"}


def test_unconditional_edges_always_hold():
    assert graph_mod.condition_holds(None, SEMI_ARID) is True
    assert graph_mod.condition_holds("", SEMI_ARID) is True


@pytest.mark.parametrize("condition", [
    "semiarid regions", "semi-arid", "dryland areas", "dry climates", "arid and semi-arid regions",
])
def test_dry_conditions_hold_on_a_semi_arid_site(condition):
    assert graph_mod.condition_holds(condition, SEMI_ARID) is True


@pytest.mark.parametrize("condition", [
    "humid", "temperate regions", "cool wet climates", "high annual rainfall areas (>1 000 mm)",
])
def test_wet_conditions_do_not_hold_on_a_semi_arid_site(condition):
    assert graph_mod.condition_holds(condition, SEMI_ARID) is False


def test_semi_arid_condition_does_not_fire_on_an_arid_site():
    # "semiarid" contains "arid"; the site is arid, so the semi-arid edge must not apply.
    assert graph_mod.condition_holds("semiarid regions", ARID) is False
    assert graph_mod.condition_holds("arid regions <336 mm rainfall", ARID) is True


def test_uninterpretable_conditions_are_conservative():
    assert graph_mod.condition_holds("legume cover crops", SEMI_ARID) is False
    assert graph_mod.condition_holds("duration <5 years", SEMI_ARID) is False


def test_land_use_conditions_match_the_site():
    assert graph_mod.condition_holds("cropland", SEMI_ARID) is True
    assert graph_mod.condition_holds("grassland", SEMI_ARID) is False


def test_zone_condition_without_a_known_zone_does_not_fire():
    assert graph_mod.condition_holds("semiarid regions", {"land_use": "cropland"}) is False


def test_node_names_normalise_to_one_vocabulary():
    assert graph_mod.normalise_node("soil_erosion") == "erosion"
    assert graph_mod.normalise_node("crop yield") == "crop_yield"
    assert graph_mod.normalise_node("soil moisture") == "soil_moisture"
    assert graph_mod.normalise_node("SOC") == "SOC"


def _toy_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge("mulching", "soil_moisture", key="e1", sign=1, effect="increase",
                   condition=None, strength="strong", support_count=5, traversable=True,
                   docs=["d1"], units=[], mechanisms=["cuts evaporation"], weight=1.0)
    graph.add_edge("soil_moisture", "SOC", key="e2", sign=1, effect="increase",
                   condition=None, strength="moderate", support_count=2, traversable=True,
                   docs=["d2"], units=[], mechanisms=[], weight=0.7)
    graph.add_edge("cover_crops", "soil_moisture", key="e3", sign=-1, effect="decrease",
                   condition="semiarid regions", strength="moderate", support_count=1,
                   traversable=True, docs=["d3"], units=[], mechanisms=[], weight=0.7)
    graph.add_edge("cover_crops", "crop_yield", key="e4", sign=1, effect="increase",
                   condition="humid", strength="strong", support_count=1, traversable=True,
                   docs=["d4"], units=[], mechanisms=[], weight=1.0)
    return graph


def test_out_edges_drop_conditions_that_do_not_apply():
    graph = _toy_graph()
    targets = [target for target, _ in graph_mod.out_edges(graph, "cover_crops", SEMI_ARID)]
    assert targets == ["soil_moisture"]
    assert graph_mod.out_edges(graph, "cover_crops", HUMID)[0][0] == "crop_yield"


def test_risk_edges_return_only_negative_applicable_effects():
    graph = _toy_graph()
    risks = graph_mod.risk_edges(graph, "cover_crops", SEMI_ARID)
    assert [target for target, _ in risks] == ["soil_moisture"]
    assert graph_mod.risk_edges(graph, "mulching", SEMI_ARID) == []
