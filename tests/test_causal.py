"""Propagation must be signed, decayed, condition-aware and loop-free."""
from __future__ import annotations

import networkx as nx

from reasoning import causal


def _toy_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    def edge(source, target, sign, condition=None, weight=1.0, key=None):
        graph.add_edge(source, target, key=key or f"{source}->{target}", sign=sign,
                       effect="increase" if sign > 0 else "decrease", condition=condition,
                       strength="strong", support_count=3, traversable=True,
                       docs=["d"], units=[], mechanisms=[], weight=weight)

    edge("mulching", "soil_moisture", 1)
    edge("soil_moisture", "SOC", 1)
    edge("SOC", "soil_moisture", 1)          # closes a reinforcing loop
    edge("cover_crops", "soil_moisture", -1, condition="semiarid regions")
    edge("cover_crops", "SOC", 1)
    return graph


SITE = {"climate_zone": "semi-arid", "land_use": "cropland"}


def test_propagation_decays_with_distance():
    effects, paths = causal.propagate(_toy_graph(), "mulching", SITE, max_hops=2, decay=0.5)
    assert effects["soil_moisture"] > 0
    assert effects["SOC"] > 0
    # SOC is one hop further than soil_moisture, so it must carry less weight.
    assert effects["SOC"] < effects["soil_moisture"]
    assert all(len(set(path["path"])) == len(path["path"]) for path in paths)


def test_a_negative_edge_produces_a_negative_effect():
    effects, _ = causal.propagate(_toy_graph(), "cover_crops", SITE, max_hops=1)
    assert effects["soil_moisture"] < 0
    assert effects["SOC"] > 0


def test_the_same_practice_scores_differently_in_a_humid_site():
    humid = {"climate_zone": "humid", "land_use": "cropland"}
    dry, _ = causal.propagate(_toy_graph(), "cover_crops", SITE, max_hops=1)
    wet, _ = causal.propagate(_toy_graph(), "cover_crops", humid, max_hops=1)
    assert dry["soil_moisture"] < 0
    assert "soil_moisture" not in wet


def test_paths_never_revisit_a_node():
    _, paths = causal.propagate(_toy_graph(), "mulching", SITE, max_hops=3)
    for path in paths:
        assert len(set(path["path"])) == len(path["path"])


def test_benefit_follows_the_direction_each_flag_wants():
    effects = {"SOC": 0.8, "soil_moisture": -0.4, "erosion": -0.5}
    assert causal.benefit_for_flags(effects, ["low_SOC"]) == 0.8
    assert causal.benefit_for_flags(effects, ["water_limited"]) == -0.4
    # bare_fallow wants erosion to fall, so a negative effect is a benefit.
    assert causal.benefit_for_flags(effects, ["bare_fallow"]) == 0.5


def test_flag_targets_ignore_flags_with_no_metric():
    assert causal.flag_targets(["low_SOC", "not_a_flag"]) == {"SOC": 1}


def test_root_causes_trace_backwards_from_a_symptom():
    found = causal.root_causes(_toy_graph(), ["SOC"], SITE, max_hops=2)
    drivers = {item["driver"] for item in found}
    assert "cover_crops" in drivers or "soil_moisture" in drivers
    assert all(item["path"][-1] == "SOC" for item in found)
