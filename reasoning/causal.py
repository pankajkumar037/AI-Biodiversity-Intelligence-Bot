"""Root-cause tracing, leverage ranking and signed effect propagation over the graph."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

import networkx as nx

import config
from reasoning import graph as graph_mod

# Which metric node a flag is about, and which way we want that metric to move.
FLAG_TARGET: dict[str, tuple[str, int]] = {
    "low_SOC": ("SOC", 1),
    "water_limited": ("soil_moisture", 1),
    "dry_subhumid": ("soil_moisture", 1),
    "monoculture": ("habitat_diversity", 1),
    "low_habitat_diversity": ("habitat_diversity", 1),
    "biodiversity_decline": ("species_richness", 1),
    "low_natural_cover": ("tree_cover", 1),
    "bare_fallow": ("erosion", -1),
    "acidic_soil": ("pH", 1),
    "alkaline_soil": ("pH", -1),
    "sodic_risk": ("salinity", -1),
}


def flag_targets(flags: list[str]) -> dict[str, int]:
    """node -> desired direction, for the flags that fired."""
    targets: dict[str, int] = {}
    for flag in flags:
        target = FLAG_TARGET.get(flag)
        if target:
            targets[target[0]] = target[1]
    return targets


def propagate(graph: nx.MultiDiGraph, practice: str, profile: dict[str, Any],
              max_hops: int = config.PROPAGATE_MAX_HOPS,
              decay: float = config.PROPAGATE_DECAY
              ) -> tuple[dict[str, float], list[dict]]:
    """Signed forward propagation from a practice node.

    Returns accumulated effects per node and the paths that produced them. A node
    is never revisited within one path, so effects cannot loop back on themselves.
    """
    effects: dict[str, float] = defaultdict(float)
    paths: list[dict] = []
    frontier = [(practice, 1.0, [practice])]

    for hop in range(max_hops):
        next_frontier = []
        for node, weight, path in frontier:
            for target, edge in graph_mod.out_edges(graph, node, profile):
                if target in path:
                    continue
                new_weight = weight * edge["sign"] * edge["weight"] * (decay ** hop)
                effects[target] += new_weight
                paths.append({
                    "path": path + [target],
                    "weight": round(new_weight, 4),
                    "effect": edge["effect"],
                    "condition": edge["condition"],
                    "docs": edge["docs"],
                    "mechanisms": edge["mechanisms"],
                })
                next_frontier.append((target, new_weight, path + [target]))
        frontier = next_frontier
        if not frontier:
            break

    return dict(effects), paths


def root_causes(graph: nx.MultiDiGraph, symptom_nodes: list[str], profile: dict[str, Any],
                max_hops: int = config.ROOT_CAUSE_MAX_HOPS) -> list[dict]:
    """Trace each symptom back to upstream drivers, keeping site-consistent paths."""
    found: list[dict] = []
    counter = 1

    for symptom in symptom_nodes:
        if symptom not in graph:
            continue
        frontier = [(symptom, [symptom])]
        for _ in range(max_hops):
            next_frontier = []
            for node, path in frontier:
                for source, edge in graph_mod.in_edges(graph, node, profile):
                    if source in path:
                        continue
                    chain = [source] + path
                    found.append({
                        "id": f"P{counter}",
                        "path": chain,
                        "symptom": symptom,
                        "driver": source,
                        "effect": edge["effect"],
                        "docs": edge["docs"],
                        "mechanisms": edge["mechanisms"],
                    })
                    counter += 1
                    next_frontier.append((source, chain))
            frontier = next_frontier
            if not frontier:
                break

    found.sort(key=lambda item: len(item["path"]))
    return found


@lru_cache(maxsize=1)
def _cycle_nodes() -> frozenset[str]:
    """Nodes that sit on a reinforcing loop, computed once per graph load."""
    graph = graph_mod.load_graph()
    simple = nx.DiGraph()
    for source, target, data in graph.edges(data=True):
        if data["traversable"] and data["sign"] != 0:
            simple.add_edge(source, target)
    nodes: set[str] = set()
    for cycle in nx.simple_cycles(simple, length_bound=4):
        nodes.update(cycle)
    return frozenset(nodes)


def leverage(graph: nx.MultiDiGraph, flags: list[str], profile: dict[str, Any],
             top_n: int = 10) -> list[dict]:
    """Rank nodes by how much of the diagnosed problem they reach.

    A node upstream of several flagged metrics outranks one that fixes a single
    symptom, which is what makes the resulting priority non-obvious but defensible.
    """
    targets = flag_targets(flags)
    if not targets:
        return []

    on_cycle = _cycle_nodes()
    scored: list[dict] = []

    for node in graph.nodes():
        if not graph.out_degree(node):
            continue
        effects, _ = propagate(graph, node, profile)
        reached = {
            target: effects[target] * desired
            for target, desired in targets.items()
            if target in effects
        }
        helpful = {name: value for name, value in reached.items() if value > 0}
        if not helpful:
            continue
        score = sum(helpful.values()) * (1 + 0.25 * (len(helpful) - 1))
        if node in on_cycle:
            score *= 1.15
        scored.append({
            "node": node,
            "score": round(score, 4),
            "reaches": sorted(helpful),
            "on_loop": node in on_cycle,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_n]


def benefit_for_flags(effects: dict[str, float], flags: list[str]) -> float:
    """How far a practice's effects move the flagged metrics the right way."""
    targets = flag_targets(flags)
    return sum(effects.get(node, 0.0) * desired for node, desired in targets.items())
