"""Order practices by their prerequisites and map the order onto time horizons."""
from __future__ import annotations

import networkx as nx

from core.schemas import PracticeCard
from knowledge import loader

HORIZON_RANK = {"short": 0, "medium": 1, "long": 2}
HORIZON_LABEL = {0: "short", 1: "medium", 2: "long"}


def sequence(practice_ids: list[str],
             cards: dict[str, PracticeCard] | None = None) -> list[dict]:
    """Topologically sort the chosen practices, pulling in missing prerequisites.

    A practice never appears before something it requires, and its horizon is
    pushed out if a prerequisite would otherwise land at the same time.
    """
    catalogue = cards if cards is not None else loader.practice_cards()
    selected = [pid for pid in practice_ids if pid in catalogue]

    included: list[str] = []
    for practice_id in selected:
        for prerequisite in catalogue[practice_id].requires_first:
            if prerequisite in catalogue and prerequisite not in selected:
                if prerequisite not in included:
                    included.append(prerequisite)
    ordered_ids = included + [pid for pid in selected if pid not in included]

    order_graph = nx.DiGraph()
    order_graph.add_nodes_from(ordered_ids)
    for practice_id in ordered_ids:
        for prerequisite in catalogue[practice_id].requires_first:
            if prerequisite in order_graph:
                order_graph.add_edge(prerequisite, practice_id)

    try:
        ordering = list(nx.topological_sort(order_graph))
    except nx.NetworkXUnfeasible:
        # NOTE: a prerequisite cycle in hand-written cards is a data error, not a
        # runtime condition. Fall back to the given order rather than failing a turn.
        ordering = ordered_ids

    horizons: dict[str, int] = {}
    for practice_id in ordering:
        card = catalogue[practice_id]
        rank = HORIZON_RANK.get(card.time_horizon, 1)
        for prerequisite in card.requires_first:
            if prerequisite in horizons:
                rank = max(rank, horizons[prerequisite] + 1)
        horizons[practice_id] = min(rank, 2)

    return [
        {
            "practice_id": practice_id,
            "name": catalogue[practice_id].name,
            "order": index + 1,
            "time_horizon": HORIZON_LABEL[horizons[practice_id]],
            "card_time_horizon": catalogue[practice_id].time_horizon,
            "requires_first": catalogue[practice_id].requires_first,
            "added_as_prerequisite": practice_id in included,
        }
        for index, practice_id in enumerate(ordering)
    ]
