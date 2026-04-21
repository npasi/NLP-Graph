"""
graph_metrics.py — Evaluation metrics for character interaction graphs.

Identical to baseline/src/evaluation/graph_metrics.py.
Copied unchanged; baseline_injected uses the same evaluation metrics so
that outputs are directly comparable with the baseline.

Implements edge-level and graph-level comparison metrics used to
benchmark predicted graphs against gold-standard annotations:

  • precision_recall_f1  — standard IR metrics over edge sets
  • jaccard_index        — set overlap similarity
  • graph_edit_distance  — structural similarity (networkx GED with timeout)
  • polarity_agreement   — fraction of matched edges with the same polarity
"""

from __future__ import annotations

import logging
from typing import Set

import networkx as nx

logger = logging.getLogger(__name__)


def precision_recall_f1(
    pred_edges: Set[frozenset],
    gold_edges: Set[frozenset],
) -> dict[str, float]:
    """Compute precision, recall, and F1 over two edge sets.

    Edges are represented as ``frozenset({char_a, char_b})`` so that
    direction is ignored.  An edge is a true positive if and only if
    the exact same character pair appears in both sets.

    Args:
        pred_edges: Set of frozensets representing predicted edges.
        gold_edges: Set of frozensets representing gold-standard edges.

    Returns:
        Dictionary with keys ``"precision"``, ``"recall"``, ``"f1"``,
        each a float in [0.0, 1.0].  All three are 0.0 when both sets
        are empty; if only one set is empty, precision or recall is 0.0.
    """
    if not pred_edges and not gold_edges:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_edges & gold_edges)
    precision = tp / len(pred_edges) if pred_edges else 0.0
    recall    = tp / len(gold_edges) if gold_edges else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def jaccard_index(
    pred_edges: Set[frozenset],
    gold_edges: Set[frozenset],
) -> float:
    """Compute the Jaccard similarity coefficient between two edge sets.

    Jaccard = |intersection| / |union|.  Returns 1.0 when both sets are
    empty (vacuously equal), and 0.0 when the union is non-empty but the
    intersection is empty.

    Args:
        pred_edges: Set of frozensets representing predicted edges.
        gold_edges: Set of frozensets representing gold-standard edges.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    if not pred_edges and not gold_edges:
        return 1.0

    intersection = len(pred_edges & gold_edges)
    union        = len(pred_edges | gold_edges)

    return round(intersection / union, 4) if union > 0 else 0.0


def graph_edit_distance(
    G1: nx.Graph,
    G2: nx.Graph,
    timeout: int = 30,
) -> float:
    """Estimate the Graph Edit Distance between two NetworkX graphs.

    Uses :func:`networkx.graph_edit_distance` with an upper-bound
    ``timeout`` to avoid exponential worst-case runtimes on large graphs.
    If the exact GED is not found within the timeout, the function returns
    the best upper-bound found so far (which may be an overestimate).

    Args:
        G1:      First graph (predicted).
        G2:      Second graph (gold-standard).
        timeout: Wall-clock seconds allowed for the computation (default 30).

    Returns:
        GED estimate as a non-negative float.  Returns ``float("inf")``
        if networkx raises an unexpected error.
    """
    try:
        ged = nx.graph_edit_distance(G1, G2, timeout=timeout)
        if ged is None:
            ub_gen = nx.optimize_graph_edit_distance(G1, G2)
            ged = next(ub_gen)
        return round(float(ged), 4)
    except Exception as exc:
        logger.error("GED computation failed: %s", exc)
        return float("inf")


def polarity_agreement(
    pred_graph: nx.Graph,
    gold_graph: nx.Graph,
) -> float:
    """Compute the fraction of matched edges with identical polarity labels.

    An edge is "matched" if the same character pair exists in both graphs.
    For each matched edge, polarity agreement is 1 if the ``polarity``
    edge attribute is identical in both graphs, 0 otherwise.

    Args:
        pred_graph: Predicted graph; edges must have a ``"polarity"``
                    attribute (``"positive"`` / ``"negative"`` / ``"neutral"``).
        gold_graph: Gold-standard graph with the same attribute convention.

    Returns:
        Agreement rate in [0.0, 1.0].  Returns 0.0 when there are no
        matched edges.
    """
    pred_edge_map: dict[frozenset, str] = {
        frozenset({u, v}): data.get("polarity", "neutral")
        for u, v, data in pred_graph.edges(data=True)
    }
    gold_edge_map: dict[frozenset, str] = {
        frozenset({u, v}): data.get("polarity", "neutral")
        for u, v, data in gold_graph.edges(data=True)
    }

    matched_keys = set(pred_edge_map) & set(gold_edge_map)
    if not matched_keys:
        logger.warning("No matched edges for polarity agreement computation.")
        return 0.0

    agreements = sum(
        1 for k in matched_keys
        if pred_edge_map[k] == gold_edge_map[k]
    )
    return round(agreements / len(matched_keys), 4)


def edges_to_frozensets(graph: nx.Graph) -> Set[frozenset]:
    """Convert a graph's edge list to a set of frozensets.

    Convenience helper for passing a NetworkX graph to
    :func:`precision_recall_f1` or :func:`jaccard_index`.

    Args:
        graph: Any NetworkX graph.

    Returns:
        Set of ``frozenset({u, v})`` for each edge ``(u, v)`` in the graph.
    """
    return {frozenset({u, v}) for u, v in graph.edges()}
