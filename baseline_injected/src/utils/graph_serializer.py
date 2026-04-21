"""
graph_serializer.py — Serialize a NetworkX graph to a human-readable string.

New module with no equivalent in baseline/.  Its primary purpose is to
produce a snapshot of G_{k-1} that is logged before each chapter is
processed in the Markov-conditioned pipeline.

Three serialization strategies are provided:

  flat  (default)
      Per-character adjacency list with inline weight and sentiment:
      "Elizabeth: Jane(+0.8,w=38), Darcy(+0.2,w=47)"

  edges
      One line per edge with weight and polarity:
      "Elizabeth--Jane: weight=38, polarity=positive"

  topN
      Same as "edges" but limited to the N highest-weight edges.
      When ``chapter_chars`` is supplied, edges connecting at least
      one character absent from the current chapter are omitted first.

Design note
-----------
The serialized string is used ONLY for logging and human inspection —
it documents the state of G_{k-1} before the Markov update is applied.
It is NOT fed into BookNLP, VADER, or any neural model in this module.

The module is designed so that future modules (e.g. M1+graph) can import
:func:`serialize` and pass the string as context to a BERT-style encoder.
"""

from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)

# Polarity score thresholds — mirror the convention used in build_graph.py
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05


def serialize(
    G: nx.Graph,
    strategy: str = "flat",
    top_n: int = 20,
    chapter_chars: set[str] | None = None,
) -> str:
    """Serialize a NetworkX graph to a human-readable string.

    Args:
        G:            Graph to serialize (typically G_{k-1}).
        strategy:     One of ``"flat"``, ``"edges"``, or ``"topN"``.
        top_n:        Maximum number of edges for the ``"topN"`` strategy.
        chapter_chars: Set of character names that appear in the current
                       chapter.  When provided with ``strategy="topN"``,
                       edges where *neither* endpoint appears in the chapter
                       are filtered out before selecting the top N.

    Returns:
        Multi-line string representation of the graph.  Returns an empty
        string if the graph has no nodes or no edges.

    Raises:
        ValueError: If ``strategy`` is not one of the three supported values.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return ""

    if strategy == "flat":
        return _serialize_flat(G)
    if strategy == "edges":
        return _serialize_edges(G)
    if strategy == "topN":
        return _serialize_topn(G, top_n=top_n, chapter_chars=chapter_chars)

    raise ValueError(
        f"Unknown serialization strategy {strategy!r}. "
        "Choose from: 'flat', 'edges', 'topN'."
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _serialize_flat(G: nx.Graph) -> str:
    """Serialize as per-character adjacency lists.

    Format per line:
      "<CharA>: <CharB>(<sign><sentiment>,w=<weight>), ..."

    Neighbours are sorted by descending edge weight.  Characters with no
    edges after filtering are omitted.

    Args:
        G: Graph to serialize.

    Returns:
        Newline-joined adjacency-list string.
    """
    lines: list[str] = []
    for node in sorted(G.nodes()):
        neighbours = []
        for nbr in G.neighbors(node):
            data = G[node][nbr]
            weight    = data.get("weight", 0)
            sentiment = data.get("sentiment", 0.0)
            sign = "+" if sentiment >= 0 else ""
            neighbours.append((weight, f"{nbr}({sign}{sentiment:.1f},w={weight:.0f})"))

        if not neighbours:
            continue

        # Sort by weight descending
        neighbours.sort(key=lambda x: x[0], reverse=True)
        neighbour_str = ", ".join(nb for _, nb in neighbours)
        lines.append(f"{node}: {neighbour_str}")

    return "\n".join(lines)


def _serialize_edges(G: nx.Graph) -> str:
    """Serialize as one line per edge with weight and polarity.

    Format per line:
      "<CharA>--<CharB>: weight=<w>, polarity=<p>"

    Edges are sorted by descending weight.

    Args:
        G: Graph to serialize.

    Returns:
        Newline-joined edge-list string.
    """
    edge_rows: list[tuple[float, str]] = []
    for u, v, data in G.edges(data=True):
        weight   = data.get("weight", 0)
        polarity = data.get("polarity", _polarity_label(data.get("sentiment", 0.0)))
        a, b = sorted([u, v])
        edge_rows.append((float(weight), f"{a}--{b}: weight={weight:.4g}, polarity={polarity}"))

    edge_rows.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(row for _, row in edge_rows)


def _serialize_topn(
    G: nx.Graph,
    top_n: int = 20,
    chapter_chars: set[str] | None = None,
) -> str:
    """Serialize the top-N edges by weight, optionally filtered to chapter chars.

    When ``chapter_chars`` is provided, an edge is eligible only if at
    least one of its endpoints is present in ``chapter_chars``.  This
    ensures the serialized context is relevant to the current chapter.

    Format per line (same as ``"edges"`` strategy):
      "<CharA>--<CharB>: weight=<w>, polarity=<p>"

    Args:
        G:            Graph to serialize.
        top_n:        Maximum number of edges to include.
        chapter_chars: Characters appearing in the current chapter.
                       ``None`` disables filtering.

    Returns:
        Newline-joined string of up to ``top_n`` edges.
    """
    edge_rows: list[tuple[float, str]] = []
    for u, v, data in G.edges(data=True):
        # Optional: filter to edges with at least one chapter character
        if chapter_chars is not None:
            if u not in chapter_chars and v not in chapter_chars:
                continue

        weight   = data.get("weight", 0)
        polarity = data.get("polarity", _polarity_label(data.get("sentiment", 0.0)))
        a, b = sorted([u, v])
        edge_rows.append((float(weight), f"{a}--{b}: weight={weight:.4g}, polarity={polarity}"))

    edge_rows.sort(key=lambda x: x[0], reverse=True)
    top = edge_rows[:top_n]

    if not top:
        return ""

    return "\n".join(row for _, row in top)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _polarity_label(sentiment: float) -> str:
    """Convert a VADER compound score to a discrete polarity string.

    Args:
        sentiment: VADER compound score in [-1.0, +1.0].

    Returns:
        ``"positive"``, ``"negative"``, or ``"neutral"``.
    """
    if sentiment > _POS_THRESHOLD:
        return "positive"
    if sentiment < _NEG_THRESHOLD:
        return "negative"
    return "neutral"
