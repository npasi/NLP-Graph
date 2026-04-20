"""
build_graph.py — NetworkX graph assembly and export for one chapter.

Consumes a list of EdgeData objects (from extract_edges) and builds an
undirected weighted NetworkX graph.  Each node carries a mention_count
attribute; each edge carries weight, sentiment, polarity, and label.

Exports two artefacts:
  • outputs/baseline_graphs/{book_id}_ch{N:02d}.graphml  (full graph)
  • outputs/baseline_graphs/{book_id}_ch{N:02d}_summary.json (lightweight)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from src.baseline.extract_edges import EdgeData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polarity helper
# ---------------------------------------------------------------------------

def _polarity_label(sentiment: float) -> str:
    """Convert a VADER compound score to a discrete polarity string.

    Args:
        sentiment: VADER compound score in [-1.0, +1.0].

    Returns:
        ``"positive"`` if sentiment > 0.05,
        ``"negative"`` if sentiment < -0.05,
        ``"neutral"`` otherwise.
    """
    if sentiment > 0.05:
        return "positive"
    if sentiment < -0.05:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    edges: list[EdgeData],
    character_positions: dict[str, list[int]],
    book_id: str,
    chapter_num: int,
    output_dir: str | Path,
) -> nx.Graph:
    """Assemble a NetworkX graph from EdgeData and export to disk.

    Node attributes
    ---------------
    name           Canonical character name (same as the node label).
    mention_count  Number of token positions where this character appears.

    Edge attributes
    ---------------
    weight         Co-occurrence frequency (int).
    sentiment      Mean VADER compound score (float, -1 to +1).
    polarity       "positive" | "negative" | "neutral".
    label          Always "co-occurrence" for the baseline.

    Args:
        edges:               List of :class:`~extract_edges.EdgeData`.
        character_positions: Canonical name → token positions (for
                             computing mention_count node attribute).
        book_id:             Short book identifier.
        chapter_num:         1-based chapter index.
        output_dir:          Root output directory; files are written to
                             ``output_dir/{book_id}_ch{N:02d}.graphml``.

    Returns:
        The assembled :class:`networkx.Graph`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = nx.Graph()
    G.graph["book_id"]     = book_id
    G.graph["chapter"]     = chapter_num
    G.graph["edge_type"]   = "co-occurrence"

    # --- Add nodes ---
    for name, positions in character_positions.items():
        G.add_node(name, name=name, mention_count=len(positions))

    # --- Add edges ---
    if not edges:
        logger.warning(
            "No edges for %s chapter %d — graph will contain nodes only.",
            book_id, chapter_num,
        )
    else:
        for ed in edges:
            polarity = _polarity_label(ed.avg_sentiment)
            G.add_edge(
                ed.char_a,
                ed.char_b,
                weight=ed.weight,
                sentiment=ed.avg_sentiment,
                polarity=polarity,
                label="co-occurrence",
            )

    # --- Export GraphML ---
    stem = f"{book_id}_ch{chapter_num:02d}"
    graphml_path = output_dir / f"{stem}.graphml"
    nx.write_graphml(G, str(graphml_path))
    logger.info("GraphML written → %s", graphml_path)

    # --- Export JSON summary ---
    summary = _build_summary(G, book_id, chapter_num)
    json_path = output_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("JSON summary written → %s", json_path)

    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary(G: nx.Graph, book_id: str, chapter_num: int) -> dict:
    """Build a lightweight serialisable summary dict for a chapter graph.

    Args:
        G:           The assembled NetworkX graph.
        book_id:     Short book identifier.
        chapter_num: 1-based chapter index.

    Returns:
        Dictionary with keys: book_id, chapter, node_count, edge_count,
        density, avg_sentiment, top_characters (by mention_count).
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0

    sentiments = [d["sentiment"] for _, _, d in G.edges(data=True) if "sentiment" in d]
    avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else 0.0

    top_chars = sorted(
        [
            {"name": node, "mention_count": data.get("mention_count", 0)}
            for node, data in G.nodes(data=True)
        ],
        key=lambda x: x["mention_count"],
        reverse=True,
    )[:10]

    return {
        "book_id":      book_id,
        "chapter":      chapter_num,
        "node_count":   n,
        "edge_count":   e,
        "density":      round(density, 6),
        "avg_sentiment": avg_sentiment,
        "top_characters": top_chars,
    }
