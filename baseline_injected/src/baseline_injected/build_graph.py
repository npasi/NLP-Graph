"""
build_graph.py — NetworkX graph assembly with Markov conditioning.

Differs from baseline/src/baseline/build_graph.py in the following ways:
  - Accepts optional ``G_prev`` (networkx.Graph) and ``alpha`` (float)
    parameters implementing the Markov update formula.
  - When ``G_prev`` is None: behaves exactly like the baseline.
  - When ``G_prev`` is provided, applies per-edge:

      new_weight     = α * G_prev[A,B].weight     + (1-α) * new_cooccurrences(k)
      new_sentiment  = α * G_prev[A,B].sentiment   + (1-α) * vader_score(k)

    Three cases are handled automatically:
      • Edge in G_prev only    → survives with weight = α * prev_weight
                                  (dropped if weight < min_weight_threshold)
      • Edge in chapter k only → born with weight = (1-α) * new_cooccurrences
      • Edge in both           → reinforced via full formula

  - Edges carry additional attributes: ``label="co-occurrence-markov"``
    and ``is_carry=True/False`` (True when the edge was inherited from
    G_prev with no new co-occurrence in chapter k).
  - JSON summary includes ``carried_edges`` count and ``alpha`` value.

Exports two artefacts per chapter:
  • outputs/baseline_injected_graphs/{book_id}_ch{N:02d}.graphml
  • outputs/baseline_injected_graphs/{book_id}_ch{N:02d}_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from src.baseline_injected.extract_edges import EdgeData

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
    G_prev: nx.Graph | None = None,
    alpha: float = 0.7,
    min_weight_threshold: float = 0.5,
) -> nx.Graph:
    """Assemble a NetworkX graph from EdgeData, applying Markov update if G_prev given.

    When ``G_prev`` is ``None`` this function behaves identically to the
    baseline's :func:`build_graph`.  When ``G_prev`` is provided every edge
    is computed via the Markov formula before being added to the graph.

    Node attributes
    ---------------
    name           Canonical character name (same as the node label).
    mention_count  Number of token positions where this character appears
                   in the current chapter (0 for carry-over-only characters).

    Edge attributes
    ---------------
    weight         Markov-updated co-occurrence frequency (float).
    sentiment      Markov-updated mean VADER compound score (float, -1 to +1).
    polarity       "positive" | "negative" | "neutral".
    label          "co-occurrence" (baseline) or "co-occurrence-markov".
    is_carry       True if the edge is carried from G_prev with no new
                   co-occurrence in chapter k (only present when G_prev given).

    Args:
        edges:                List of :class:`~extract_edges.EdgeData` from
                              the current chapter.
        character_positions:  Canonical name → token positions for the current
                              chapter (for computing mention_count).
        book_id:              Short book identifier.
        chapter_num:          1-based chapter index.
        output_dir:           Root output directory.
        G_prev:               Previous-chapter graph G_{k-1}; ``None`` for
                              chapter 1 (produces baseline behaviour).
        alpha:                Decay factor in [0.0, 1.0].  Higher values
                              favour prior graph; lower values favour the
                              current chapter.
        min_weight_threshold: Carry-over edges below this threshold are pruned
                              (default 0.5).

    Returns:
        The assembled :class:`networkx.Graph` for chapter k.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = nx.Graph()
    G.graph["book_id"]   = book_id
    G.graph["chapter"]   = chapter_num
    G.graph["alpha"]     = alpha
    G.graph["edge_type"] = "co-occurrence-markov" if G_prev is not None else "co-occurrence"

    if G_prev is None:
        # ---------------------------------------------------------------
        # Baseline mode: identical behaviour to baseline/build_graph.py
        # ---------------------------------------------------------------
        G.graph["edge_type"] = "co-occurrence"

        for name, positions in character_positions.items():
            G.add_node(name, name=name, mention_count=len(positions))

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
    else:
        # ---------------------------------------------------------------
        # Markov mode
        # ---------------------------------------------------------------
        # Build lookup: frozenset({a,b}) → EdgeData for this chapter
        new_edge_map: dict[frozenset, EdgeData] = {
            frozenset({ed.char_a, ed.char_b}): ed for ed in edges
        }

        # Build lookup: frozenset({a,b}) → (weight, sentiment) from G_prev
        prev_edge_map: dict[frozenset, tuple[float, float]] = {
            frozenset({u, v}): (
                float(data.get("weight", 0)),
                float(data.get("sentiment", 0.0)),
            )
            for u, v, data in G_prev.edges(data=True)
        }

        # Add nodes: characters from current chapter + carry-over characters
        for name, positions in character_positions.items():
            G.add_node(name, name=name, mention_count=len(positions))

        # Add carry-over characters from G_prev that may appear on carried edges
        prev_chars_with_edges: set[str] = set()
        for key in prev_edge_map:
            if key not in new_edge_map:
                # Potential carry edge — collect its nodes
                chars = list(key)
                prev_chars_with_edges.update(chars)
        for char in prev_chars_with_edges:
            if char not in G:
                G.add_node(char, name=char, mention_count=0)

        # ---- Edges present in chapter k --------------------------------
        for key, ed in new_edge_map.items():
            chars = sorted(list(key))
            if len(chars) != 2:
                continue
            char_a, char_b = chars

            prev_weight, prev_sentiment = prev_edge_map.get(key, (0.0, 0.0))
            new_weight    = alpha * prev_weight    + (1.0 - alpha) * ed.weight
            new_sentiment = alpha * prev_sentiment + (1.0 - alpha) * ed.avg_sentiment

            polarity = _polarity_label(new_sentiment)
            G.add_edge(
                char_a, char_b,
                weight=round(new_weight, 4),
                sentiment=round(new_sentiment, 4),
                polarity=polarity,
                label="co-occurrence-markov",
                is_carry=False,
            )

        # ---- Carry-over edges: in G_prev but absent in chapter k --------
        for key, (prev_weight, prev_sentiment) in prev_edge_map.items():
            if key in new_edge_map:
                continue  # already handled above

            decayed_weight = alpha * prev_weight
            if decayed_weight < min_weight_threshold:
                continue

            decayed_sentiment = alpha * prev_sentiment
            chars = sorted(list(key))
            if len(chars) != 2:
                continue
            char_a, char_b = chars

            polarity = _polarity_label(decayed_sentiment)
            G.add_edge(
                char_a, char_b,
                weight=round(decayed_weight, 4),
                sentiment=round(decayed_sentiment, 4),
                polarity=polarity,
                label="co-occurrence-markov",
                is_carry=True,
            )

    # --- Export GraphML ---
    stem = f"{book_id}_ch{chapter_num:02d}"
    graphml_path = output_dir / f"{stem}.graphml"
    nx.write_graphml(G, str(graphml_path))
    logger.info("GraphML written → %s", graphml_path)

    # --- Export JSON summary ---
    summary = _build_summary(G, book_id, chapter_num, alpha)
    json_path = output_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("JSON summary written → %s", json_path)

    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary(
    G: nx.Graph,
    book_id: str,
    chapter_num: int,
    alpha: float,
) -> dict:
    """Build a lightweight serialisable summary dict for a chapter graph.

    Args:
        G:           The assembled NetworkX graph.
        book_id:     Short book identifier.
        chapter_num: 1-based chapter index.
        alpha:       Decay factor used for this chapter.

    Returns:
        Dictionary with keys: book_id, chapter, node_count, edge_count,
        carried_edges, density, avg_sentiment, alpha, top_characters.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0

    sentiments = [d["sentiment"] for _, _, d in G.edges(data=True) if "sentiment" in d]
    avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else 0.0

    carried_edges = sum(
        1 for _, _, d in G.edges(data=True) if d.get("is_carry", False)
    )

    top_chars = sorted(
        [
            {"name": node, "mention_count": data.get("mention_count", 0)}
            for node, data in G.nodes(data=True)
        ],
        key=lambda x: x["mention_count"],
        reverse=True,
    )[:10]

    return {
        "book_id":        book_id,
        "chapter":        chapter_num,
        "node_count":     n,
        "edge_count":     e,
        "carried_edges":  carried_edges,
        "density":        round(density, 6),
        "avg_sentiment":  avg_sentiment,
        "alpha":          alpha,
        "top_characters": top_chars,
    }
