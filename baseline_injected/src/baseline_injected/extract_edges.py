"""
extract_edges.py — Co-occurrence edge extraction with VADER sentiment.

Differs from baseline/src/baseline/extract_edges.py in one way:
  - Accepts an optional ``G_prev`` (networkx.Graph) parameter.
  - When ``G_prev`` is not None, logs the character overlap between the
    current chapter and the previous graph for transparency and auditing.
  - The return type is always ``list[EdgeData]`` (new co-occurrences only,
    same as baseline).  The Markov combination with G_prev is handled
    exclusively in build_graph.py, keeping responsibilities cleanly separated.

Given a flat token list and a character→token-positions map (both
produced by run_booknlp), this module:

  1. Slides a window of W tokens over the chapter.
  2. Records every pair of distinct characters that appear within the
     same window as a co-occurrence event.
  3. Discards pairs whose total count falls below ``min_freq``.
  4. Runs vaderSentiment on the context window of each co-occurrence
     event and averages compound scores into a final edge sentiment.
  5. Returns a list of EdgeData named-tuples ready for graph assembly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EdgeData:
    """Represents a weighted, sentiment-scored edge between two characters.

    Attributes:
        char_a:        Canonical name of the first character.
        char_b:        Canonical name of the second character.
        weight:        Number of co-occurrence events (after thresholding).
        avg_sentiment: Mean VADER compound score across all co-occurrence
                       windows. Range [-1.0, +1.0].
        raw_scores:    Individual compound scores (not serialised to graph).
    """

    char_a: str
    char_b: str
    weight: int
    avg_sentiment: float
    raw_scores: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# VADER wrapper
# ---------------------------------------------------------------------------

def _get_vader_analyzer():
    """Lazy-load the VADER SentimentIntensityAnalyzer.

    Returns:
        SentimentIntensityAnalyzer instance.

    Raises:
        ImportError: If vaderSentiment is not installed.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment is not installed.  Run: pip install vaderSentiment"
        ) from exc
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _character_presence_map(
    character_positions: dict[str, list[int]],
    num_tokens: int,
) -> dict[int, list[str]]:
    """Build a token-index → [character names] lookup.

    Inverts the character→positions dict for O(1) lookups during
    the sliding-window scan.

    Args:
        character_positions: Canonical name → sorted list of token positions.
        num_tokens:          Total token count in the chapter.

    Returns:
        Dict mapping each token index to a list of character names
        that are mentioned at that position.
    """
    presence: dict[int, list[str]] = defaultdict(list)
    for name, positions in character_positions.items():
        for pos in positions:
            if 0 <= pos < num_tokens:
                presence[pos].append(name)
    return dict(presence)


def _window_generator(
    presence: dict[int, list[str]],
    num_tokens: int,
    window: int,
) -> Generator[tuple[int, set[str]], None, None]:
    """Yield (window_center, set_of_characters_in_window) for each token.

    Slides a window of size ``window`` tokens centred on each token that
    has at least one character mention.  Only yields windows that
    contain ≥ 2 distinct characters.

    Args:
        presence:   Token-index → character-names lookup.
        num_tokens: Total number of tokens.
        window:     Half-window radius in tokens; the full window spans
                    [center - window//2, center + window//2].

    Yields:
        (center_index, set_of_character_names) for qualifying windows.
    """
    half = window // 2
    for center in presence:
        lo = max(0, center - half)
        hi = min(num_tokens, center + half + 1)
        chars_in_window: set[str] = set()
        for idx in range(lo, hi):
            if idx in presence:
                chars_in_window.update(presence[idx])
        if len(chars_in_window) >= 2:
            yield center, chars_in_window


def extract_edges(
    tokens: list[str],
    character_positions: dict[str, list[int]],
    window: int = 100,
    min_freq: int = 3,
    G_prev: "nx.Graph | None" = None,
) -> list[EdgeData]:
    """Extract co-occurrence edges with VADER sentiment from a token list.

    Behaves identically to the baseline when ``G_prev`` is ``None``.
    When ``G_prev`` is provided, logs the character overlap between the
    current chapter and the previous graph.  The Markov combination with
    G_prev is handled exclusively in :func:`~build_graph.build_graph`.

    Algorithm
    ---------
    1. Build an inverted token→character lookup.
    2. For every token that is a character mention, gather all characters
       within a ±window/2 radius.
    3. Emit an event for every pair in that set.
    4. After scanning, discard pairs with total events < min_freq.
    5. For each surviving pair, average the VADER compound score of
       each co-occurrence window.

    Args:
        tokens:              Ordered list of token strings for the chapter.
        character_positions: Canonical name → sorted token-position list.
        window:              Co-occurrence window width in tokens (default 100).
        min_freq:            Minimum co-occurrence count to keep an edge
                             (default 3).
        G_prev:              Previous-chapter graph G_{k-1}.  Accepted to
                             satisfy the Markov-pipeline API contract and to
                             log character continuity; not used in extraction.

    Returns:
        List of :class:`EdgeData` instances, one per character pair that
        survived the frequency threshold.  Returns an empty list if
        fewer than 2 characters are detected.
    """
    if len(character_positions) < 2:
        logger.warning("Fewer than 2 characters found; no edges to extract.")
        return []

    num_tokens = len(tokens)
    if num_tokens == 0:
        logger.warning("Token list is empty; skipping edge extraction.")
        return []

    if G_prev is not None and G_prev.number_of_nodes() > 0:
        prev_chars = set(G_prev.nodes())
        curr_chars = set(character_positions.keys())
        overlap   = prev_chars & curr_chars
        new_chars = curr_chars - prev_chars
        lost_chars = prev_chars - curr_chars
        logger.info(
            "G_prev overlap: %d shared chars, %d new chars, %d absent chars.",
            len(overlap), len(new_chars), len(lost_chars),
        )

    logger.info(
        "Extracting edges: %d tokens, %d characters, window=%d, min_freq=%d",
        num_tokens, len(character_positions), window, min_freq,
    )

    analyzer = _get_vader_analyzer()
    presence = _character_presence_map(character_positions, num_tokens)

    # pair_key → list of compound sentiment scores (one per co-occurrence event)
    pair_scores: dict[frozenset, list[float]] = defaultdict(list)

    half = window // 2
    for center, chars_in_window in _window_generator(presence, num_tokens, window):
        lo = max(0, center - half)
        hi = min(num_tokens, center + half + 1)
        context = " ".join(tokens[lo:hi])
        compound = analyzer.polarity_scores(context)["compound"]

        for char_a, char_b in combinations(sorted(chars_in_window), 2):
            pair_key: frozenset = frozenset({char_a, char_b})
            pair_scores[pair_key].append(compound)

    edges: list[EdgeData] = []
    for pair_key, scores in pair_scores.items():
        if len(scores) < min_freq:
            continue
        chars = list(pair_key)
        char_a, char_b = (chars[0], chars[1]) if len(chars) == 2 else (chars[0], chars[0])
        avg_sentiment = sum(scores) / len(scores)
        edges.append(
            EdgeData(
                char_a=char_a,
                char_b=char_b,
                weight=len(scores),
                avg_sentiment=round(avg_sentiment, 4),
                raw_scores=scores,
            )
        )

    logger.info(
        "Edge extraction complete: %d pairs survived min_freq=%d.",
        len(edges), min_freq,
    )
    return edges
