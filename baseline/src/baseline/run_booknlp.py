"""
run_booknlp.py — NER and coreference resolution via BookNLP.

Wraps the BookNLP library (model "en_booknlp") to process a single
chapter .txt file.  Extracts PERSON-class character mentions, resolves
coreference clusters, and returns a canonical-name → token-position
mapping.  Outputs are cached to data/processed/{book_id}_ch{N}/ so
repeated runs do not re-invoke the model.
"""

import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_name(mentions: list[str]) -> str:
    """Choose the most-frequent surface form as the canonical character name.

    Args:
        mentions: All surface-form strings for a coreference cluster.

    Returns:
        The most common non-pronoun, title-bearing or plain-name mention.
        Falls back to the overall most-frequent form if no clear name found.
    """
    # Strip leading/trailing whitespace
    mentions = [m.strip() for m in mentions if m.strip()]
    if not mentions:
        return "UNKNOWN"

    # Prefer multi-token mentions (likely full names) over single pronouns
    pronoun_set = {
        "he", "she", "they", "him", "her", "them", "his", "hers",
        "their", "theirs", "it", "its", "i", "we", "us", "our",
        "you", "your", "me", "my",
    }
    named = [m for m in mentions if m.lower() not in pronoun_set]
    pool = named if named else mentions
    return Counter(pool).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_booknlp(
    chapter_path: str | Path,
    book_id: str,
    chapter_num: int,
    processed_dir: str | Path,
) -> dict[str, list[int]]:
    """Run BookNLP on a single chapter and return character token positions.

    If BookNLP output already exists in ``processed_dir`` the model is NOT
    re-run; cached results are loaded instead.

    Args:
        chapter_path:  Path to the chapter plain-text file.
        book_id:       Short book identifier (e.g. ``"pride_prejudice"``).
        chapter_num:   1-based chapter index (used for output sub-directory).
        processed_dir: Root directory for BookNLP cached outputs.

    Returns:
        Dictionary mapping each canonical character name (str) to a sorted
        list of token start-positions (int, 0-based) where that character
        is mentioned.

    Raises:
        FileNotFoundError: If ``chapter_path`` does not exist.
        ImportError: If the ``booknlp`` package is not installed.
        RuntimeError: If BookNLP processing fails unexpectedly.
    """
    chapter_path = Path(chapter_path)
    processed_dir = Path(processed_dir)

    if not chapter_path.exists():
        raise FileNotFoundError(f"Chapter file not found: {chapter_path}")

    # Output directory for this chapter's BookNLP artefacts
    output_subdir = processed_dir / f"{book_id}_ch{chapter_num:02d}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # BookNLP writes several files; the entities table is the key one.
    entities_file = output_subdir / f"{book_id}_ch{chapter_num:02d}.entities"
    tokens_file   = output_subdir / f"{book_id}_ch{chapter_num:02d}.tokens"

    # --- Run BookNLP if outputs are missing ---
    if not entities_file.exists() or not tokens_file.exists():
        logger.info("Running BookNLP on '%s' …", chapter_path.name)
        try:
            from booknlp.booknlp import BookNLP  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "booknlp is not installed.  Run: pip install booknlp"
            ) from exc

        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": "big",
        }
        booknlp_instance = BookNLP("en_booknlp", model_params)
        input_file = str(chapter_path)
        output_prefix = str(output_subdir / f"{book_id}_ch{chapter_num:02d}")

        try:
            booknlp_instance.process(input_file, output_prefix)
        except Exception as exc:
            raise RuntimeError(
                f"BookNLP failed on '{chapter_path}': {exc}"
            ) from exc
    else:
        logger.info(
            "Using cached BookNLP output for chapter %d in '%s'.",
            chapter_num,
            output_subdir,
        )

    # --- Parse the tokens file ---
    token_texts = _parse_tokens_file(tokens_file)

    # --- Parse entity / coreference data ---
    character_positions = _parse_entities_file(entities_file, token_texts)

    logger.info(
        "Chapter %d: found %d distinct characters.",
        chapter_num,
        len(character_positions),
    )
    return character_positions


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_tokens_file(tokens_file: Path) -> list[str]:
    """Load the BookNLP tokens TSV and return an ordered list of token texts.

    The BookNLP ``.tokens`` file is a tab-separated table with columns:
    ``token_id``, ``token``, ``POS``, ``fine_POS``, ``lemma``,
    ``chapter``, ``para_id``, ``sentence_id``, ``is_quote``,
    ``mention_start``, ``mention_end``, ``event``.

    Args:
        tokens_file: Path to the ``.tokens`` file produced by BookNLP.

    Returns:
        List of token strings in order (index == token_id).
    """
    tokens: list[str] = []
    with tokens_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("token_id"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tokens.append(parts[1])
    return tokens


def _parse_entities_file(
    entities_file: Path,
    token_texts: list[str],
) -> dict[str, list[int]]:
    """Build canonical-name → token-positions map from BookNLP entities file.

    The ``.entities`` file has columns:
    ``COREF``, ``start_token``, ``end_token``, ``text``, ``entity_type``.

    All mentions sharing the same ``COREF`` cluster id and tagged as
    ``PER`` (person) are grouped.  The most-frequent non-pronoun surface
    form becomes the canonical name.

    Args:
        entities_file: Path to the ``.entities`` file produced by BookNLP.
        token_texts:   Ordered token list (used to reconstruct spans when
                       the text column is empty).

    Returns:
        Dict mapping canonical name → sorted list of ``start_token`` ints.
    """
    # coref_id → {mentions: list[str], positions: list[int]}
    clusters: dict[int, dict] = {}

    with entities_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("COREF"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue

            coref_id_str, start_str, end_str, text, etype = (
                parts[0], parts[1], parts[2], parts[3], parts[4]
            )

            if etype.upper() not in {"PER", "PERSON"}:
                continue

            try:
                coref_id = int(coref_id_str)
                start    = int(start_str)
                end      = int(end_str)
            except ValueError:
                continue

            # Reconstruct surface form from token list if text column is empty
            if not text.strip() and token_texts:
                text = " ".join(token_texts[start : end + 1])

            if coref_id not in clusters:
                clusters[coref_id] = {"mentions": [], "positions": []}
            clusters[coref_id]["mentions"].append(text)
            clusters[coref_id]["positions"].append(start)

    # Build canonical name → positions
    character_positions: dict[str, list[int]] = {}
    for cluster in clusters.values():
        if not cluster["positions"]:
            continue
        name = _canonical_name(cluster["mentions"])
        if name in character_positions:
            character_positions[name].extend(cluster["positions"])
        else:
            character_positions[name] = list(cluster["positions"])

    # Sort positions
    for name in character_positions:
        character_positions[name].sort()

    return character_positions
