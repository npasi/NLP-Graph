"""
chapter_splitter.py — Splits a full book .txt into per-chapter files.

Identical to baseline/src/utils/chapter_splitter.py.
Copied unchanged; baseline_injected uses the same chapter-splitting logic.

Detects chapter boundaries using regex patterns common in Project
Gutenberg plain-text editions (e.g. "CHAPTER I", "Chapter 1",
"CHAPTER ONE"). Saves each chapter as data/raw/{book_id}_ch{N:02d}.txt.
Preamble before chapter 1 and appendices/backmatter are discarded.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chapter-heading detection patterns
# ---------------------------------------------------------------------------
_ORDINAL_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|"
    "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    "eighteen|nineteen|twenty|twenty[-\\s]one|twenty[-\\s]two|"
    "twenty[-\\s]three|twenty[-\\s]four|twenty[-\\s]five|"
    "twenty[-\\s]six|twenty[-\\s]seven|twenty[-\\s]eight|"
    "twenty[-\\s]nine|thirty|thirty[-\\s]one|thirty[-\\s]two|"
    "forty|fifty|sixty|seventy|eighty|ninety|one hundred"
)

_ROMAN = r"M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"

_CHAPTER_RE = re.compile(
    r"^\s*(?:chapter|chap\.?)\s+"          # keyword
    r"(?:"
    r"\d{1,3}"                              # Arabic numeral
    r"|" + _ROMAN + r"(?!\w)"              # Roman numeral (not mid-word)
    r"|(?:" + _ORDINAL_WORDS + r")"        # ordinal word
    r")"
    r"[\s.:\-–—]*"                         # optional separator
    r"(?:.*)?$",                           # optional title text
    re.IGNORECASE | re.MULTILINE,
)


def _find_chapter_starts(text: str) -> list[tuple[int, int]]:
    """Return (match_start, match_end) pairs for each chapter heading.

    Args:
        text: Full book text string.

    Returns:
        Sorted list of (start, end) character offsets for chapter headings.
    """
    return [(m.start(), m.end()) for m in _CHAPTER_RE.finditer(text)]


def split_book_into_chapters(
    book_path: str | Path,
    book_id: str,
    output_dir: str | Path,
) -> list[Path]:
    """Split a full-book .txt file into individual chapter files.

    Detects chapter headings with a multi-pattern regex, extracts the
    text between consecutive headings, and writes each chapter to
    ``output_dir/{book_id}_ch{N:02d}.txt``.  The preamble before the
    first detected chapter and any backmatter after the last chapter
    are silently discarded.

    Args:
        book_path: Path to the full-book plain-text file.
        book_id:   Short identifier used as a filename prefix
                   (e.g. ``"pride_prejudice"``).
        output_dir: Directory where per-chapter files are written.
                    Created if it does not exist.

    Returns:
        Ordered list of :class:`~pathlib.Path` objects for every chapter
        file that was written (empty chapters are skipped).

    Raises:
        FileNotFoundError: If ``book_path`` does not exist.
        ValueError: If no chapter headings are detected in the file.
    """
    book_path = Path(book_path)
    output_dir = Path(output_dir)

    if not book_path.exists():
        raise FileNotFoundError(f"Book file not found: {book_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    text = book_path.read_text(encoding="utf-8", errors="replace")

    from src.utils.text_cleaner import strip_gutenberg_header_footer  # local import
    text = strip_gutenberg_header_footer(text)

    chapter_starts = _find_chapter_starts(text)

    if not chapter_starts:
        raise ValueError(
            f"No chapter headings detected in '{book_path}'. "
            "Check that the file uses a supported heading format."
        )

    logger.info(
        "Detected %d chapter headings in '%s'.", len(chapter_starts), book_path.name
    )

    written: list[Path] = []
    for idx, (start, _end) in enumerate(chapter_starts, start=1):
        if idx < len(chapter_starts):
            next_start = chapter_starts[idx][0]
            chapter_text = text[start:next_start].strip()
        else:
            chapter_text = text[start:].strip()

        if not chapter_text:
            logger.warning("Chapter %d is empty — skipping.", idx)
            continue

        out_path = output_dir / f"{book_id}_ch{idx:02d}.txt"
        out_path.write_text(chapter_text, encoding="utf-8")
        written.append(out_path)
        logger.debug("Wrote chapter %d → %s (%d chars)", idx, out_path, len(chapter_text))

    logger.info("Wrote %d chapter files to '%s'.", len(written), output_dir)
    return written
