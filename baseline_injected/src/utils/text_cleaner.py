"""
text_cleaner.py — Preprocessing utilities for raw book text.

Identical to baseline/src/utils/text_cleaner.py.
Copied unchanged; baseline_injected uses the same text-cleaning logic.

Provides functions to normalize whitespace, strip Gutenberg boilerplate,
remove non-printable characters, and tokenize text into a flat token list
suitable for downstream co-occurrence analysis.
"""

import re
import unicodedata
from typing import Generator


_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG",
    re.IGNORECASE,
)

# Sentence-ending punctuation kept as tokens so VADER windows stay coherent.
_TOKENIZE_PATTERN = re.compile(r"[A-Za-z0-9''\-]+|[.,!?;:\"()\[\]]")


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate.

    Args:
        text: Raw full-book string, possibly including Gutenberg metadata.

    Returns:
        The book text with header/footer removed. If markers are not found
        the original text is returned unchanged.
    """
    start_match = _GUTENBERG_START.search(text)
    end_match = _GUTENBERG_END.search(text)

    start = 0
    end = len(text)

    if start_match:
        start = text.index("\n", start_match.end()) + 1
    if end_match:
        end = end_match.start()

    return text[start:end]


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing space.

    Args:
        text: Input string.

    Returns:
        String with all whitespace sequences replaced by a single space.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_non_printable(text: str) -> str:
    """Remove non-printable Unicode control characters.

    Args:
        text: Input string.

    Returns:
        String with control characters (category Cc/Cs) removed,
        preserving newlines and tabs.
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cs"} or ch in "\n\t"
    )


def clean_chapter_text(text: str) -> str:
    """Apply the full cleaning pipeline to a single chapter string.

    Steps: remove non-printable chars → normalize whitespace.
    Does NOT strip Gutenberg markers (that is handled at the book level).

    Args:
        text: Raw chapter text.

    Returns:
        Cleaned chapter text ready for NLP processing.
    """
    text = remove_non_printable(text)
    text = normalize_whitespace(text)
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text into a flat list of word/punctuation tokens.

    Uses a simple regex that keeps contractions and hyphenated words
    intact. Suitable for co-occurrence window computation.

    Args:
        text: Cleaned chapter text.

    Returns:
        List of lowercase string tokens.
    """
    return [tok.lower() for tok in _TOKENIZE_PATTERN.findall(text)]


def token_generator(text: str) -> Generator[str, None, None]:
    """Yield tokens one at a time for memory-efficient iteration.

    Args:
        text: Cleaned chapter text.

    Yields:
        Lowercase string tokens.
    """
    for tok in _TOKENIZE_PATTERN.finditer(text):
        yield tok.group().lower()
