"""Dynamic chapter splitter that produces exactly target_n segments.

Strategy (in order):
1. Run natural chapter detector. If count == target_n → done.
2. If count > target_n → merge consecutive chapters evenly into target_n buckets.
3. If count < target_n (including 1, i.e. detection failed) → divide the raw
   text into target_n equal parts at sentence boundaries.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.chapter_splitter import split_into_chapters
from src.step1_split_only import _strip_gutenberg_footer

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_GUTENBERG_START_RE = re.compile(
    r'^\*\*\*\s*START OF (THE )?PROJECT GUTENBERG[^\n]*\n',
    re.IGNORECASE | re.MULTILINE,
)


def _strip_gutenberg_header(text: str) -> str:
    """Remove the Gutenberg preamble (everything before *** START OF ***)."""
    m = _GUTENBERG_START_RE.search(text)
    return text[m.end():] if m else text


def _group_chapters(chapters: List[dict], target_n: int) -> List[dict]:
    """Merge consecutive natural chapters into exactly target_n buckets."""
    n = len(chapters)
    out: List[dict] = []
    for i in range(target_n):
        start = round(i * n / target_n)
        end   = round((i + 1) * n / target_n)
        bucket = chapters[start:end]
        if not bucket:
            continue
        merged = "\n\n".join(ch["text"] for ch in bucket)
        out.append({
            "chapter_id":  i,
            "title":       bucket[0]["title"],
            "text":        merged,
            "num_tokens":  len(merged.split()),
            "short":       len(merged.split()) < 200,
            "raw_heading": bucket[0].get("raw_heading", ""),
        })
    return out


def _split_by_position(text: str, target_n: int) -> List[dict]:
    """Divide text into target_n parts at sentence boundaries."""
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        sentences = [text.strip()]
    total = len(sentences)
    out: List[dict] = []
    for i in range(target_n):
        start = round(i * total / target_n)
        end   = round((i + 1) * total / target_n)
        chunk = " ".join(sentences[start:end])
        out.append({
            "chapter_id":  i,
            "title":       f"Part {i + 1}",
            "text":        chunk,
            "num_tokens":  len(chunk.split()),
            "short":       len(chunk.split()) < 200,
            "raw_heading": "",
        })
    return out


def split_dynamic(text: str, target_n: int) -> List[dict]:
    """Return exactly target_n chapter dicts from *text*.

    Parameters
    ----------
    text:
        Raw book text. Gutenberg footer is stripped automatically.
    target_n:
        Desired number of output segments (= GT chapter count for training).
    """
    if target_n < 1:
        raise ValueError(f"target_n must be >= 1, got {target_n}")

    text = _strip_gutenberg_header(_strip_gutenberg_footer(text))
    chapters = split_into_chapters(text)
    n = len(chapters)

    if n == target_n:
        return chapters

    if n > target_n:
        return _group_chapters(chapters, target_n)

    # n < target_n (includes n==1 fallback when detector found nothing)
    return _split_by_position(text, target_n)


__all__ = ["split_dynamic"]
