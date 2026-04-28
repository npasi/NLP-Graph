"""Split raw book text into chapters.

Gutenberg books use a surprising variety of heading formats. We run
several detectors in priority order and keep the first one whose result
looks like a progression of chapter numbers. Supported formats:

1. ``CHAPTER I`` / ``Chapter 1`` / ``CHAPTER ONE``
   (+ ``BOOK``, ``PART``, ``VOLUME``, ``SECTION`` variants).
2. ``I`` / ``II`` / ``III`` — bare Roman numerals on their own line
   (*The Turn of the Screw*).
3. ``01 My Early Home`` — numeric prefix + inline title
   (*Black Beauty*).
4. ``1`` / ``2`` / ``3`` — bare arabic numerals on their own line.

Each record returned is a plain ``dict`` with the schema the pipeline
expects::

    {
        "chapter_id":  int,    # 0-based position in the book
        "title":       str,    # "Chapter 3: A New Hope"
        "text":        str,    # body (heading excluded)
        "num_tokens":  int,    # whitespace-token count
        "short":       bool,   # True when num_tokens < 200
        "raw_heading": str,    # the literal line we split on
    }

Robustness features:

* A **table of contents** at the front of the book is detected and
  dropped (headings in the first ~5% of the text with a tiny body).
* Duplicate chapter numbers (TOC + body) are merged: we keep the
  occurrence with the longest body.
* Detector output is **validated for progression** (numbers monotone +
  small gaps) so we don't mistake e.g. a single in-line "Part I" for a
  full chapter scheme.

If no detector yields a valid set of headings, the whole text is
returned as a single chapter titled ``"Full text"``.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SHORT_CHAPTER_THRESHOLD = 200
TOC_ZONE_FRAC = 0.05
TOC_BODY_MAX_TOKENS = 500
MIN_HEADINGS_FOR_VALID_SCHEME = 3
MIN_PROGRESSION_RATIO = 0.6


_SPELLED_NUMBERS = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
    "HUNDRED": 100,
    "FIRST": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "SIXTH": 6,
    "SEVENTH": 7,
    "EIGHTH": 8,
    "NINTH": 9,
    "TENTH": 10,
    "ELEVENTH": 11,
    "TWELFTH": 12,
    "THIRTEENTH": 13,
    "FOURTEENTH": 14,
    "FIFTEENTH": 15,
    "SIXTEENTH": 16,
    "SEVENTEENTH": 17,
    "EIGHTEENTH": 18,
    "NINETEENTH": 19,
    "TWENTIETH": 20,
    "THIRTIETH": 30,
    "FORTIETH": 40,
    "FIFTIETH": 50,
}

_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
_ARABIC_RE = re.compile(r"^\d+$")


def _roman_to_int(roman: str) -> Optional[int]:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(roman.upper()):
        if ch not in values:
            return None
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if total > 0 else None


def _spelled_to_int(token: str) -> Optional[int]:
    parts = re.split(r"[\s\-]+", token.strip().upper())
    values = []
    for p in parts:
        if p and p in _SPELLED_NUMBERS:
            values.append(_SPELLED_NUMBERS[p])
        else:
            return None
    if not values:
        return None
    total = 0
    pending = 0
    for v in values:
        if v == 100:
            pending = max(pending, 1) * 100
        else:
            pending += v
    return (total + pending) if pending else None


def _parse_number(token: str) -> Optional[int]:
    token = token.strip().strip(",.;:!?")
    if _ARABIC_RE.match(token):
        try:
            return int(token)
        except ValueError:
            return None
    if _ROMAN_RE.match(token):
        return _roman_to_int(token)
    return _spelled_to_int(token)


_SECTION_WORDS = (
    r"CHAPTER|Chapter|chapter|"
    r"BOOK|Book|book|"
    r"PART|Part|part|"
    r"VOLUME|Volume|volume|"
    r"SECTION|Section|section|"
    r"ACT|Act|act|"
    r"SCENE|Scene|scene|"
    r"STAVE|Stave|stave|"
    r"CANTO|Canto|canto|"
    r"LETTER|Letter|letter"
)

_WORD_HEADING_RE = re.compile(
    r"""
    ^[ \t]*
    (?P<word>"""
    + _SECTION_WORDS
    + r""")
    [ \t.:]+
    (?:THE[ \t]+)?
    (?P<num>[A-Za-z0-9\-]+)
    (?:[ \t.:\-\u2013\u2014]+
       (?P<title>[^\r\n]{0,80}))?
    [ \t.\r]*
    $
    """,
    re.MULTILINE | re.VERBOSE,
)

_ROMAN_ONLY_RE = re.compile(r"^[ \t]*(?P<num>[IVXLCDM]{1,6})\.?[ \t\r]*$", re.MULTILINE)
_NUMBER_TITLED_RE = re.compile(r"^[ \t]*(?P<num>\d{1,3})[ \t]+(?P<title>[A-Z][^\r\n]{0,80})\r?$", re.MULTILINE)
_NUMBER_ONLY_RE = re.compile(r"^[ \t]*(?P<num>\d{1,3})\.?[ \t\r]*$", re.MULTILINE)


HeadingRec = Dict[str, object]


def _detect_word_headings(text: str) -> List[HeadingRec]:
    out: List[HeadingRec] = []
    for m in _WORD_HEADING_RE.finditer(text):
        num_val = _parse_number(m.group("num"))
        if num_val is None:
            continue
        out.append(
            {
                "start": m.start(),
                "end": m.end(),
                "num": num_val,
                "num_text": m.group("num").strip(),
                "title": (m.group("title") or "").strip(" .:-\u2013\u2014\t"),
                "raw": m.group(0).strip(),
                "word": m.group("word"),
            }
        )
    return out


def _detect_roman_only(text: str) -> List[HeadingRec]:
    out: List[HeadingRec] = []
    for m in _ROMAN_ONLY_RE.finditer(text):
        num_val = _roman_to_int(m.group("num"))
        if num_val is None:
            continue
        out.append(
            {
                "start": m.start(),
                "end": m.end(),
                "num": num_val,
                "num_text": m.group("num"),
                "title": "",
                "raw": m.group(0).strip(),
                "word": "",
            }
        )
    return out


def _detect_number_titled(text: str) -> List[HeadingRec]:
    out: List[HeadingRec] = []
    for m in _NUMBER_TITLED_RE.finditer(text):
        try:
            num_val = int(m.group("num"))
        except ValueError:
            continue
        title = m.group("title").strip(" .:-\u2013\u2014\t")
        if sum(1 for w in title.split() if w and w[0].islower()) > 2:
            continue
        out.append(
            {
                "start": m.start(),
                "end": m.end(),
                "num": num_val,
                "num_text": m.group("num"),
                "title": title,
                "raw": m.group(0).strip(),
                "word": "",
            }
        )
    return out


def _detect_number_only(text: str) -> List[HeadingRec]:
    out: List[HeadingRec] = []
    for m in _NUMBER_ONLY_RE.finditer(text):
        try:
            num_val = int(m.group("num"))
        except ValueError:
            continue
        out.append(
            {
                "start": m.start(),
                "end": m.end(),
                "num": num_val,
                "num_text": m.group("num"),
                "title": "",
                "raw": m.group(0).strip(),
                "word": "",
            }
        )
    return out


_DETECTORS: List[Tuple[str, Callable[[str], List[HeadingRec]]]] = [
    ("word", _detect_word_headings),
    ("roman-only", _detect_roman_only),
    ("number-titled", _detect_number_titled),
    ("number-only", _detect_number_only),
]


def _drop_toc_and_duplicates(text: str, headings: List[HeadingRec]) -> List[HeadingRec]:
    if not headings:
        return headings

    toc_cutoff = int(len(text) * TOC_ZONE_FRAC)

    for i, h in enumerate(headings):
        body_end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
        h["body_len"] = len(text[h["end"] : body_end].split())
        h["in_toc_zone"] = h["start"] < toc_cutoff

    kept = [h for h in headings if not (h["in_toc_zone"] and h["body_len"] < TOC_BODY_MAX_TOKENS)]
    if len(kept) < len(headings):
        logger.info("Dropped %d TOC heading(s).", len(headings) - len(kept))

    def _priority(h: HeadingRec) -> Tuple[int, int]:
        return (0 if h["in_toc_zone"] else 1, int(h["body_len"]))

    by_key: Dict[Tuple[str, int], HeadingRec] = {}
    for h in kept:
        key = (str(h.get("word", "")).upper(), int(h["num"]))
        if key not in by_key or _priority(h) > _priority(by_key[key]):
            by_key[key] = h
    deduped = sorted(by_key.values(), key=lambda x: int(x["start"]))
    if len(deduped) < len(kept):
        logger.info("Merged %d duplicate chapter number(s).", len(kept) - len(deduped))
    return deduped


def _progression_score(headings: List[HeadingRec]) -> float:
    if len(headings) < 2:
        return 0.0
    nums = [int(h["num"]) for h in headings]
    gaps = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    ones = sum(1 for g in gaps if g == 1)
    monotonic = all(g >= 0 for g in gaps)
    if not monotonic:
        return 0.0
    return ones / len(gaps)


def _looks_valid(headings: List[HeadingRec]) -> bool:
    if len(headings) < MIN_HEADINGS_FOR_VALID_SCHEME:
        return False
    score = _progression_score(headings)
    return score >= MIN_PROGRESSION_RATIO


def _best_valid_subscheme(headings: List[HeadingRec]) -> List[HeadingRec]:
    """For the 'word' detector we may mix unrelated schemes (e.g. CHAPTER + SECTION).

    Pick the best valid subscheme by grouping on heading 'word'.
    """
    if not headings:
        return []
    groups: Dict[str, List[HeadingRec]] = {}
    for h in headings:
        key = str(h.get("word", "") or "").upper()
        groups.setdefault(key, []).append(h)

    best: List[HeadingRec] = []
    best_score: float = -1.0
    for key, group in groups.items():
        if not key or len(group) < MIN_HEADINGS_FOR_VALID_SCHEME:
            continue
        group_sorted = sorted(group, key=lambda x: int(x["start"]))
        if not _looks_valid(group_sorted):
            continue
        score = _progression_score(group_sorted)
        if (len(group_sorted) > len(best)) or (len(group_sorted) == len(best) and score > best_score):
            best = group_sorted
            best_score = score
    return best


def _format_title(word: str, num_text: str, title: str) -> str:
    label = word.capitalize() if word else "Chapter"
    base = f"{label} {num_text}".strip()
    if title:
        return f"{base}: {title}"
    return base


def _make_record(chapter_id: int, heading: HeadingRec, body: str) -> dict:
    n_tokens = len(body.split())
    return {
        "chapter_id": chapter_id,
        "title": _format_title(str(heading.get("word", "")), str(heading["num_text"]), str(heading.get("title", ""))),
        "text": body,
        "num_tokens": n_tokens,
        "short": n_tokens < SHORT_CHAPTER_THRESHOLD,
        "raw_heading": str(heading["raw"]),
    }


def _single_full_text_record(text: str) -> List[dict]:
    body = text.strip()
    return [
        {
            "chapter_id": 0,
            "title": "Full text",
            "text": body,
            "num_tokens": len(body.split()),
            "short": len(body.split()) < SHORT_CHAPTER_THRESHOLD,
            "raw_heading": "",
        }
    ]


def split_into_chapters(text: str) -> List[dict]:
    winner_name: Optional[str] = None
    winner_headings: List[HeadingRec] = []

    for name, detector in _DETECTORS:
        raw = detector(text)
        if len(raw) < MIN_HEADINGS_FOR_VALID_SCHEME:
            logger.debug("Detector %s: %d raw matches (skipped).", name, len(raw))
            continue
        cleaned = _drop_toc_and_duplicates(text, raw)
        if name == "word":
            cleaned = _best_valid_subscheme(cleaned) or cleaned
        if not _looks_valid(cleaned):
            logger.debug(
                "Detector %s rejected: %d headings, score %.2f.",
                name,
                len(cleaned),
                _progression_score(cleaned),
            )
            continue

        if winner_name is None or len(cleaned) >= 2 * len(winner_headings):
            logger.info(
                "Splitter detector '%s' now leading: %d headings, progression score %.2f.",
                name,
                len(cleaned),
                _progression_score(cleaned),
            )
            winner_name = name
            winner_headings = cleaned

    if not winner_headings:
        logger.warning("No valid chapter scheme found — returning whole text as 1 chapter.")
        return _single_full_text_record(text)

    logger.info("Final splitter choice: detector '%s' with %d chapters.", winner_name, len(winner_headings))

    chapters: List[dict] = []
    for i, h in enumerate(winner_headings):
        body_start = int(h["end"])
        body_end = int(winner_headings[i + 1]["start"]) if i + 1 < len(winner_headings) else len(text)
        body = text[body_start:body_end].strip()
        chapters.append(_make_record(i, h, body))

    for ch in chapters:
        if ch["short"]:
            logger.warning("Short chapter flagged: %s — only %d tokens.", ch["title"], ch["num_tokens"])
    return chapters


def split_into_chapters_with_debug(text: str) -> Tuple[List[dict], dict]:
    debug: dict = {"detectors": []}
    winner_name: Optional[str] = None
    winner_headings: List[HeadingRec] = []

    for name, detector in _DETECTORS:
        raw = detector(text)
        cleaned = _drop_toc_and_duplicates(text, raw) if raw else []
        subscheme = _best_valid_subscheme(cleaned) if (name == "word" and cleaned) else []
        if subscheme:
            cleaned = subscheme
        prog = _progression_score(cleaned) if cleaned else 0.0
        valid = _looks_valid(cleaned) if cleaned else False
        debug["detectors"].append(
            {
                "name": name,
                "raw_matches": len(raw),
                "cleaned_headings": len(cleaned),
                "progression_score": round(prog, 4),
                "valid": bool(valid),
                "sample_headings": [h.get("raw", "") for h in cleaned[:5]],
            }
        )

        if not valid:
            continue
        if winner_name is None or len(cleaned) >= 2 * len(winner_headings):
            winner_name = name
            winner_headings = cleaned

    debug["winner"] = {"name": winner_name, "headings": len(winner_headings)}

    if not winner_headings:
        debug["fallback"] = "single_chapter"
        return _single_full_text_record(text), debug

    chapters: List[dict] = []
    for i, h in enumerate(winner_headings):
        body_start = int(h["end"])
        body_end = int(winner_headings[i + 1]["start"]) if i + 1 < len(winner_headings) else len(text)
        body = text[body_start:body_end].strip()
        chapters.append(_make_record(i, h, body))

    return chapters, debug


__all__ = ["split_into_chapters", "split_into_chapters_with_debug"]

