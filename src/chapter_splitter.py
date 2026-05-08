"""Split raw book text into chapters.

Gutenberg books use a surprising variety of heading formats. We run
several detectors in priority order and keep the first one whose result
looks like a progression of chapter numbers. Supported formats:

1. ``CHAPTER I`` / ``Chapter 1`` / ``CHAPTER ONE``
   (+ ``BOOK``, ``PART``, ``VOLUME``, ``SECTION``, ``LETTER``,
    ``EPISODE``, ``ACT``, ``SCENE``, ``STAVE``, ``CANTO`` variants).
2. Hierarchical containers (PART / BOOK / VOLUME / ACT) followed by
   inner chapters (CHAPTER / SCENE / etc.).  Numbers may reset between
   containers.  Produces "Part I - Chapter One" style titles.
3. ALL CAPS literary titles on their own line, preceded by a blank line
   (e.g. *Dr. Jekyll and Mr. Hyde*: ``STORY OF THE DOOR``).
4. ``I`` / ``II`` / ``III`` — bare Roman numerals on their own line
   (*The Turn of the Screw*).
5. ``01 My Early Home`` — numeric prefix + inline title
   (*Black Beauty*).
6. ``1`` / ``2`` / ``3`` — bare arabic numerals on their own line.

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

* The text is clipped to the span between the Gutenberg ``*** START OF
  … ***`` and ``*** END OF … ***`` markers before detection, so license
  sections (``SECTION 1``, ``GENERAL TERMS OF USE``, …) are never seen
  by any detector.
* Gutenberg boilerplate lines and title-page metadata lines are filtered
  from the ALL CAPS detector.
* A **table of contents** at the front of the book is detected and
  dropped (headings in the first ~5% of the text with a tiny body).
  For container headings (PART / BOOK / VOLUME), the body is measured
  as text to the *next container*, so a "Part I" at the start of real
  content (with thousands of words under it) is never mistaken for a
  TOC entry.
* Duplicate chapter numbers (TOC + body) are merged: we keep the
  occurrence with the longest body.
* Sections with zero words in their body are suppressed.
* Detector output is **validated for progression** (numbers monotone +
  small gaps) so we don't mistake e.g. a single in-line "Part I" for a
  full chapter scheme.  ALL CAPS headings use a body-length heuristic
  instead because they carry no numeric sequence.

If no detector yields a valid set of headings, the whole text is
returned as a single chapter titled ``"Full text"``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SHORT_CHAPTER_THRESHOLD = 200
TOC_ZONE_FRAC = 0.05
TOC_BODY_MAX_TOKENS = 500
MIN_HEADINGS_FOR_VALID_SCHEME = 3
MIN_PROGRESSION_RATIO = 0.6
ALLCAPS_MIN_AVG_BODY = 50  # all-caps chapters must have avg body ≥ 50 words

# Words that act as CONTAINERS for inner chapters (numbers may reset inside them).
_CONTAINER_WORDS = frozenset({"PART", "BOOK", "VOLUME", "ACT"})

# Hierarchy level for containers (lower number = higher in the hierarchy).
# VOLUME > BOOK/PART > ACT.
_CONTAINER_LEVEL: Dict[str, int] = {
    "VOLUME": 1,
    "BOOK": 2,
    "PART": 2,
    "ACT": 2,
}


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

# ── Gutenberg structural markers ─────────────────────────────────────────────

_GUTENBERG_START_RE = re.compile(
    r"^\*{3}\s*START\s+OF\s+(?:THE\s+)?PROJECT\s+GUTENBERG[^\r\n]*\*{3}[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_GUTENBERG_END_RE = re.compile(
    r"^\*{3}\s*END\s+OF\s+(?:THE\s+)?PROJECT\s+GUTENBERG[^\r\n]*\*{3}[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


def _find_book_bounds(text: str) -> Tuple[int, int]:
    """Return (start, end) of the actual book content.

    Clips away the Project Gutenberg preamble (before the START marker)
    and the license footer (after the END marker).  If markers are absent
    the whole text is returned.
    """
    start = 0
    end = len(text)
    m = _GUTENBERG_START_RE.search(text)
    if m:
        start = m.end()
    m = _GUTENBERG_END_RE.search(text)
    if m:
        end = m.start()
    return start, end


# ── ALL CAPS detector patterns ───────────────────────────────────────────────

# Matches a standalone line that is entirely upper-case with common punctuation,
# contains 2–10 words, and at most 80 characters of content.
# The character class includes both straight (U+0027) and curly (U+2018/U+2019)
# apostrophes so that headings like "JEKYLL’S" in Gutenberg texts match.
_ALLCAPS_HEADING_RE = re.compile(
    "(?m)^[ \\t]*"
    "(?P<title>[A-Z][A-Z0-9''‘’.\\-&,]*"
    "(?:[ \\t]+[A-Z0-9''‘’.\\-&,]+){1,9})"
    "[ \\t]*$"
)

# Lines that are Gutenberg boilerplate even when the text is not clipped.
_ALLCAPS_BOILERPLATE_RE = re.compile(
    r"^(?:"
    r"PROJECT\s+GUTENBERG|"
    r"GENERAL\s+TERMS?\s+OF\s+USE|"
    r"SECTION\s+\d|"
    r"(?:FULL\s+)?LICEN[SC]E|"
    r"DONATIONS?|"
    r"INFORMATION\s+ABOUT|"
    r"PREAMBLE|"
    r"DISCLAIMER|"
    r"LIMITATION\s+ON|"
    r"INDEMNITY|"
    r"WARRANTY|"
    r"TRADEMARK"
    r")",
    re.IGNORECASE,
)

# Lines that are title-page metadata.
_ALLCAPS_METADATA_RE = re.compile(
    r"^(?:"
    r"BY\b|"
    r"TRANSLATED\s+BY|"
    r"EDITED\s+BY|"
    r"ILLUSTRATED?\s+BY|"
    r"WITH\s+(?:AN?\s+)?(?:INTRODUCTION|PREFACE)|"
    r"DEDICATION|"
    r"DEDICATED\s+TO|"
    r"PRODUCED\s+BY"
    r")",
    re.IGNORECASE,
)


def _has_blank_line_before(text: str, pos: int) -> bool:
    """True if *pos* is preceded by a blank line or by only whitespace.

    The "only whitespace" branch handles the first chapter heading in a book,
    which may appear at or near position 0 of the clipped book text.
    """
    segment = text[max(0, pos - 300): pos]
    return bool(re.search(r"\n\s*\n\s*$", segment)) or not segment.strip()


# ── Number helpers ────────────────────────────────────────────────────────────


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


# ── Heading detector regexes ──────────────────────────────────────────────────

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
    r"LETTER|Letter|letter|"
    r"EPISODE|Episode|episode"
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
    (?:[ \t.:\-–—]+
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
                "title": (m.group("title") or "").strip(" .:-–—\t"),
                "raw": m.group(0).strip(),
                "word": m.group("word"),
            }
        )
    return out


def _detect_allcaps_headings(text: str) -> List[HeadingRec]:
    """Detect ALL CAPS literary chapter headings with no numeric sequence.

    Returns headings with sequential ``num`` values (1, 2, …) and
    ``word = "allcaps"`` so that ``_format_title`` can output the raw
    title instead of ``Chapter N: …``.
    """
    out: List[HeadingRec] = []
    for m in _ALLCAPS_HEADING_RE.finditer(text):
        title = m.group("title").strip()
        if len(title.split()) < 2:
            continue
        if _ALLCAPS_BOILERPLATE_RE.match(title):
            continue
        if _ALLCAPS_METADATA_RE.match(title):
            continue
        if not _has_blank_line_before(text, m.start()):
            continue
        out.append(
            {
                "start": m.start(),
                "end": m.end(),
                "num": 0,       # placeholder; filled below
                "num_text": "",
                "title": title,
                "raw": title,
                "word": "allcaps",
            }
        )
    for i, h in enumerate(out):
        h["num"] = i + 1
        h["num_text"] = str(i + 1)
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
        title = m.group("title").strip(" .:-–—\t")
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
    ("allcaps", _detect_allcaps_headings),
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


def _looks_valid_allcaps(headings: List[HeadingRec], text: str) -> bool:
    """Validate ALL CAPS headings by count and average body length.

    ALL CAPS headings carry no numeric sequence, so progression scoring
    does not apply.  Instead we require enough headings with substantial
    bodies so we don't confuse a handful of bold section labels for real
    chapters.
    """
    if len(headings) < MIN_HEADINGS_FOR_VALID_SCHEME:
        return False
    for i, h in enumerate(headings):
        if "body_len" not in h:
            body_end = int(headings[i + 1]["start"]) if i + 1 < len(headings) else len(text)
            h["body_len"] = len(text[int(h["end"]): body_end].split())
    avg_body = sum(int(h["body_len"]) for h in headings) / len(headings)
    if avg_body < ALLCAPS_MIN_AVG_BODY:
        return False
    return True


def _looks_valid_mixed(headings: List[HeadingRec]) -> bool:
    """Validate a mixed-word-type heading sequence.

    Accepts when the total count meets the minimum AND each word type is
    internally strictly monotone by parsed number.  This handles books
    that open with LETTER sections and continue with CHAPTER sections:
    e.g. Letter I, Letter II, Chapter I, Chapter II, Chapter III.
    """
    if len(headings) < MIN_HEADINGS_FOR_VALID_SCHEME:
        return False
    by_type: Dict[str, List[int]] = defaultdict(list)
    for h in headings:  # assumed sorted by start already
        key = str(h.get("word", "")).upper()
        by_type[key].append(int(h["num"]))
    for nums in by_type.values():
        if any(nums[i + 1] <= nums[i] for i in range(len(nums) - 1)):
            return False
    return True


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
    if word == "allcaps":
        # The title IS the chapter heading; no numeric prefix needed.
        return title
    label = word.capitalize() if word else "Chapter"
    base = f"{label} {num_text}".strip()
    if title:
        return f"{base}: {title}"
    return base


def _make_record(chapter_id: int, heading: HeadingRec, body: str) -> dict:
    n_tokens = len(body.split())
    # Use pre-computed composite title (set by nested detection).
    if "_heading_label" in heading:
        title = str(heading["_heading_label"])
    else:
        title = _format_title(
            str(heading.get("word", "")),
            str(heading["num_text"]),
            str(heading.get("title", "")),
        )
    return {
        "chapter_id": chapter_id,
        "title": title,
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


# ── Nested container + chapter detection ─────────────────────────────────────


def _try_nested_chapters(
    text: str, raw_headings: List[HeadingRec]
) -> Optional[List[HeadingRec]]:
    """Detect hierarchical container + chapter structure.

    Handles: PART I → Chapter One, VOLUME I → BOOK FIRST → CHAPTER I,
    ACT I → SCENE I, and similar nested patterns where chapter numbers
    reset within each container.

    Returns a flat list of composite heading records (with ``_heading_label``
    set to the combined "Part I - Chapter One" form), or ``None`` if no
    valid nested structure is found.
    """
    if not raw_headings:
        return None

    toc_cutoff = int(len(text) * TOC_ZONE_FRAC)

    # Annotate every heading with in_toc_zone.
    hs: List[HeadingRec] = [dict(h) for h in raw_headings]
    for h in hs:
        h["in_toc_zone"] = int(h["start"]) < toc_cutoff
        h["_is_container"] = str(h.get("word", "")).upper() in _CONTAINER_WORDS

    # Compute body lengths differently for containers and leaves:
    # - Container: text from this container end to the NEXT container start
    #   (captures the full part/book/volume, not just the whitespace before
    #   the first child chapter).
    # - Leaf: text from this heading end to the next heading of any type.
    containers_list = [h for h in hs if h["_is_container"]]

    # Leaf body: next heading (any type)
    for i, h in enumerate(hs):
        body_end = int(hs[i + 1]["start"]) if i + 1 < len(hs) else len(text)
        h["body_len"] = len(text[int(h["end"]):body_end].split())

    # Container body: next container
    for i, h in enumerate(containers_list):
        next_cont = int(containers_list[i + 1]["start"]) if i + 1 < len(containers_list) else len(text)
        h["container_body_len"] = len(text[int(h["end"]):next_cont].split())

    # TOC filter:
    # - Container: filter if in_toc_zone AND container_body_len < threshold
    # - Leaf: filter if in_toc_zone AND body_len < threshold
    def _is_toc(h: HeadingRec) -> bool:
        if not h["in_toc_zone"]:
            return False
        if h["_is_container"]:
            return h.get("container_body_len", 0) < TOC_BODY_MAX_TOKENS
        return int(h["body_len"]) < TOC_BODY_MAX_TOKENS

    non_toc = [h for h in hs if not _is_toc(h)]

    containers = [h for h in non_toc if h["_is_container"]]
    leaves = [h for h in non_toc if not h["_is_container"]]

    if len(containers) < 2 or len(leaves) < MIN_HEADINGS_FOR_VALID_SCHEME:
        return None

    # Walk all non-TOC headings in document order, maintaining a context stack.
    # Stack: list of (hierarchy_level, formatted_label, word_upper, num).
    context_stack: List[Tuple[int, str, str, int]] = []
    result: List[HeadingRec] = []

    for h in sorted(non_toc, key=lambda x: int(x["start"])):
        if h["_is_container"]:
            word_upper = str(h.get("word", "")).upper()
            level = _CONTAINER_LEVEL.get(word_upper, 2)
            label = _format_title(
                str(h["word"]), str(h["num_text"]), str(h.get("title", ""))
            )
            # Pop context entries at the same or deeper level.
            context_stack = [
                (l, lb, w, n) for l, lb, w, n in context_stack if l < level
            ]
            context_stack.append((level, label, word_upper, int(h["num"])))
        else:
            if not context_stack:
                continue  # Chapter before any container — skip.
            ctx_parts = [lb for _, lb, _, _ in context_stack]
            ch_label = _format_title(
                str(h["word"]), str(h["num_text"]), str(h.get("title", ""))
            )
            full_title = " - ".join(ctx_parts + [ch_label])
            # Key used for reset detection: immediate parent word+num.
            immediate_key = context_stack[-1][2] + str(context_stack[-1][3])
            rec = dict(h)
            rec["_heading_label"] = full_title
            rec["_parent_key"] = immediate_key
            result.append(rec)

    if len(result) < MIN_HEADINGS_FOR_VALID_SCHEME:
        return None

    # Require that chapter numbers reset across at least two containers
    # (same number appears under different parent keys).
    nums_by_parent: Dict[str, set] = defaultdict(set)
    for r in result:
        nums_by_parent[str(r["_parent_key"])].add(int(r["num"]))

    parent_sets = list(nums_by_parent.values())
    has_reset = any(
        parent_sets[i] & parent_sets[j]
        for i in range(len(parent_sets))
        for j in range(i + 1, len(parent_sets))
    )

    if not has_reset:
        logger.debug("Nested structure found but no number reset; using standard detection.")
        return None

    logger.info(
        "Nested structure detected: %d containers, %d composite sections.",
        len(containers),
        len(result),
    )
    return result


# ── Chapter record builder ────────────────────────────────────────────────────


def _build_chapters_from_headings(
    headings: List[HeadingRec],
    text: str,
    suppress_empty: bool = True,
) -> List[dict]:
    """Build final chapter records from a heading list.

    Sections with an empty body (zero tokens) are suppressed when
    *suppress_empty* is True.  This removes spurious entries like letter
    signatures ("HASTIE LANYON.") that the ALL CAPS detector picks up
    but whose body is immediately followed by the next real heading.
    """
    raw: List[Tuple[HeadingRec, str]] = []
    for i, h in enumerate(headings):
        body_start = int(h["end"])
        body_end = int(headings[i + 1]["start"]) if i + 1 < len(headings) else len(text)
        body = text[body_start:body_end].strip()
        raw.append((h, body))

    if suppress_empty:
        dropped = sum(1 for _, b in raw if not b.split())
        if dropped:
            logger.info("Suppressed %d zero-token section(s).", dropped)
        raw = [(h, b) for h, b in raw if b.split()]

    if not raw:
        return _single_full_text_record(text)

    chapters: List[dict] = []
    for chapter_id, (h, body) in enumerate(raw):
        rec = _make_record(chapter_id, h, body)
        chapters.append(rec)
        if rec["short"]:
            logger.warning(
                "Short chapter flagged: %s — only %d tokens.",
                rec["title"],
                rec["num_tokens"],
            )
    return chapters


# ── Public API ────────────────────────────────────────────────────────────────


def split_into_chapters(text: str) -> List[dict]:
    book_start, book_end = _find_book_bounds(text)
    book_text = text[book_start:book_end]

    # ── 1. Try nested container + chapter detection ───────────────────────────
    raw_word = _detect_word_headings(book_text)
    if raw_word:
        nested = _try_nested_chapters(book_text, raw_word)
        if nested is not None:
            return _build_chapters_from_headings(nested, book_text)

    # ── 2. Standard detector cascade ─────────────────────────────────────────
    winner_name: Optional[str] = None
    winner_headings: List[HeadingRec] = []

    for name, detector in _DETECTORS:
        raw = detector(book_text)
        if len(raw) < MIN_HEADINGS_FOR_VALID_SCHEME:
            logger.debug("Detector %s: %d raw matches (skipped).", name, len(raw))
            continue
        cleaned = _drop_toc_and_duplicates(book_text, raw)
        if name == "word":
            sub = _best_valid_subscheme(cleaned)
            if sub:
                cleaned = sub
            # Fallback for mixed section-type books (e.g. LETTER + CHAPTER).
            if not _looks_valid(cleaned) and _looks_valid_mixed(cleaned):
                logger.info("Using mixed-word scheme: %d headings.", len(cleaned))
                valid = True
            else:
                valid = _looks_valid(cleaned)
        elif name == "allcaps":
            valid = _looks_valid_allcaps(cleaned, book_text)
        else:
            valid = _looks_valid(cleaned)

        if not valid:
            logger.debug("Detector %s rejected: %d headings.", name, len(cleaned))
            continue

        if winner_name is None or len(cleaned) >= 2 * len(winner_headings):
            logger.info(
                "Splitter detector '%s' now leading: %d headings.", name, len(cleaned)
            )
            winner_name = name
            winner_headings = cleaned

    if not winner_headings:
        logger.warning("No valid chapter scheme found — returning whole text as 1 chapter.")
        return _single_full_text_record(book_text)

    logger.info(
        "Final splitter choice: detector '%s' with %d chapters.",
        winner_name,
        len(winner_headings),
    )
    return _build_chapters_from_headings(winner_headings, book_text)


def split_into_chapters_with_debug(text: str) -> Tuple[List[dict], dict]:
    book_start, book_end = _find_book_bounds(text)
    book_text = text[book_start:book_end]

    debug: dict = {
        "detectors": [],
        "book_bounds": {"start": book_start, "end": book_end},
        "nested": {"attempted": False, "success": False, "sections": 0},
    }

    # ── 1. Try nested detection ───────────────────────────────────────────────
    raw_word = _detect_word_headings(book_text)
    debug["nested"]["attempted"] = bool(raw_word)
    if raw_word:
        nested = _try_nested_chapters(book_text, raw_word)
        if nested is not None:
            debug["nested"]["success"] = True
            debug["nested"]["sections"] = len(nested)
            debug["nested"]["sample_titles"] = [
                str(h.get("_heading_label", "")) for h in nested[:5]
            ]
            debug["winner"] = {"name": "nested", "headings": len(nested)}
            return _build_chapters_from_headings(nested, book_text), debug

    # ── 2. Standard detector cascade ─────────────────────────────────────────
    winner_name: Optional[str] = None
    winner_headings: List[HeadingRec] = []

    for name, detector in _DETECTORS:
        raw = detector(book_text)
        cleaned = _drop_toc_and_duplicates(book_text, raw) if raw else []
        if name == "word":
            sub = _best_valid_subscheme(cleaned) if cleaned else []
            if sub:
                cleaned = sub
            if not _looks_valid(cleaned) and _looks_valid_mixed(cleaned):
                valid = True
                prog = 1.0
            else:
                prog = _progression_score(cleaned) if cleaned else 0.0
                valid = _looks_valid(cleaned) if cleaned else False
        elif name == "allcaps":
            valid = _looks_valid_allcaps(cleaned, book_text) if cleaned else False
            prog = 1.0 if valid else 0.0
        else:
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
        return _single_full_text_record(book_text), debug

    return _build_chapters_from_headings(winner_headings, book_text), debug


__all__ = ["split_into_chapters", "split_into_chapters_with_debug"]
