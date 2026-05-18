"""Step 1 of the Longformer pipeline.

Parse per-chapter interaction files (BookNLP-derived) into Longformer-ready
strings.

Input file format
-----------------

    ======================================================================
    CHAPTER N - PROPER NAME INTERACTIONS
    Total unique proper name pairs: K
    ======================================================================

    [PAIR 1] NameA <UNICODE-ARROW> NameB
    ----------------------------------------------------------------------
      * line 1 (bullet)
      * line 2 (bullet)
      ...

The bullet character is U+2022 (`*` is just used in this docstring as a
placeholder). Names may contain spaces, apostrophes, or dots
(e.g. ``"Scrooge 's nephew"``, ``"Three Spirits"``, ``"Dr. Heywood Floyd"``).
Consecutive bullet lines belong to the same text block and must be joined
with a single space.

Output
------

For each pair, ``process_chapter_file`` produces a dict containing the raw
joined text and a Longformer-ready string with entity markers
``[E1] ... [/E1]`` for ``char_a`` and ``[E2] ... [/E2]`` for ``char_b``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


_PAIR_HEADER_RE = re.compile(
    r"^\s*\[PAIR\s+\d+\]\s+(?P<a>.+?)\s+\u25C4\u2500\u2500\u25BA\s+(?P<b>.+?)\s*$",
    flags=re.MULTILINE,
)
_BULLET_LINE_RE = re.compile(r"^\s*\u2022\s?(.*)$")


def parse_interactions_file(filepath: str) -> List[Dict]:
    """Parse one interactions ``.txt`` file into a list of pair dicts.

    Each returned dict has keys ``char_a``, ``char_b`` and ``text`` where
    ``text`` is the bullet lines joined into a single space-separated string.
    Pairs with empty text are dropped.
    """
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="replace")

    headers = list(_PAIR_HEADER_RE.finditer(raw))
    pairs: List[Dict] = []

    for i, m in enumerate(headers):
        char_a = m.group("a").strip()
        char_b = m.group("b").strip()
        block_start = m.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        block = raw[block_start:block_end]

        bullet_lines: List[str] = []
        for line in block.splitlines():
            bm = _BULLET_LINE_RE.match(line)
            if not bm:
                continue
            content = bm.group(1).rstrip()
            if content:
                bullet_lines.append(content)

        if not bullet_lines:
            continue

        joined = " ".join(bullet_lines)
        joined = re.sub(r"\s+", " ", joined).strip()
        if not joined:
            continue

        pairs.append({"char_a": char_a, "char_b": char_b, "text": joined})

    print(f"[parse] File: {filepath}")
    print(f"[parse] Found {len(pairs)} pairs")
    for p in pairs:
        print(
            f"[parse]   {p['char_a']} \u25C4\u2500\u2500\u25BA {p['char_b']}: "
            f"{len(p['text'])} chars, first 80: {p['text'][:80]}..."
        )

    assert isinstance(pairs, list), "pairs must be a list"
    for p in pairs:
        assert "char_a" in p and "char_b" in p and "text" in p, (
            f"Missing keys in pair: {list(p.keys())}"
        )
        assert len(p["text"]) > 0, (
            f"Empty text for pair {p['char_a']}-{p['char_b']}"
        )
        assert p["char_a"] != p["char_b"], (
            f"Self-pair detected: {p['char_a']}"
        )

    return pairs


def _build_name_pattern(name: str) -> str:
    """Compile a regex body for a character name with flexible whitespace.

    - Tokens separated by whitespace are joined with ``\\s+`` so that
      ``"Three Spirits"`` matches ``"Three Spirits"`` and ``"Three  Spirits"``.
    - A leading apostrophe token (e.g. ``"'s"``) is reattached to the previous
      token so that ``"Scrooge 's nephew"`` matches ``"Scrooge's nephew"``
      (which is how BookNLP-derived bullet text usually writes it).
    - The full pattern is wrapped in ``\\b`` word boundaries.
    """
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        raise ValueError(f"Empty name: {name!r}")

    merged: List[str] = []
    for p in parts:
        if p.startswith("'") and merged:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)

    parts_escaped = [re.escape(p) for p in merged]
    body = parts_escaped[0] if len(parts_escaped) == 1 else r"\s+".join(parts_escaped)
    return rf"\b{body}\b"


def add_entity_markers(text: str, char_a: str, char_b: str) -> str:
    """Wrap occurrences of ``char_a``/``char_b`` with ``[E1]``/``[E2]`` markers."""
    if not char_a or not char_b:
        raise ValueError("Both char_a and char_b must be non-empty")

    items = [(char_a, 1), (char_b, 2)]
    items.sort(key=lambda kv: -len(kv[0]))

    pat_first = _build_name_pattern(items[0][0])
    pat_second = _build_name_pattern(items[1][0])

    if items[0][1] == 1:
        combined = rf"(?P<g1>{pat_first})|(?P<g2>{pat_second})"
    else:
        combined = rf"(?P<g2>{pat_first})|(?P<g1>{pat_second})"

    def _replace(m: re.Match) -> str:
        if m.group("g1") is not None:
            return f"[E1] {m.group('g1').strip()} [/E1]"
        return f"[E2] {m.group('g2').strip()} [/E2]"

    marked = re.sub(combined, _replace, text)

    count_e1 = marked.count("[E1]")
    count_e2 = marked.count("[E2]")
    print(f"[markers] {char_a}: {count_e1} markers, {char_b}: {count_e2} markers")
    assert marked.count("[E1]") == marked.count("[/E1]"), "Mismatched [E1] tags"
    assert marked.count("[E2]") == marked.count("[/E2]"), "Mismatched [E2] tags"
    assert "[E1] [E2]" not in marked, "Nested markers detected"

    return marked


def build_longformer_input(
    char_a: str,
    char_b: str,
    marked_text: str,
    *,
    prev_score: float | None = None,
) -> str:
    """Compose the final string fed to Longformer.

    Option A (state-as-text): prepend a short line with the previous chapter score
    when available. This does NOT require any tokenizer/vocab changes.
    """
    if prev_score is None:
        prev_line = "Previous: none.\n\n"
    else:
        prev_line = f"Previous: {float(prev_score):.3f}.\n\n"

    result = (
        prev_line
        + f"Characters: [E1] {char_a} [/E1] and [E2] {char_b} [/E2].\n"
        + "\n"
        + f"{marked_text}"
    )
    print(f"[build] Input length: {len(result)} chars")
    print(f"[build] First 200 chars: {result[:200]}")
    assert "Characters:" in result[:50], "Must include header near start"
    assert "[E1]" in result and "[E2]" in result, "Missing entity markers"
    return result


def process_chapter_file(filepath: str) -> List[Dict]:
    """End-to-end: parse, mark, and build Longformer inputs for one chapter."""
    print(f"\n{'=' * 60}")
    print(f"[process] Processing: {filepath}")
    print(f"{'=' * 60}")

    pairs = parse_interactions_file(filepath)

    results: List[Dict] = []
    for i, pair in enumerate(pairs):
        print(
            f"\n[process] --- Pair {i + 1}/{len(pairs)}: "
            f"{pair['char_a']} \u25C4\u2500\u2500\u25BA {pair['char_b']} ---"
        )
        marked_text = add_entity_markers(pair["text"], pair["char_a"], pair["char_b"])
        longformer_input = build_longformer_input(
            pair["char_a"], pair["char_b"], marked_text,
        )
        results.append(
            {
                "char_a": pair["char_a"],
                "char_b": pair["char_b"],
                "raw_text": pair["text"],
                "longformer_input": longformer_input,
            }
        )

    print(f"\n[process] Done. Produced {len(results)} Longformer inputs.")
    assert len(results) == len(pairs), "Result count mismatch"
    return results


def write_report(results: List[Dict], out_path: str) -> None:
    """Write a human-readable report for debugging/inspection."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("INPUT BUILDER REPORT")
    lines.append("=" * 80)
    lines.append(f"Pairs: {len(results)}")
    lines.append("")

    for i, r in enumerate(results, 1):
        char_a = r["char_a"]
        char_b = r["char_b"]
        raw_text = r["raw_text"]
        lf = r["longformer_input"]

        lines.append("-" * 80)
        lines.append(f"PAIR {i}: {char_a} ◄──► {char_b}")
        lines.append("-" * 80)
        lines.append(f"Raw text length: {len(raw_text)}")
        lines.append(f"Longformer input length: {len(lf)}")
        lines.append(f"[E1] count: {lf.count('[E1]')}   [E2] count: {lf.count('[E2]')}")
        lines.append("")
        lines.append("RAW TEXT:")
        lines.append(raw_text)
        lines.append("")
        lines.append("LONGFORMER INPUT:")
        lines.append(lf)
        lines.append("")

    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[report] Wrote report to: {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("filepath", help="Path to chapter interactions .txt file")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to write a human-readable .txt report",
    )
    args = ap.parse_args()

    results = process_chapter_file(args.filepath)
    if args.out:
        write_report(results, args.out)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['char_a']} \u25C4\u2500\u2500\u25BA {r['char_b']}")
        print(f"    Raw text length: {len(r['raw_text'])} chars")
        print(f"    Longformer input length: {len(r['longformer_input'])} chars")
        print(f"    Preview: {r['longformer_input'][:150]}...")
        print()
