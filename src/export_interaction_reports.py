from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ARROW = "◄──►"
SEP = "=" * 70
SUBSEP = "-" * 70


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def character_name_lookup(characters: list[dict[str, Any]]) -> dict[str, str]:
    return {
        c["canonical_id"]: c.get("name", c["canonical_id"])
        for c in characters
    }


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def evidence_text(item: dict[str, Any]) -> str:
    for key in ("text", "snippet", "sentence", "sentence_text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    pred = item.get("predicate")
    if pred:
        return f"predicate={pred}"
    return "[no text available]"


def evidence_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    etype = item.get("evidence_type", "")
    text = evidence_text(item)
    priority = 0 if etype == "DIRECT_EVENT" else 1
    return priority, len(text), text


def dedupe_evidence(evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for ev in sorted(evidences, key=evidence_sort_key):
        sentence_id = ev.get("sentence_id")
        text = evidence_text(ev)

        if sentence_id is not None:
            key = f"sid:{sentence_id}|{ev.get('evidence_type', '')}|{ev.get('predicate', '')}"
        else:
            key = f"text:{normalize_text(text).lower()}|{ev.get('evidence_type', '')}|{ev.get('predicate', '')}"

        if key in seen:
            continue

        seen.add(key)
        out.append(ev)

    return out


def pair_sort_key(pair_key: str) -> tuple[str, str]:
    parts = pair_key.split("||")
    if len(parts) == 2:
        return parts[0], parts[1]
    return pair_key, ""


def summarize_evidence(evidences: list[dict[str, Any]]) -> dict[str, Any]:
    direct         = [e for e in evidences if e.get("evidence_type") == "DIRECT_EVENT"]
    dialogue_turns = [e for e in evidences if e.get("evidence_type") == "DIALOGUE_TURN"]
    quote_abouts   = [e for e in evidences if e.get("evidence_type") == "QUOTE_ABOUT"]
    co             = [e for e in evidences if e.get("evidence_type") == "CO_PRESENCE"]

    predicates = Counter(
        str(e.get("predicate"))
        for e in direct
        if e.get("predicate")
    )

    return {
        "direct_event_count":   len(direct),
        "dialogue_turn_count":  len(dialogue_turns),
        "quote_about_count":    len(quote_abouts),
        "co_presence_count":    len(co),
        "total_evidence_count": len(evidences),
        "top_predicates":       predicates.most_common(5),
    }


def format_predicates(predicates: list[tuple[str, int]]) -> str:
    if not predicates:
        return "none"
    return ", ".join(f"{p}×{c}" if c > 1 else p for p, c in predicates)


def export_chapter(
    *,
    book_id: str,
    chapter_id: str,
    chapter_data: dict[str, list[dict[str, Any]]],
    names: dict[str, str],
    output_dir: Path,
    max_evidence_per_pair: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"chapter_{int(chapter_id) + 1}_interactions.txt"

    pair_items = sorted(chapter_data.items(), key=lambda x: pair_sort_key(x[0]))

    lines: list[str] = []
    lines.append(SEP)
    lines.append(f"CHAPTER {int(chapter_id) + 1} - CHARACTER INTERACTIONS")
    lines.append(f"Book ID: {book_id}")
    lines.append(f"Total unique character pairs: {len(pair_items)}")
    lines.append(SEP)
    lines.append("")

    for i, (pair_key, raw_evidences) in enumerate(pair_items, start=1):
        parts = pair_key.split("||")
        if len(parts) == 2:
            a_id, b_id = parts
        else:
            a_id, b_id = pair_key, "UNKNOWN"

        a_name = names.get(a_id, a_id)
        b_name = names.get(b_id, b_id)

        evidences = dedupe_evidence(raw_evidences)
        summary = summarize_evidence(evidences)

        lines.append(f"[PAIR {i}] {a_name} {ARROW} {b_name}")
        lines.append(SUBSEP)
        lines.append(f"  Direct events:   {summary['direct_event_count']}")
        lines.append(f"  Dialogue turns:  {summary['dialogue_turn_count']}")
        lines.append(f"  Quote-about:     {summary['quote_about_count']}")
        lines.append(f"  Co-presence:     {summary['co_presence_count']}")
        lines.append(f"  Total evidence:  {summary['total_evidence_count']}")
        lines.append(f"  Top predicates:  {format_predicates(summary['top_predicates'])}")
        lines.append("")

        shown = evidences[:max_evidence_per_pair]

        if not shown:
            lines.append("  • [no evidence]")
        else:
            for ev in shown:
                etype = ev.get("evidence_type", "EVIDENCE")
                text = evidence_text(ev)
                predicate = ev.get("predicate")

                if etype == "DIRECT_EVENT" and predicate:
                    lines.append(f"  • [DIRECT_EVENT] ({predicate}) {text}")
                else:
                    lines.append(f"  • [{etype}] {text}")

        if len(evidences) > max_evidence_per_pair:
            remaining = len(evidences) - max_evidence_per_pair
            lines.append(f"  • ... {remaining} more evidence item(s)")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export readable chapter interaction reports from normalized evidence."
    )
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-evidence-per-pair", type=int, default=5)
    args = parser.parse_args()

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / args.book_id
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("data") / "reports" / args.book_id / "interactions"
    )

    characters = load_json(booknlp_root / "canonical_characters.json")
    evidence = load_json(booknlp_root / "normalized_pair_evidence_by_chapter.json")

    names = character_name_lookup(characters)

    written: list[Path] = []
    for chapter_id, chapter_data in sorted(evidence.items(), key=lambda x: int(x[0])):
        written.append(
            export_chapter(
                book_id=args.book_id,
                chapter_id=chapter_id,
                chapter_data=chapter_data,
                names=names,
                output_dir=output_dir,
                max_evidence_per_pair=args.max_evidence_per_pair,
            )
        )

    index_path = output_dir / "README.txt"
    index_lines = [
        f"Interaction reports for book_id={args.book_id}",
        f"Total chapters exported: {len(written)}",
        "Report format: DIRECT_EVENT first, then shortest CO_PRESENCE evidence.",
        f"Max evidence per pair: {args.max_evidence_per_pair}",
        "",
        "Files:",
    ]
    index_lines.extend(f"- {p.name}" for p in written)
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"Wrote {len(written)} chapter report(s) to {output_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
