from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def evidence_text(ev: dict[str, Any]) -> str:
    for key in ("text", "snippet", "sentence", "sentence_text"):
        value = ev.get(key)
        if isinstance(value, str) and value.strip():
            return norm_text(value)
    predicate = ev.get("predicate")
    if predicate:
        return f"[EVENT predicate={predicate}]"
    return ""


def pair_to_chars(pair_key: str) -> tuple[str, str]:
    parts = pair_key.split("||")
    if len(parts) == 2:
        return parts[0], parts[1]
    return pair_key, ""


def character_names(characters: list[dict[str, Any]]) -> dict[str, str]:
    return {
        c.get("canonical_id"): c.get("name", c.get("canonical_id"))
        for c in characters
        if c.get("canonical_id")
    }


def should_include_pair(
    score: dict[str, Any],
    *,
    min_score: float,
    require_direct: bool,
) -> bool:
    direct = int(score.get("direct_event_count", 0) or 0)
    interaction_score = float(score.get("interaction_score", 0.0) or 0.0)

    if require_direct:
        return direct > 0

    return direct > 0 or interaction_score >= min_score


def build_examples(
    *,
    book_id: str,
    characters: list[dict[str, Any]],
    evidence_by_chapter: dict[str, Any],
    scored_by_chapter: dict[str, Any],
    min_score: float,
    require_direct: bool,
    max_examples_per_pair: int,
) -> list[dict[str, Any]]:
    names = character_names(characters)
    examples: list[dict[str, Any]] = []

    for chapter_id, scored_pairs in sorted(scored_by_chapter.items(), key=lambda x: int(x[0])):
        chapter_evidence = evidence_by_chapter.get(chapter_id, {})

        for pair_key, score in scored_pairs.items():
            if not should_include_pair(score, min_score=min_score, require_direct=require_direct):
                continue

            char_a, char_b = pair_to_chars(pair_key)
            char_a_name = names.get(char_a, char_a)
            char_b_name = names.get(char_b, char_b)

            evs = chapter_evidence.get(pair_key, [])
            if not isinstance(evs, list):
                continue

            # DIRECT_EVENT first, then CO_PRESENCE.
            evs = sorted(
                evs,
                key=lambda ev: (
                    0 if ev.get("evidence_type") == "DIRECT_EVENT" else 1,
                    len(evidence_text(ev)),
                ),
            )

            kept = 0
            seen_texts: set[str] = set()

            for ev in evs:
                text = evidence_text(ev)
                if not text:
                    continue

                text_key = text.lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                examples.append({
                    "book_id": book_id,
                    "chapter_id": int(chapter_id),
                    "pair_key": pair_key,
                    "char_a": char_a,
                    "char_b": char_b,
                    "char_a_name": char_a_name,
                    "char_b_name": char_b_name,
                    "text": text,
                    "evidence_type": ev.get("evidence_type", ""),
                    "predicate": ev.get("predicate", ""),
                    "sentence_id": ev.get("sentence_id", ""),
                    "interaction_score": score.get("interaction_score", 0.0),
                    "direct_event_count": score.get("direct_event_count", 0),
                    "co_presence_count": score.get("co_presence_count", 0),
                    "relation_type": score.get("relation_type", ""),
                    "relation_confidence": score.get("confidence", ""),
                    "label": None,
                    "polarity_score": None,
                    "split": "unassigned",
                })

                kept += 1
                if kept >= max_examples_per_pair:
                    break

    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}

    for ex in examples:
        evidence_counts[ex["evidence_type"]] = evidence_counts.get(ex["evidence_type"], 0) + 1
        relation_counts[ex["relation_type"]] = relation_counts.get(ex["relation_type"], 0) + 1

    unique_pairs = sorted({ex["pair_key"] for ex in examples})

    return {
        "example_count": len(examples),
        "unique_pair_count": len(unique_pairs),
        "evidence_type_counts": evidence_counts,
        "relation_type_counts": relation_counts,
        "note": "Labels are placeholders. Add polarity labels/scores downstream.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BERT-ready relation examples.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-score", type=float, default=2.0)
    parser.add_argument("--require-direct", action="store_true")
    parser.add_argument("--max-examples-per-pair", type=int, default=5)
    args = parser.parse_args()

    book_id = args.book_id

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / book_id
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("data") / "ml" / book_id
    )

    characters = load_json(booknlp_root / "canonical_characters.json", default=[])
    evidence = load_json(booknlp_root / "normalized_pair_evidence_by_chapter.json", default={})
    scored = load_json(booknlp_root / "scored_pair_evidence_by_chapter.json", default={})

    examples = build_examples(
        book_id=book_id,
        characters=characters,
        evidence_by_chapter=evidence,
        scored_by_chapter=scored,
        min_score=args.min_score,
        require_direct=args.require_direct,
        max_examples_per_pair=args.max_examples_per_pair,
    )

    jsonl_path = output_dir / "bert_relation_examples.jsonl"
    summary_path = output_dir / "bert_relation_summary.json"

    write_jsonl(jsonl_path, examples)
    write_json(summary_path, build_summary(examples))

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    main()
