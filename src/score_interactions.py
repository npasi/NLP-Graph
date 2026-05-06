from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def evidence_text(ev: dict[str, Any]) -> str:
    for key in ("text", "snippet", "sentence", "sentence_text"):
        value = ev.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def evidence_dedupe_key(ev: dict[str, Any]) -> str:
    sentence_id = ev.get("sentence_id")
    etype = ev.get("evidence_type", "")
    predicate = ev.get("predicate", "")

    if sentence_id is not None:
        return f"sid:{sentence_id}|{etype}|{predicate}"

    text = evidence_text(ev).lower()
    if text:
        return f"text:{text}|{etype}|{predicate}"

    return f"fallback:{json.dumps(ev, sort_keys=True)}"


def dedupe_evidence(evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for ev in evidences:
        key = evidence_dedupe_key(ev)
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)

    return out


def score_pair(evidences: list[dict[str, Any]]) -> dict[str, Any]:
    deduped = dedupe_evidence(evidences)

    direct_events = [e for e in deduped if e.get("evidence_type") == "DIRECT_EVENT"]
    co_presence = [e for e in deduped if e.get("evidence_type") == "CO_PRESENCE"]

    predicates = Counter(
        str(e.get("predicate"))
        for e in direct_events
        if e.get("predicate")
    )

    direct_event_count = len(direct_events)
    co_presence_count = len(co_presence)
    total_evidence_count = len(deduped)

    # Conservative interaction score:
    # DIRECT_EVENT is strong evidence.
    # CO_PRESENCE is weak scene-level evidence.
    interaction_score = (
        2.0 * direct_event_count
        + 0.25 * co_presence_count
    )

    if direct_event_count >= 3:
        confidence = "high"
    elif direct_event_count >= 1:
        confidence = "medium"
    elif co_presence_count >= 5:
        confidence = "low_medium"
    elif co_presence_count >= 1:
        confidence = "low"
    else:
        confidence = "none"

    relation_type = "direct_interaction" if direct_event_count > 0 else "co_presence_only"

    return {
        "direct_event_count": direct_event_count,
        "co_presence_count": co_presence_count,
        "total_evidence_count": total_evidence_count,
        "interaction_score": round(interaction_score, 4),
        "confidence": confidence,
        "relation_type": relation_type,
        "top_predicates": [
            {"predicate": p, "count": c}
            for p, c in predicates.most_common(10)
        ],
    }


def score_all_chapters(evidence_by_chapter: dict[str, Any]) -> dict[str, Any]:
    scored: dict[str, Any] = {}

    for chapter_id, pairs in evidence_by_chapter.items():
        scored[chapter_id] = {}

        for pair_key, evidences in pairs.items():
            if not isinstance(evidences, list):
                continue

            scored[chapter_id][pair_key] = score_pair(evidences)

    return scored


def build_summary(scored: dict[str, Any]) -> dict[str, Any]:
    total_pairs = 0
    direct_pairs = 0
    co_only_pairs = 0
    total_score = 0.0

    chapter_summaries: dict[str, Any] = {}

    for chapter_id, pairs in scored.items():
        chapter_pairs = len(pairs)
        chapter_direct = sum(
            1 for p in pairs.values()
            if p.get("direct_event_count", 0) > 0
        )
        chapter_co_only = sum(
            1 for p in pairs.values()
            if p.get("direct_event_count", 0) == 0 and p.get("co_presence_count", 0) > 0
        )
        chapter_score = sum(float(p.get("interaction_score", 0.0)) for p in pairs.values())

        total_pairs += chapter_pairs
        direct_pairs += chapter_direct
        co_only_pairs += chapter_co_only
        total_score += chapter_score

        chapter_summaries[chapter_id] = {
            "pairs": chapter_pairs,
            "direct_event_pairs": chapter_direct,
            "co_presence_only_pairs": chapter_co_only,
            "total_interaction_score": round(chapter_score, 4),
        }

    return {
        "total_pairs": total_pairs,
        "direct_event_pairs": direct_pairs,
        "co_presence_only_pairs": co_only_pairs,
        "total_interaction_score": round(total_score, 4),
        "chapters": chapter_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score normalized character interaction evidence."
    )
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / args.book_id
    )

    output_dir = Path(args.output_dir) if args.output_dir else booknlp_root
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = booknlp_root / "normalized_pair_evidence_by_chapter.json"
    evidence = load_json(evidence_path)

    scored = score_all_chapters(evidence)
    summary = build_summary(scored)

    scored_path = output_dir / "scored_pair_evidence_by_chapter.json"
    summary_path = output_dir / "interaction_score_summary.json"

    write_json(scored_path, scored)
    write_json(summary_path, summary)

    print(f"Wrote {scored_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
