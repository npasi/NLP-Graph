from __future__ import annotations

import argparse
import csv
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


def norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def evidence_text(ev: dict[str, Any]) -> str:
    for key in ("text", "snippet", "sentence", "sentence_text"):
        v = ev.get(key)
        if isinstance(v, str) and v.strip():
            return norm_text(v)
    return ""


def pair_to_chars(pair_key: str) -> tuple[str, str]:
    parts = pair_key.split("||")
    if len(parts) == 2:
        return parts[0], parts[1]
    return pair_key, ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_characters(
    *,
    book_id: str,
    characters: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    rows = []
    for c in characters:
        rows.append({
            "book_id": book_id,
            "canonical_id": c.get("canonical_id"),
            "name": c.get("name"),
            "type": c.get("type"),
            "confidence": c.get("confidence"),
            "alias_count": len(c.get("aliases", []) or []),
            "aliases": " | ".join(c.get("aliases", []) or []),
            "source_cluster_count": len(c.get("source_clusters", []) or []),
        })

    write_csv(
        out_dir / "characters.csv",
        rows,
        [
            "book_id",
            "canonical_id",
            "name",
            "type",
            "confidence",
            "alias_count",
            "aliases",
            "source_cluster_count",
        ],
    )


def export_edges(
    *,
    book_id: str,
    scored: dict[str, Any],
    names: dict[str, str],
    out_dir: Path,
) -> None:
    rows = []

    for chapter_id, pairs in scored.items():
        for pair_key, score in pairs.items():
            char_a, char_b = pair_to_chars(pair_key)
            top_preds = score.get("top_predicates", []) or []
            pred_string = " | ".join(
                f"{p.get('predicate')}:{p.get('count')}"
                for p in top_preds
            )

            rows.append({
                "book_id": book_id,
                "chapter_id": chapter_id,
                "pair_key": pair_key,
                "char_a": char_a,
                "char_b": char_b,
                "char_a_name": names.get(char_a, char_a),
                "char_b_name": names.get(char_b, char_b),
                "direct_event_count":   score.get("direct_event_count", 0),
                "dialogue_turn_count":  score.get("dialogue_turn_count", 0),
                "quote_about_count":    score.get("quote_about_count", 0),
                "strong_evidence_count": score.get("strong_evidence_count", 0),
                "co_presence_count":    score.get("co_presence_count", 0),
                "total_evidence_count": score.get("total_evidence_count", 0),
                "interaction_score":    score.get("interaction_score", 0),
                "confidence":           score.get("confidence", ""),
                "relation_type":        score.get("relation_type", ""),
                "top_predicates":       pred_string,
            })

    rows.sort(
        key=lambda r: (
            int(r["chapter_id"]),
            -float(r["interaction_score"]),
            r["char_a_name"],
            r["char_b_name"],
        )
    )

    write_csv(
        out_dir / "edges.csv",
        rows,
        [
            "book_id",
            "chapter_id",
            "pair_key",
            "char_a",
            "char_b",
            "char_a_name",
            "char_b_name",
            "direct_event_count",
            "dialogue_turn_count",
            "quote_about_count",
            "strong_evidence_count",
            "co_presence_count",
            "total_evidence_count",
            "interaction_score",
            "confidence",
            "relation_type",
            "top_predicates",
        ],
    )


def export_evidence(
    *,
    book_id: str,
    evidence: dict[str, Any],
    names: dict[str, str],
    out_dir: Path,
) -> None:
    rows = []

    for chapter_id, pairs in evidence.items():
        for pair_key, evs in pairs.items():
            char_a, char_b = pair_to_chars(pair_key)
            for idx, ev in enumerate(evs):
                rows.append({
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "pair_key": pair_key,
                    "char_a": char_a,
                    "char_b": char_b,
                    "char_a_name": names.get(char_a, char_a),
                    "char_b_name": names.get(char_b, char_b),
                    "evidence_index": idx,
                    "evidence_type": ev.get("evidence_type", ""),
                    "sentence_id": ev.get("sentence_id", ""),
                    "predicate": ev.get("predicate", ""),
                    "subject": ev.get("subject", ""),
                    "object": ev.get("object", ""),
                    "confidence": ev.get("confidence", ""),
                    "text": evidence_text(ev),
                })

    write_csv(
        out_dir / "evidence.csv",
        rows,
        [
            "book_id",
            "chapter_id",
            "pair_key",
            "char_a",
            "char_b",
            "char_a_name",
            "char_b_name",
            "evidence_index",
            "evidence_type",
            "sentence_id",
            "predicate",
            "subject",
            "object",
            "confidence",
            "text",
        ],
    )


def export_chapter_metrics(
    *,
    book_id: str,
    quality: dict[str, Any],
    score_summary: dict[str, Any],
    out_dir: Path,
) -> None:
    graph_chapters = {
        str(c.get("chapter_id")): c
        for c in ((quality.get("graphs") or {}).get("chapters") or [])
    }

    score_chapters = score_summary.get("chapters", {}) or {}

    all_chapter_ids = sorted(
        set(graph_chapters) | set(score_chapters),
        key=lambda x: int(x),
    )

    rows = []

    for chapter_id in all_chapter_ids:
        g = graph_chapters.get(chapter_id, {})
        s = score_chapters.get(chapter_id, {})

        rows.append({
            "book_id": book_id,
            "chapter_id": chapter_id,
            "nodes": g.get("nodes", 0),
            "edges": g.get("edges", 0),
            "pairs": s.get("pairs", 0),
            "direct_event_pairs": s.get("direct_event_pairs", 0),
            "co_presence_only_pairs": s.get("co_presence_only_pairs", 0),
            "total_interaction_score": s.get("total_interaction_score", 0),
        })

    write_csv(
        out_dir / "chapter_metrics.csv",
        rows,
        [
            "book_id",
            "chapter_id",
            "nodes",
            "edges",
            "pairs",
            "direct_event_pairs",
            "co_presence_only_pairs",
            "total_interaction_score",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NLP-Graph outputs as CSV tables.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    book_id = args.book_id

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / book_id
    )
    reports_root = Path(args.reports_root) if args.reports_root else (
        Path("data") / "reports" / book_id
    )
    out_dir = Path(args.output_dir) if args.output_dir else (
        reports_root / "tables"
    )

    characters = load_json(booknlp_root / "canonical_characters.json", default=[])
    evidence = load_json(booknlp_root / "normalized_pair_evidence_by_chapter.json", default={})
    scored = load_json(booknlp_root / "scored_pair_evidence_by_chapter.json", default={})
    score_summary = load_json(booknlp_root / "interaction_score_summary.json", default={})
    quality = load_json(reports_root / "quality_report.json", default={})

    names = {
        c.get("canonical_id"): c.get("name", c.get("canonical_id"))
        for c in characters
    }

    export_characters(book_id=book_id, characters=characters, out_dir=out_dir)
    export_edges(book_id=book_id, scored=scored, names=names, out_dir=out_dir)
    export_evidence(book_id=book_id, evidence=evidence, names=names, out_dir=out_dir)
    export_chapter_metrics(
        book_id=book_id,
        quality=quality,
        score_summary=score_summary,
        out_dir=out_dir,
    )

    print(f"Wrote CSV tables to {out_dir}")
    print(f"- {out_dir / 'characters.csv'}")
    print(f"- {out_dir / 'edges.csv'}")
    print(f"- {out_dir / 'evidence.csv'}")
    print(f"- {out_dir / 'chapter_metrics.csv'}")


if __name__ == "__main__":
    main()
