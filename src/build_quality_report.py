from __future__ import annotations

import argparse
import json
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


def make_warnings(
    *,
    chapter_count: int,
    canonical_characters: int,
    unresolved_count: int,
    review_count: int,
    total_pairs: int,
    direct_event_pairs: int,
    strong_evidence_pairs: int,
    co_presence_only_pairs: int,
    graph_chapters: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []

    if chapter_count == 1:
        warnings.append(
            "Only one chapter detected. This may be valid, but check chapter splitting for long novels."
        )

    if canonical_characters > 80:
        warnings.append(
            f"High canonical character count ({canonical_characters}). Possible noisy identity layer."
        )

    if canonical_characters < 5:
        warnings.append(
            f"Very low canonical character count ({canonical_characters}). Possible over-filtering."
        )

    if review_count > canonical_characters * 10 and canonical_characters > 0:
        warnings.append(
            f"Many REVIEW clusters ({review_count}) relative to canonical characters ({canonical_characters}). Consider alias refinement."
        )

    if unresolved_count > canonical_characters * 15 and canonical_characters > 0:
        warnings.append(
            f"Many unresolved entities ({unresolved_count}) relative to canonical characters ({canonical_characters})."
        )

    if total_pairs > 0:
        strong_ratio = strong_evidence_pairs / total_pairs
        co_only_ratio = co_presence_only_pairs / total_pairs

        if strong_ratio < 0.15:
            warnings.append(
                f"Low strong-evidence ratio ({strong_ratio:.2%}). "
                "Graph is mostly co-presence (consider quote evidence or alias refinement)."
            )

        if co_only_ratio > 0.80:
            warnings.append(
                f"High co-presence-only ratio ({co_only_ratio:.2%}). Consider stronger edge filtering."
            )

    if graph_chapters:
        max_nodes = max(int(c.get("nodes", 0)) for c in graph_chapters)
        if max_nodes > 60:
            warnings.append(
                f"Some chapter graphs are large (max nodes={max_nodes}). Possible noisy nodes."
            )

        empty_chapters = [
            c.get("chapter_id")
            for c in graph_chapters
            if int(c.get("nodes", 0)) == 0 and int(c.get("edges", 0)) == 0
        ]
        if empty_chapters:
            warnings.append(
                f"Empty graph chapters detected: {empty_chapters}. This may indicate over-filtering or low interaction density."
            )

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NLP-Graph quality report.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--graphs-root", default="data/graphs/chapters")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    book_id = args.book_id

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / book_id
    )
    graphs_root = Path(args.graphs_root)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("data") / "reports" / book_id
    )

    identity_report = load_json(booknlp_root / "identity_report.json", default={})
    canonical = load_json(booknlp_root / "canonical_characters.json", default=[])
    unresolved = load_json(booknlp_root / "unresolved_entities.json", default=[])
    score_summary = load_json(booknlp_root / "interaction_score_summary.json", default={})
    graph_meta = load_json(graphs_root / f"{book_id}_normalized_metadata.json", default={})

    graph_chapters = graph_meta.get("chapters", [])
    chapter_count = len(graph_chapters)

    canonical_count = int(
        identity_report.get("canonical_characters", len(canonical))
    )

    decision_counts = (
        identity_report.get("decision_counts")
        or identity_report.get("decisions")
        or {}
    )

    review_count = int(decision_counts.get("REVIEW", 0))
    abstain_count = int(decision_counts.get("ABSTAIN", 0))
    reject_count = int(decision_counts.get("REJECT", 0))

    total_pairs           = int(score_summary.get("total_pairs", 0))
    direct_event_pairs    = int(score_summary.get("direct_event_pairs", 0))
    dialogue_turn_pairs   = int(score_summary.get("dialogue_turn_pairs", 0))
    quote_about_pairs     = int(score_summary.get("quote_about_pairs", 0))
    strong_evidence_pairs = int(score_summary.get("strong_evidence_pairs", 0))
    co_presence_only_pairs = int(score_summary.get("co_presence_only_pairs", 0))

    direct_event_ratio = (
        direct_event_pairs / total_pairs if total_pairs else 0.0
    )
    strong_evidence_ratio = (
        strong_evidence_pairs / total_pairs if total_pairs else 0.0
    )
    co_presence_only_ratio = (
        co_presence_only_pairs / total_pairs if total_pairs else 0.0
    )

    warnings = make_warnings(
        chapter_count=chapter_count,
        canonical_characters=canonical_count,
        unresolved_count=len(unresolved),
        review_count=review_count,
        total_pairs=total_pairs,
        direct_event_pairs=direct_event_pairs,
        strong_evidence_pairs=strong_evidence_pairs,
        co_presence_only_pairs=co_presence_only_pairs,
        graph_chapters=graph_chapters,
    )

    report = {
        "book_id": book_id,
        "chapter_count": chapter_count,
        "identity": {
            "canonical_characters": canonical_count,
            "local_clusters_read": identity_report.get("local_clusters_read"),
            "clusters_kept": identity_report.get("clusters_kept"),
            "clusters_discarded": identity_report.get("clusters_discarded"),
            "unresolved_entities": len(unresolved),
            "decision_counts": {
                "AUTO_MERGE": int(decision_counts.get("AUTO_MERGE", 0)),
                "REVIEW": review_count,
                "ABSTAIN": abstain_count,
                "REJECT": reject_count,
            },
            "category_counts": identity_report.get("category_counts", {}),
        },
        "interactions": {
            "total_pairs":           total_pairs,
            "direct_event_pairs":    direct_event_pairs,
            "dialogue_turn_pairs":   dialogue_turn_pairs,
            "quote_about_pairs":     quote_about_pairs,
            "strong_evidence_pairs": strong_evidence_pairs,
            "co_presence_only_pairs": co_presence_only_pairs,
            "direct_event_ratio":    round(direct_event_ratio, 4),
            "strong_evidence_ratio": round(strong_evidence_ratio, 4),
            "co_presence_only_ratio": round(co_presence_only_ratio, 4),
            "total_interaction_score": score_summary.get("total_interaction_score", 0.0),
        },
        "graphs": {
            "chapters": graph_chapters,
            "max_nodes": max([int(c.get("nodes", 0)) for c in graph_chapters], default=0),
            "max_edges": max([int(c.get("edges", 0)) for c in graph_chapters], default=0),
            "total_nodes_across_chapters": sum(int(c.get("nodes", 0)) for c in graph_chapters),
            "total_edges_across_chapters": sum(int(c.get("edges", 0)) for c in graph_chapters),
        },
        "warnings": warnings,
    }

    output_path = output_dir / "quality_report.json"
    write_json(output_path, report)

    print(f"Wrote {output_path}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("No warnings.")


if __name__ == "__main__":
    main()
