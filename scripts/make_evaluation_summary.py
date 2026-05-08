"""Generate evaluation_summary.csv from pipeline outputs.

Legacy mode (no args):
    python scripts/make_evaluation_summary.py

    Reads from data/reports/<book_id>/, data/ml/<book_id>/, data/books/<book_id>/
    for books listed in LEGACY_BOOKS. Writes data/reports/evaluation_summary.csv.

Run mode (--run-dir):
    python scripts/make_evaluation_summary.py --run-dir outputs/runs/<run_id>

    Discovers book_ids from subdirs of <run_dir>/reports/.
    Writes <run_dir>/evaluation_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LEGACY_BOOKS = [
    "gatsby",
    "the_trial",
    "frankenstein",
    "jekyll",
    "heart_darkness",
    "madame_bovary",
]

COLUMNS = [
    "book_id",
    "has_quality_report",
    "chapter_count",
    "canonical_characters",
    "unresolved_entities",
    "review_clusters",
    "total_pairs",
    "direct_event_pairs",
    "dialogue_turn_pairs",
    "quote_about_pairs",
    "quote_evidence_pairs",
    "strong_evidence_pairs",
    "co_presence_only_pairs",
    "direct_event_ratio",
    "co_presence_only_ratio",
    "strong_evidence_ratio",
    "quote_evidence_ratio",
    "dialogue_turn_evidence",
    "quote_about_evidence",
    "strong_evidence_total",
    "max_nodes",
    "max_edges",
    "bert_examples",
    "bert_unique_pairs",
    "bert_example_density",
    "warnings",
    "overall_status",
]


def _load_score_stats(score_summary_path: Path, scored_path: Path) -> dict:
    """Return quote/strong evidence stats from summary + scored files.

    Reads pair counts from interaction_score_summary.json (already aggregated).
    Reads per-evidence totals and quote_evidence_pairs from scored_pair_evidence_by_chapter.json.
    Degrades gracefully if either file is absent or missing new fields.
    """
    stats: dict = {
        "dialogue_turn_pairs": 0,
        "quote_about_pairs": 0,
        "quote_evidence_pairs": 0,
        "strong_evidence_pairs": 0,
        "dialogue_turn_evidence": 0,
        "quote_about_evidence": 0,
        "strong_evidence_total": 0,
    }

    if score_summary_path.exists():
        try:
            s = json.loads(score_summary_path.read_text(encoding="utf-8"))
            stats["dialogue_turn_pairs"] = int(s.get("dialogue_turn_pairs", 0) or 0)
            stats["quote_about_pairs"] = int(s.get("quote_about_pairs", 0) or 0)
            stats["strong_evidence_pairs"] = int(s.get("strong_evidence_pairs", 0) or 0)
        except Exception:
            pass

    if scored_path.exists():
        try:
            scored = json.loads(scored_path.read_text(encoding="utf-8"))
            dt_ev = 0
            qa_ev = 0
            se_total = 0
            qe_pairs = 0
            dt_pairs = 0
            qa_pairs = 0
            strong_pairs = 0

            for _chapter_id, pairs in scored.items():
                for _pair_key, pair_score in pairs.items():
                    dt = int(pair_score.get("dialogue_turn_count", 0) or 0)
                    qa = int(pair_score.get("quote_about_count", 0) or 0)
                    de = int(pair_score.get("direct_event_count", 0) or 0)
                    strong = int(
                        pair_score.get("strong_evidence_count", de + dt + qa) or 0
                    )

                    dt_ev += dt
                    qa_ev += qa
                    se_total += strong

                    if dt > 0 or qa > 0:
                        qe_pairs += 1
                    if dt > 0:
                        dt_pairs += 1
                    if qa > 0:
                        qa_pairs += 1
                    if strong > 0:
                        strong_pairs += 1

            stats["dialogue_turn_evidence"] = dt_ev
            stats["quote_about_evidence"] = qa_ev
            stats["strong_evidence_total"] = se_total
            stats["quote_evidence_pairs"] = qe_pairs

            # Fall back to scored-computed values if summary lacked new fields
            if stats["dialogue_turn_pairs"] == 0 and dt_pairs > 0:
                stats["dialogue_turn_pairs"] = dt_pairs
            if stats["quote_about_pairs"] == 0 and qa_pairs > 0:
                stats["quote_about_pairs"] = qa_pairs
            if stats["strong_evidence_pairs"] == 0 and strong_pairs > 0:
                stats["strong_evidence_pairs"] = strong_pairs

        except Exception:
            pass

    return stats


def _status(row: dict) -> str:
    if not row["has_quality_report"]:
        return "MISSING_REPORT"

    total = row["total_pairs"]
    if total == "" or total == 0:
        return "FAIL_NO_PAIRS"

    bert = row["bert_examples"]
    if bert == "" or bert == 0:
        return "FAIL_NO_BERT_EXAMPLES"

    strong = int(row["strong_evidence_pairs"] or 0)
    bert = int(bert or 0)
    co_ratio = float(row["co_presence_only_ratio"] or 0.0)
    strong_ratio = float(row["strong_evidence_ratio"] or 0.0)

    # High co-presence: evaluate quality based on residual strong evidence
    if co_ratio >= 0.75:
        if strong >= 10 and bert >= 20:
            return "EVALUABLE_CO_PRESENCE_HEAVY"
        if strong >= 5 and bert > 0:
            return "BORDERLINE_CO_PRESENCE_HEAVY"

    # Low co-presence: standard strong-evidence tiers
    if strong >= 20 and bert >= 50:
        return "EVALUABLE_STRONG"
    if strong >= 5 and bert >= 20:
        return "EVALUABLE"

    # Weak evidence floor
    if strong < 5 or strong_ratio < 0.05:
        return "BORDERLINE_WEAK_EVIDENCE"

    return "EVALUABLE"


def _build_row(
    book_id: str,
    quality_path: Path,
    bert_path: Path,
    chapters_path: Path,
    score_summary_path: Path,
    scored_path: Path,
) -> dict:
    row: dict = {col: "" for col in COLUMNS}
    row["book_id"] = book_id
    row["has_quality_report"] = quality_path.exists()

    if chapters_path.exists():
        try:
            chapters = json.loads(chapters_path.read_text(encoding="utf-8"))
            row["chapter_count"] = len(chapters)
        except Exception:
            pass

    if quality_path.exists():
        try:
            q = json.loads(quality_path.read_text(encoding="utf-8"))
        except Exception:
            q = {}

        identity = q.get("identity", {})
        interactions = q.get("interactions", {})
        graphs = q.get("graphs", {})

        row["chapter_count"] = q.get("chapter_count", row["chapter_count"])
        row["canonical_characters"] = identity.get("canonical_characters", "")
        row["unresolved_entities"] = identity.get("unresolved_entities", "")
        row["review_clusters"] = identity.get("decision_counts", {}).get("REVIEW", "")

        row["total_pairs"] = interactions.get("total_pairs", "")
        row["direct_event_pairs"] = interactions.get("direct_event_pairs", "")
        row["co_presence_only_pairs"] = interactions.get("co_presence_only_pairs", "")
        row["direct_event_ratio"] = interactions.get("direct_event_ratio", "")
        row["co_presence_only_ratio"] = interactions.get("co_presence_only_ratio", "")

        row["max_nodes"] = graphs.get("max_nodes", "")
        row["max_edges"] = graphs.get("max_edges", "")

        row["warnings"] = " | ".join(q.get("warnings", []))

    # Load quote/strong evidence stats
    score_stats = _load_score_stats(score_summary_path, scored_path)
    row["dialogue_turn_pairs"] = score_stats["dialogue_turn_pairs"]
    row["quote_about_pairs"] = score_stats["quote_about_pairs"]
    row["quote_evidence_pairs"] = score_stats["quote_evidence_pairs"]
    row["strong_evidence_pairs"] = score_stats["strong_evidence_pairs"]
    row["dialogue_turn_evidence"] = score_stats["dialogue_turn_evidence"]
    row["quote_about_evidence"] = score_stats["quote_about_evidence"]
    row["strong_evidence_total"] = score_stats["strong_evidence_total"]

    total_pairs = int(row["total_pairs"] or 0)
    strong_pairs = int(row["strong_evidence_pairs"] or 0)
    qe_pairs = int(row["quote_evidence_pairs"] or 0)

    row["strong_evidence_ratio"] = (
        round(strong_pairs / total_pairs, 4) if total_pairs else 0.0
    )
    row["quote_evidence_ratio"] = (
        round(qe_pairs / total_pairs, 4) if total_pairs else 0.0
    )

    if bert_path.exists():
        try:
            b = json.loads(bert_path.read_text(encoding="utf-8"))
            row["bert_examples"] = b.get("example_count", "")
            row["bert_unique_pairs"] = b.get("unique_pair_count", "")
        except Exception:
            pass

    bert_examples = int(row["bert_examples"] or 0)
    row["bert_example_density"] = (
        round(bert_examples / strong_pairs, 4) if strong_pairs else 0.0
    )

    row["overall_status"] = _status(row)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_legacy() -> None:
    reports_root = Path("data/reports")
    ml_root = Path("data/ml")
    books_root = Path("data/books")
    booknlp_root = Path("data/booknlp_chapter_output")
    out = reports_root / "evaluation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        _build_row(
            book_id=book_id,
            quality_path=reports_root / book_id / "quality_report.json",
            bert_path=ml_root / book_id / "bert_relation_summary.json",
            chapters_path=books_root / book_id / "chapters" / "chapters.json",
            score_summary_path=booknlp_root / book_id / "interaction_score_summary.json",
            scored_path=booknlp_root / book_id / "scored_pair_evidence_by_chapter.json",
        )
        for book_id in LEGACY_BOOKS
    ]

    _write_csv(out, rows)
    print(f"Wrote {out}")


def run_run_mode(run_dir: Path) -> None:
    reports_root = run_dir / "reports"
    ml_root = run_dir / "ml"
    books_root = run_dir / "books"
    booknlp_root = run_dir / "booknlp_chapter_output"
    out = run_dir / "evaluation_summary.csv"

    book_ids: list[str] = []
    if reports_root.exists():
        book_ids = sorted(p.name for p in reports_root.iterdir() if p.is_dir())

    if not book_ids:
        print(f"No book directories found under {reports_root}. Nothing to summarize.")
        return

    rows = [
        _build_row(
            book_id=book_id,
            quality_path=reports_root / book_id / "quality_report.json",
            bert_path=ml_root / book_id / "bert_relation_summary.json",
            chapters_path=books_root / book_id / "chapters" / "chapters.json",
            score_summary_path=booknlp_root / book_id / "interaction_score_summary.json",
            scored_path=booknlp_root / book_id / "scored_pair_evidence_by_chapter.json",
        )
        for book_id in book_ids
    ]

    _write_csv(out, rows)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation_summary.csv from pipeline outputs."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to a run directory (e.g. outputs/runs/run_2026_05_07_eval_v1). "
             "Omit for legacy data/ layout.",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_run_mode(Path(args.run_dir))
    else:
        run_legacy()


if __name__ == "__main__":
    main()
