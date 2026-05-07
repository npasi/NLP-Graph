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
    "co_presence_only_pairs",
    "direct_event_ratio",
    "co_presence_only_ratio",
    "max_nodes",
    "max_edges",
    "bert_examples",
    "bert_unique_pairs",
    "warnings",
    "overall_status",
]


def _status(row: dict) -> str:
    if not row["has_quality_report"]:
        return "NOT_EVALUATED"
    if row["chapter_count"] in (0, 1, ""):
        return "FAIL_SPLIT_OR_MISSING"
    if row["total_pairs"] == 0 or row["bert_examples"] == 0:
        return "FAIL_GRAPH_OR_EVIDENCE"
    if isinstance(row["direct_event_ratio"], float) and row["direct_event_ratio"] < 0.10:
        return "BORDERLINE_LOW_DIRECT"
    return "EVALUABLE"


def _build_row(
    book_id: str,
    quality_path: Path,
    bert_path: Path,
    chapters_path: Path,
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

    if bert_path.exists():
        try:
            b = json.loads(bert_path.read_text(encoding="utf-8"))
            row["bert_examples"] = b.get("example_count", "")
            row["bert_unique_pairs"] = b.get("unique_pair_count", "")
        except Exception:
            pass

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
    out = reports_root / "evaluation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        _build_row(
            book_id=book_id,
            quality_path=reports_root / book_id / "quality_report.json",
            bert_path=ml_root / book_id / "bert_relation_summary.json",
            chapters_path=books_root / book_id / "chapters" / "chapters.json",
        )
        for book_id in LEGACY_BOOKS
    ]

    _write_csv(out, rows)
    print(f"Wrote {out}")


def run_run_mode(run_dir: Path) -> None:
    reports_root = run_dir / "reports"
    ml_root = run_dir / "ml"
    books_root = run_dir / "books"
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
