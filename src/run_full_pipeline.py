from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import logging

from src.utils.gutenberg_cleaner import clean_gutenberg_text
from src.utils.output_paths import OutputPaths
from src.utils.manifest import make_book_entry, write_manifest

logger = logging.getLogger(__name__)


def run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def ensure_alias_file(book_id: str) -> Path:
    alias_dir = Path("data/aliases")
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias_path = alias_dir / f"{book_id}.json"

    if not alias_path.exists():
        logger.info("Creating empty alias file: %s", alias_path)
        alias_path.write_text(
            '{\n  "aliases": {},\n  "chapter_aliases": {}\n}\n',
            encoding="utf-8"
        )

    return alias_path


def main():
    parser = argparse.ArgumentParser(description="Run full NLP-Graph pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--model-size", default="big")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root for run-based outputs (e.g. outputs/runs). "
             "Requires --run-id. Omit to use legacy data/ layout.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (e.g. run_2026_05_07_eval_v1). "
             "Outputs go to <output-root>/<run-id>/. "
             "Omit to use legacy data/ layout.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    paths = OutputPaths(
        book_id=args.book_id,
        output_root=args.output_root,
        run_id=args.run_id,
    )

    if paths.run_dir is not None:
        logger.info("Run mode: outputs under %s", paths.run_dir)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Legacy mode: outputs under data/")

    # ------------------------------------------------------------------ #
    # Gutenberg cleaning (writes alongside input; not run-specific)
    # ------------------------------------------------------------------ #
    clean_path = input_path.parent / f"{input_path.stem}_clean.txt"
    logger.info("Cleaning Gutenberg text...")
    cleaned_input = clean_gutenberg_text(input_path, clean_path)
    logger.info("Using cleaned file: %s", cleaned_input)

    alias_file = ensure_alias_file(args.book_id)

    pipeline_error: str | None = None

    try:
        # ---- Step 1: chapter splitting -------------------------------- #
        step1_cmd = [
            "python", "src/step1_split_only.py",
            "--input", str(cleaned_input),
            "--book-id", args.book_id,
            "--debug",
            "--log-level", args.log_level,
        ]
        if paths.run_dir is not None:
            step1_cmd += ["--books-root", str(paths.books_root)]
        run(step1_cmd)

        # ---- Step 2: BookNLP per chapter ------------------------------ #
        booknlp_cmd = [
            "python", "run_booknlp_per_chapter.py",
            "--book-id", args.book_id,
            "--model-size", args.model_size,
            "--pipeline", "entity,quote,supersense,event,coref",
            "--log-level", args.log_level,
        ]
        if paths.run_dir is not None:
            booknlp_cmd += [
                "--chapters-dir", str(paths.book_chapters_dir),
                "--output-root", str(paths.booknlp_book_dir),
            ]
        run(booknlp_cmd)

        # ---- Step 3: character identity layer ------------------------- #
        identity_cmd = [
            "python", "-m", "src.step2b_character_identity",
            "--book-id", args.book_id,
            "--alias-file", str(alias_file),
            "--log-level", args.log_level,
        ]
        if paths.run_dir is not None:
            identity_cmd += ["--booknlp-root", str(paths.booknlp_book_dir)]
        run(identity_cmd)

        # ---- Step 4: normalized predicates ---------------------------- #
        predicates_cmd = [
            "python", "-m", "src.step3b_normalized_predicates",
            "--book-id", args.book_id,
            "--log-level", args.log_level,
        ]
        if paths.run_dir is not None:
            predicates_cmd += ["--booknlp-root", str(paths.booknlp_book_dir)]
        run(predicates_cmd)

        # ---- Step 5: normalized evidence ------------------------------ #
        evidence_cmd = [
            "python", "-m", "src.step4b_normalized_evidence",
            "--book-id", args.book_id,
            "--log-level", args.log_level,
        ]
        if paths.run_dir is not None:
            evidence_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--chapters-root", str(paths.book_chapters_dir),
            ]
        run(evidence_cmd)

        # ---- Step 6: build normalized graphs -------------------------- #
        graphs_cmd = [
            "python", "-m", "src.build_normalized_graphs",
            "--book-id", args.book_id,
            "--log-level", args.log_level,
        ]
        if paths.run_dir is not None:
            graphs_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--graphs-dir", str(paths.graphs_chapters_dir),
            ]
        run(graphs_cmd)

        # ---- Step 7: score interactions ------------------------------ #
        score_cmd = [
            "python", "-m", "src.score_interactions",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            score_cmd += ["--booknlp-root", str(paths.booknlp_book_dir)]
        run(score_cmd)

        # ---- Step 8: build filtered graphs --------------------------- #
        filtered_cmd = [
            "python", "-m", "src.build_filtered_graphs",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            filtered_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--graphs-root", str(paths.graphs_chapters_dir),
            ]
        run(filtered_cmd)

        # ---- Step 9: export BERT dataset ----------------------------- #
        bert_cmd = [
            "python", "-m", "src.export_bert_dataset",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            bert_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--output-dir", str(paths.ml_book_dir),
            ]
        run(bert_cmd)

        # ---- Step 10: quality report --------------------------------- #
        quality_cmd = [
            "python", "-m", "src.build_quality_report",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            quality_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--graphs-root", str(paths.graphs_chapters_dir),
                "--output-dir", str(paths.reports_book_dir),
            ]
        run(quality_cmd)

        # ---- Step 11: suggest aliases -------------------------------- #
        aliases_cmd = [
            "python", "-m", "src.suggest_aliases",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            aliases_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--output-dir", str(paths.reports_book_dir),
            ]
        run(aliases_cmd)

        # ---- Step 12: export tables ---------------------------------- #
        tables_cmd = [
            "python", "-m", "src.export_tables",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            tables_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--reports-root", str(paths.reports_book_dir),
            ]
        run(tables_cmd)

        # ---- Step 13: export interaction reports --------------------- #
        reports_cmd = [
            "python", "-m", "src.export_interaction_reports",
            "--book-id", args.book_id,
        ]
        if paths.run_dir is not None:
            reports_cmd += [
                "--booknlp-root", str(paths.booknlp_book_dir),
                "--output-dir", str(paths.reports_book_dir / "interactions"),
            ]
        run(reports_cmd)

    except RuntimeError as exc:
        pipeline_error = str(exc)
        logger.error("Pipeline failed: %s", pipeline_error)

    # ------------------------------------------------------------------ #
    # Write manifest (run mode only)
    # ------------------------------------------------------------------ #
    if paths.run_dir is not None:
        status = "failed" if pipeline_error else "completed"
        entry = make_book_entry(
            book_id=args.book_id,
            input_path=str(input_path),
            paths=paths,
            status=status,
            error=pipeline_error,
        )
        manifest_path = write_manifest(paths.run_dir, args.run_id, entry)
        logger.info("Manifest written to %s", manifest_path)

    if pipeline_error:
        raise SystemExit(1)

    logger.info("FULL PIPELINE COMPLETED for %s", args.book_id)


if __name__ == "__main__":
    main()
