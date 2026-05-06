from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import logging

from src.utils.gutenberg_cleaner import clean_gutenberg_text

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

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # 🔥 CLEAN AUTOMATICALLY
    clean_path = input_path.parent / f"{input_path.stem}_clean.txt"
    logger.info("Cleaning Gutenberg text...")
    cleaned_input = clean_gutenberg_text(input_path, clean_path)
    logger.info("Using cleaned file: %s", cleaned_input)

    alias_file = ensure_alias_file(args.book_id)

    run([
        "python", "src/step1_split_only.py",
        "--input", str(cleaned_input),
        "--book-id", args.book_id,
        "--debug",
        "--log-level", args.log_level,
    ])

    run([
        "python", "run_booknlp_per_chapter.py",
        "--book-id", args.book_id,
        "--model-size", args.model_size,
        "--pipeline", "entity,quote,supersense,event,coref",
        "--log-level", args.log_level,
    ])

    run([
        "python", "-m", "src.step2b_character_identity",
        "--book-id", args.book_id,
        "--alias-file", str(alias_file),
        "--log-level", args.log_level,
    ])

    run([
        "python", "-m", "src.step3b_normalized_predicates",
        "--book-id", args.book_id,
        "--log-level", args.log_level,
    ])

    run([
        "python", "-m", "src.step4b_normalized_evidence",
        "--book-id", args.book_id,
        "--log-level", args.log_level,
    ])

    run([
        "python", "-m", "src.build_normalized_graphs",
        "--book-id", args.book_id,
        "--log-level", args.log_level,
    ])

    run([
        "python", "-m", "src.score_interactions",
        "--book-id", args.book_id,
    ])

    run([
        "python", "-m", "src.build_filtered_graphs",
        "--book-id", args.book_id,
    ])

    run([
        "python", "-m", "src.export_bert_dataset",
        "--book-id", args.book_id,
    ])

    run([
        "python", "-m", "src.build_quality_report",
        "--book-id", args.book_id,
    ])

    run([
        "python", "-m", "src.suggest_aliases",
        "--book-id", args.book_id,
    ])

    run([
        "python", "-m", "src.export_tables",
        "--book-id", args.book_id,
    ])

    run([
        "python", "-m", "src.export_interaction_reports",
        "--book-id", args.book_id,
    ])

    logger.info("FULL PIPELINE COMPLETED for %s", args.book_id)


if __name__ == "__main__":
    main()
