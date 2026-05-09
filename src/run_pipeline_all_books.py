"""Full pipeline batch runner: split → chapters → BookNLP → step2 → step4.

For each book in the requested split it runs:
  1. split_dynamic(text, N_gt)  → chapter .txt files
  2. run_booknlp_per_chapter    → BookNLP outputs per chapter
  3. step2 (characters)         → characters_by_chapter.json
  4. step4 (sentences per pair) → sentences_by_pair_by_chapter.json

Restartable: each step is skipped if its output already exists.

Usage (Colab example):
    python src/run_pipeline_all_books.py --split train
    python src/run_pipeline_all_books.py --split val
    python src/run_pipeline_all_books.py --split test
    python src/run_pipeline_all_books.py --split all

Optional overrides (useful when running from Google Drive on Colab):
    --data-root /content/drive/MyDrive/book_graph_pipeline/data

Input layouts supported under data-root:

1) Dataset layout (preferred when present):
    dataset/texts/         book .txt files named like "<book_id>_*.txt"
    dataset/ground_truth/  GT .csv files named like "<book_id>_*.csv"

2) Archive layout (legacy fallback):
    archive/corpus/   book .txt files named like "<book_id>_*.txt"
    archive/csv/      GT .csv files named like "<book_id>_*.csv"

Shared structure:
    splits/train_val_test.json
    models/           BookNLP model weights
    output/chapters/  chapter splits (written here)
    output/booknlp/   BookNLP per-chapter output (written here)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.split_dynamic import split_dynamic
from src.step2_character_list_per_chapter import extract_characters_per_chapter
from src.step4_sentences_by_character_pair import extract_sentences_by_pair_by_chapter
from src.run_booknlp_per_chapter import run_booknlp_per_chapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(directory: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Find a unique file in directory whose name starts with prefix and ends with suffix."""
    matches = list(directory.glob(f"{prefix}*{suffix}"))
    return matches[0] if len(matches) == 1 else None


def _count_gt_chapters(csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
    return sum(1 for col in header if re.match(r"^chapter_\d+$", col.strip(), re.IGNORECASE))

def _choose_input_dirs(
    data_root: Path,
    texts_dir: Optional[Path],
    gt_dir: Optional[Path],
) -> tuple[Path, Path]:
    """Pick input directories for texts + GT.

    Priority:
      1) CLI overrides (--texts-dir/--gt-dir)
      2) dataset/ layout if it exists and has files
      3) archive/ layout (legacy)
    """
    if (texts_dir is None) != (gt_dir is None):
        raise ValueError("Use both --texts-dir and --gt-dir together (or neither).")
    if texts_dir is not None and gt_dir is not None:
        return texts_dir, gt_dir

    dataset_texts = data_root / "dataset" / "texts"
    dataset_gt = data_root / "dataset" / "ground_truth"
    if dataset_texts.is_dir() and dataset_gt.is_dir():
        if any(dataset_texts.glob("*.txt")) and any(dataset_gt.glob("*.csv")):
            return dataset_texts, dataset_gt

    return data_root / "archive" / "corpus", data_root / "archive" / "csv"


def _write_chapters(book_id: str, chapters: List[dict], books_root: Path) -> Path:
    """Write chapter dicts to disk as chapter_000.txt, chapter_001.txt, ..."""
    chapters_dir = books_root / book_id / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    for ch in chapters:
        fname = chapters_dir / f"chapter_{int(ch['chapter_id']):03d}.txt"
        fname.write_text(ch["text"], encoding="utf-8")
    (chapters_dir / "chapters.json").write_text(
        json.dumps([
            {"chapter_id": ch["chapter_id"], "title": ch["title"],
             "num_tokens": ch["num_tokens"]}
            for ch in chapters
        ], indent=2),
        encoding="utf-8",
    )
    return chapters_dir


# ---------------------------------------------------------------------------
# Per-book pipeline
# ---------------------------------------------------------------------------

def process_book(
    book_id: str,
    *,
    corpus_dir: Path,
    csv_dir: Path,
    books_root: Path,
    booknlp_output_root: Path,
    model_path: Path,
    model_size: str,
    pipeline: str,
) -> bool:
    """Run the full pipeline for one book. Returns True on success."""

    # --- locate input files ---
    corpus_file = _find_file(corpus_dir, f"{book_id}_", ".txt")
    csv_file    = _find_file(csv_dir,    f"{book_id}_", ".csv")
    if not corpus_file or not csv_file:
        logger.error("[%s] Missing corpus or CSV file — skipping.", book_id)
        return False

    # --- output paths ---
    booknlp_root = booknlp_output_root / book_id
    sentences_json = booknlp_root / "sentences_by_pair_by_chapter.json"
    characters_json = booknlp_root / "characters_by_chapter.json"
    chapters_dir = books_root / book_id / "chapters"

    # Full pipeline already done?
    if sentences_json.exists():
        logger.info("[%s] Already complete — skipping.", book_id)
        return True

    logger.info("[%s] Starting pipeline.", book_id)

    # --- STEP 1: split_dynamic → chapter files ---
    if not chapters_dir.exists() or not list(chapters_dir.glob("chapter_*.txt")):
        logger.info("[%s] Splitting into chapters...", book_id)
        text = corpus_file.read_text(encoding="utf-8", errors="replace")
        n_gt = _count_gt_chapters(csv_file)
        chapters = split_dynamic(text, n_gt)
        _write_chapters(book_id, chapters, books_root)
        logger.info("[%s] Wrote %d chapter files.", book_id, len(chapters))
    else:
        n_chapters = len(list(chapters_dir.glob("chapter_*.txt")))
        logger.info("[%s] Chapter files already exist (%d) — skipping split.", book_id, n_chapters)

    # --- STEP 2: BookNLP per chapter ---
    # Check if all chapters have booknlp output
    chapter_files = sorted(chapters_dir.glob("chapter_*.txt"))
    all_done = all(
        (booknlp_root / f"booknlp_{book_id}_chapter_{int(cf.stem.split('_')[-1]):04d}").exists()
        for cf in chapter_files
    )
    if not all_done:
        logger.info("[%s] Running BookNLP on %d chapters...", book_id, len(chapter_files))
        run_booknlp_per_chapter(
            book_id=book_id,
            chapters_dir=chapters_dir,
            output_root=booknlp_root,
            model_path=model_path,
            model_size=model_size,
            pipeline=pipeline,
        )
    else:
        logger.info("[%s] BookNLP outputs already exist — skipping.", book_id)

    # --- STEP 3: characters_by_chapter.json ---
    if not characters_json.exists():
        logger.info("[%s] Extracting characters...", book_id)
        chars = extract_characters_per_chapter(booknlp_root=booknlp_root)
        characters_json.write_text(
            json.dumps(chars, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        logger.info("[%s] characters_by_chapter.json already exists — skipping.", book_id)

    # --- STEP 4: sentences_by_pair_by_chapter.json ---
    logger.info("[%s] Extracting sentences by pair...", book_id)
    result = extract_sentences_by_pair_by_chapter(
        book_id=book_id,
        booknlp_root=booknlp_root,
        chapters_root=chapters_dir,
    )
    sentences_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[%s] Done. %d chapters written to %s", book_id, len(result), sentences_json)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run full pipeline for a train/val/test split.")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--data-root", default=None,
                   help="Override data root (default: <project>/data). "
                        "Useful on Colab with Google Drive.")
    p.add_argument(
        "--texts-dir",
        default=None,
        help="Override texts directory (expects '<book_id>_*.txt'). If set, you must also set --gt-dir.",
    )
    p.add_argument(
        "--gt-dir",
        default=None,
        help="Override ground-truth directory (expects '<book_id>_*.csv'). If set, you must also set --texts-dir.",
    )
    p.add_argument("--model-size", choices=["small", "big"], default="big")
    p.add_argument("--pipeline", default="entity,quote,supersense,event,coref")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root) if args.data_root else _PROJECT_ROOT / "data"

    corpus_dir, csv_dir = _choose_input_dirs(
        data_root,
        Path(args.texts_dir) if args.texts_dir else None,
        Path(args.gt_dir) if args.gt_dir else None,
    )
    books_root          = data_root / "output" / "chapters"
    booknlp_output_root = data_root / "output" / "booknlp"
    model_path          = data_root / "models"
    split_file          = data_root / "splits" / "train_val_test.json"

    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    if args.split == "all":
        book_ids = split_data["train"] + split_data["val"] + split_data["test"]
    else:
        book_ids = split_data[args.split]

    logger.info("Running pipeline for split=%s (%d books)", args.split, len(book_ids))
    logger.info("Input texts: %s", corpus_dir.resolve())
    logger.info("Input GT:    %s", csv_dir.resolve())

    ok, failed = 0, []
    for i, book_id in enumerate(book_ids, 1):
        logger.info("=== [%d/%d] Book %s ===", i, len(book_ids), book_id)
        try:
            success = process_book(
                book_id,
                corpus_dir=corpus_dir,
                csv_dir=csv_dir,
                books_root=books_root,
                booknlp_output_root=booknlp_output_root,
                model_path=model_path,
                model_size=args.model_size,
                pipeline=args.pipeline,
            )
            if success:
                ok += 1
            else:
                failed.append(book_id)
        except Exception:
            logger.error("[%s] Unhandled error:\n%s", book_id, traceback.format_exc())
            failed.append(book_id)

    logger.info("=== DONE: %d/%d succeeded ===", ok, len(book_ids))
    if failed:
        logger.warning("Failed books: %s", failed)


if __name__ == "__main__":
    main()
