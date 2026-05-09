"""Evaluate chapter splitting on raw books without running BookNLP.

Usage:
    python scripts/evaluate_splitter_only.py --run-id test_splitter_fix_v1

Cleans Gutenberg text into the run directory, runs the splitter, writes
chapters.json and split_debug.json, then prints a compact table.

Default evaluation list:
    data/raw/9_The_Great_Gatsby_F._Scott_Fitzgerald.txt
    data/raw/23_Dr._Jekyll_and_Mr._Hyde_Robert_Louis_Stevenson.txt
    data/raw/102_Madame_Bovary_Gustave_Flaubert.txt
    data/raw/3_Heart_of_Darkness_Joseph_Conrad.txt
    data/raw/19_Frankenstein_Mary_Shelley.txt
    data/raw/68_The_Trial_Franz_Kafka.txt

Pass --book-id and --input to add a custom book, or use only that book.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.chapter_splitter import split_into_chapters_with_debug
from src.utils.gutenberg_cleaner import clean_gutenberg_text

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s: %(message)s")

_DEFAULT_BOOKS = [
    ("gatsby",       "data/raw/9_The_Great_Gatsby_F._Scott_Fitzgerald.txt"),
    ("jekyll",       "data/raw/23_Dr._Jekyll_and_Mr._Hyde_Robert_Louis_Stevenson.txt"),
    ("bovary",       "data/raw/102_Madame_Bovary_Gustave_Flaubert.txt"),
    ("heart",        "data/raw/3_Heart_of_Darkness_Joseph_Conrad.txt"),
    ("frankenstein", "data/raw/19_Frankenstein_Mary_Shelley.txt"),
    ("trial",        "data/raw/68_The_Trial_Franz_Kafka.txt"),
]


def _clean_text(input_path: Path, run_dir: Path) -> str:
    clean_dir = run_dir / "cache" / "cleaned_texts"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / f"{input_path.stem}_clean.txt"
    if not clean_path.exists():
        clean_gutenberg_text(input_path, clean_path)
    return clean_path.read_text(encoding="utf-8", errors="replace")


def _evaluate_book(book_id: str, input_path: Path, run_dir: Path) -> dict:
    text = _clean_text(input_path, run_dir)
    chapters, debug = split_into_chapters_with_debug(text)

    out_dir = run_dir / "books" / book_id / "chapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for ch in chapters:
        out_txt = out_dir / f"chapter_{ch['chapter_id']:03d}.txt"
        out_txt.write_text(ch["text"], encoding="utf-8")
        records.append({
            "chapter_id": ch["chapter_id"],
            "title": ch["title"],
            "num_words": ch["num_tokens"],
            "short": ch["short"],
            "raw_heading": ch["raw_heading"],
            "path": str(out_txt),
        })

    (out_dir / "chapters.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "split_debug.json").write_text(
        json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "book_id": book_id,
        "chapter_count": len(chapters),
        "min_words": min((ch["num_tokens"] for ch in chapters), default=0),
        "max_words": max((ch["num_tokens"] for ch in chapters), default=0),
        "zero_word_sections": sum(1 for ch in chapters if ch["num_tokens"] == 0),
        "short_sections": sum(1 for ch in chapters if ch["short"] and ch["num_tokens"] > 0),
        "winner": debug.get("winner", {}).get("name", "?"),
        "first_titles": [ch["title"] for ch in chapters[:3]],
    }


def _print_table(results: list[dict]) -> None:
    hdr = f"{'book_id':<16} {'ch':>4} {'min':>6} {'max':>7} {'zero':>5} {'short':>6} {'winner':<12} first_titles"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        titles_str = " | ".join(r["first_titles"][:2])
        print(
            f"{r['book_id']:<16} {r['chapter_count']:>4} "
            f"{r['min_words']:>6} {r['max_words']:>7} "
            f"{r['zero_word_sections']:>5} {r['short_sections']:>6} "
            f"{(r.get('winner') or 'fallback'):<12} {titles_str}"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate splitter on real books (no BookNLP).")
    parser.add_argument("--run-id", default="test_splitter_eval", help="Run identifier.")
    parser.add_argument("--output-root", default="outputs/runs", help="Output root directory.")
    parser.add_argument("--book-id", default=None, help="Evaluate only this book ID.")
    parser.add_argument("--input", default=None, help="Path to raw .txt for --book-id.")
    args = parser.parse_args(argv)

    run_dir = Path(args.output_root) / args.run_id

    if args.book_id and args.input:
        books = [(args.book_id, args.input)]
    elif args.book_id:
        # Look it up in the default list.
        books = [(bid, p) for bid, p in _DEFAULT_BOOKS if bid == args.book_id]
        if not books:
            print(f"Book ID '{args.book_id}' not in default list. Use --input to specify path.")
            sys.exit(1)
    else:
        books = _DEFAULT_BOOKS

    results = []
    for book_id, input_str in books:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"  SKIP {book_id}: file not found ({input_path})")
            continue
        print(f"Processing {book_id} ...")
        try:
            r = _evaluate_book(book_id, input_path, run_dir)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR {book_id}: {exc}")

    print()
    if results:
        _print_table(results)
    print(f"\nOutput: {run_dir}/books/<book_id>/chapters/")


if __name__ == "__main__":
    main()
