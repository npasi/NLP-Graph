from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chapter_splitter import split_into_chapters
from src.chapter_splitter import split_into_chapters_with_debug
from src.utils.io import data_dir, ensure_dir, safe_slug, write_json, write_text

logger = logging.getLogger(__name__)


def split_only(
    *,
    input_path: Path,
    book_id: Optional[str] = None,
    max_chapters: Optional[int] = None,
    write_debug: bool = False,
) -> Path:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    bid = book_id or safe_slug(input_path.stem)

    out_root = ensure_dir(data_dir() / "books" / bid / "chapters")
    if write_debug:
        chapters, debug = split_into_chapters_with_debug(text)
    else:
        chapters = split_into_chapters(text)
        debug = None
    if max_chapters is not None:
        chapters = chapters[: int(max_chapters)]

    records: List[Dict[str, Any]] = []
    for ch in chapters:
        ch_id = int(ch["chapter_id"])
        ch_text = ch["text"]
        out_txt = out_root / f"chapter_{ch_id:03d}.txt"
        write_text(out_txt, ch_text, encoding="utf-8")
        records.append(
            {
                "chapter_id": ch_id,
                "title": ch.get("title", ""),
                "num_words": int(ch.get("num_tokens", len(ch_text.split()))),
                "short": bool(ch.get("short", False)),
                "raw_heading": ch.get("raw_heading", ""),
                "path": str(out_txt),
            }
        )

    out_json = out_root / "chapters.json"
    write_json(out_json, records, indent=2)
    if write_debug and debug is not None:
        write_json(out_root / "split_debug.json", debug, indent=2)

    logger.info("SPLIT ONLY: input=%s", input_path)
    logger.info("book_id=%s chapters=%d", bid, len(records))
    logger.info("First 5: %s", [(r["chapter_id"], r["title"], r["num_words"]) for r in records[:5]])
    if len(records) > 5:
        logger.info("Last 5: %s", [(r["chapter_id"], r["title"], r["num_words"]) for r in records[-5:]])
    logger.info("Output: %s", out_root)
    return out_json


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chapter splitter ONLY (no parsing/cleaning).")
    p.add_argument("--input", required=True, help="Path to an already-prepared .txt file.")
    p.add_argument("--book-id", default=None, help="Optional output folder name under data/books/<book-id>/")
    p.add_argument("--max-chapters", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="Write split_debug.json with detector diagnostics.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    split_only(
        input_path=Path(args.input),
        book_id=args.book_id,
        max_chapters=args.max_chapters,
        write_debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()

