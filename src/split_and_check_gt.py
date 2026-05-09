from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.split_dynamic import split_dynamic
from src.utils.io import data_dir, ensure_dir, write_json, write_text

logger = logging.getLogger(__name__)


def _find_unique(directory: Path, prefix: str, suffix: str) -> Optional[Path]:
    matches = list(directory.glob(f"{prefix}*{suffix}"))
    if len(matches) != 1:
        return None
    return matches[0]


def _count_gt_chapters(csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
    return sum(1 for col in header if re.match(r"^chapter_\d+$", col.strip(), re.IGNORECASE))


def _detect_method(text: str, gt_n: int) -> str:
    # Must mirror split_dynamic’s decision boundaries.
    from src.chapter_splitter import split_into_chapters
    from src.split_dynamic import _strip_gutenberg_header
    from src.step1_split_only import _strip_gutenberg_footer

    clean = _strip_gutenberg_header(_strip_gutenberg_footer(text))
    nat_n = len(split_into_chapters(clean))
    if nat_n == gt_n:
        return "natural"
    if nat_n > gt_n:
        return "grouped"
    return "position"


def split_and_check(
    *,
    book_id: str,
    texts_dir: Path,
    gt_dir: Path,
    out_root: Path,
) -> dict:
    text_path = _find_unique(texts_dir, f"{book_id}_", ".txt")
    gt_path = _find_unique(gt_dir, f"{book_id}_", ".csv")
    if not text_path:
        raise FileNotFoundError(f"Could not find unique text file in {texts_dir} with prefix '{book_id}_'")
    if not gt_path:
        raise FileNotFoundError(f"Could not find unique GT file in {gt_dir} with prefix '{book_id}_'")

    text = text_path.read_text(encoding="utf-8", errors="replace")
    gt_n = _count_gt_chapters(gt_path)
    chapters = split_dynamic(text, gt_n)
    got_n = len(chapters)
    method = _detect_method(text, gt_n)

    book_out = ensure_dir(out_root / book_id)
    chapters_dir = ensure_dir(book_out / "chapters")

    records = []
    for ch in chapters:
        ch_id = int(ch["chapter_id"])
        out_txt = chapters_dir / f"chapter_{ch_id:03d}.txt"
        write_text(out_txt, str(ch["text"]), encoding="utf-8")
        records.append(
            {
                "chapter_id": ch_id,
                "title": str(ch.get("title", "")),
                "num_tokens": int(ch.get("num_tokens", len(str(ch["text"]).split()))),
                "path": str(out_txt),
            }
        )

    write_json(chapters_dir / "chapters.json", records, indent=2)
    report = {
        "book_id": book_id,
        "text_file": text_path.name,
        "gt_file": gt_path.name,
        "gt_chapters": gt_n,
        "got_chapters": got_n,
        "ok": bool(got_n == gt_n),
        "method": method,
        "chapters_dir": str(chapters_dir),
    }
    (book_out / "split_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split a book into exactly N chapters (from GT) and check 1:1 alignment.")
    p.add_argument("--book-id", required=True, help="Numeric book id (e.g., 22).")
    p.add_argument("--texts-dir", default=None, help="Default: data/dataset/texts")
    p.add_argument("--gt-dir", default=None, help="Default: data/dataset/ground_truth")
    p.add_argument("--out-root", default=None, help="Default: data/output/chapters")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    book_id = str(args.book_id)
    texts_dir = Path(args.texts_dir) if args.texts_dir else (data_dir() / "dataset" / "texts")
    gt_dir = Path(args.gt_dir) if args.gt_dir else (data_dir() / "dataset" / "ground_truth")
    out_root = Path(args.out_root) if args.out_root else (data_dir() / "output" / "chapters")

    rep = split_and_check(book_id=book_id, texts_dir=texts_dir, gt_dir=gt_dir, out_root=out_root)
    logger.info(
        "Split check: book=%s ok=%s gt=%d got=%d method=%s",
        rep["book_id"],
        rep["ok"],
        rep["gt_chapters"],
        rep["got_chapters"],
        rep["method"],
    )
    logger.info("Wrote chapters to: %s", rep["chapters_dir"])


if __name__ == "__main__":
    main()

