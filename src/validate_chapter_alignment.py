"""Validate that split_dynamic produces exactly N chapters for all 116 books.

Outputs a console table and saves data/output/alignment_report.json.

Usage:
    python src/validate_chapter_alignment.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.split_dynamic import split_dynamic

CORPUS_DIR = _PROJECT_ROOT / "data" / "archive" / "corpus"
CSV_DIR    = _PROJECT_ROOT / "data" / "archive" / "csv"
REPORT_OUT = _PROJECT_ROOT / "data" / "output" / "alignment_report.json"


def _book_id(filename: str) -> Optional[str]:
    m = re.match(r"^(\d+)_", filename)
    return m.group(1) if m else None


def _count_gt_chapters(csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
    return sum(1 for col in header if re.match(r"^chapter_\d+$", col.strip(), re.IGNORECASE))


def main() -> None:
    corpus_by_id: Dict[str, Path] = {}
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        bid = _book_id(p.name)
        if bid:
            corpus_by_id[bid] = p

    csv_by_id: Dict[str, Path] = {}
    for p in sorted(CSV_DIR.glob("*.csv")):
        bid = _book_id(p.name)
        if bid:
            csv_by_id[bid] = p

    all_ids = sorted(set(corpus_by_id) | set(csv_by_id), key=lambda x: int(x))

    results: List[dict] = []
    counts = {"ok": 0, "error": 0, "no_corpus": 0, "no_csv": 0}

    print(f"\n{'ID':>4}  {'GT':>4}  {'Got':>4}  {'Method':<12}  Book")
    print("-" * 90)

    for bid in all_ids:
        if bid not in corpus_by_id:
            print(f"{bid:>4}  {'?':>4}  {'?':>4}  {'NO_CORPUS':<12}")
            counts["no_corpus"] += 1
            results.append({"book_id": bid, "status": "no_corpus"})
            continue
        if bid not in csv_by_id:
            print(f"{bid:>4}  {'?':>4}  {'?':>4}  {'NO_CSV':<12}")
            counts["no_csv"] += 1
            results.append({"book_id": bid, "status": "no_csv"})
            continue

        gt_n = _count_gt_chapters(csv_by_id[bid])
        text  = corpus_by_id[bid].read_text(encoding="utf-8", errors="replace")
        chapters = split_dynamic(text, gt_n)
        got_n = len(chapters)

        # Detect which strategy was used (for the report)
        from src.chapter_splitter import split_into_chapters
        from src.split_dynamic import _strip_gutenberg_header
        from src.step1_split_only import _strip_gutenberg_footer
        clean = _strip_gutenberg_header(_strip_gutenberg_footer(text))
        nat_n = len(split_into_chapters(clean))
        if nat_n == gt_n:
            method = "natural"
        elif nat_n > gt_n:
            method = "grouped"
        else:
            method = "position"

        ok = got_n == gt_n
        status = "OK" if ok else "ERROR"
        counts["ok" if ok else "error"] += 1
        print(f"{bid:>4}  {gt_n:>4}  {got_n:>4}  {method:<12}  {corpus_by_id[bid].stem}")
        results.append({
            "book_id":        bid,
            "status":         status,
            "method":         method,
            "gt_chapters":    gt_n,
            "natural_chapters": nat_n,
            "got_chapters":   got_n,
            "corpus_file":    corpus_by_id[bid].name,
            "csv_file":       csv_by_id[bid].name,
        })

    print("-" * 90)
    print(f"OK={counts['ok']}  ERROR={counts['error']}  "
          f"NO_CORPUS={counts['no_corpus']}  NO_CSV={counts['no_csv']}\n")

    methods = {}
    for r in results:
        m = r.get("method", "n/a")
        methods[m] = methods.get(m, 0) + 1
    print("Methods used:", "  ".join(f"{k}={v}" for k, v in sorted(methods.items())))

    REPORT_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {REPORT_OUT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # suppress splitter INFO chatter
    main()
