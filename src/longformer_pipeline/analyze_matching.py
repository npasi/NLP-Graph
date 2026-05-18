"""Utility: measure how effective fuzzy matching is in Step 3 (GT↔interactions)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from longformer_pipeline.build_dataset import (
    _book_id_from_filename,
    _CHAPTER_COL_RE,
    _find_interactions_file,
    _fuzzy_ratio,
    _name_variants,
    _pair_key,
)
from longformer_pipeline.input_builder import process_chapter_file


def _ensure_src_on_path() -> None:
    this_file = Path(__file__).resolve()
    src_dir = this_file.parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _reconfigure_stdio_utf8() -> None:
    # input_builder prints non-ascii; avoid Windows cp1252 crashes
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _load_gt_cells(csv_path: str) -> List[Dict]:
    """Return GT cells (pair,chapter,val) for values in [0,1] only."""
    df = pd.read_csv(csv_path)
    chapter_cols = [c for c in df.columns if _CHAPTER_COL_RE.match(str(c).strip())]
    out: List[Dict] = []
    for _, row in df.iterrows():
        a = str(row["character_1"]).strip()
        b = str(row["character_2"]).strip()
        for col in chapter_cols:
            m = _CHAPTER_COL_RE.match(str(col).strip())
            if not m:
                continue
            ch = int(m.group(1)) - 1
            try:
                val = float(row[col])
            except Exception:
                continue
            if 0.0 <= val <= 1.0:
                out.append({"char_a": a, "char_b": b, "chapter": ch, "score": val})
    return out


def _build_chapter_index(
    *,
    interactions_dir: str,
    book_id: str,
    chapters: List[int],
) -> Tuple[Dict[int, List[Dict]], Dict[int, Dict[Tuple[str, str], Dict]]]:
    chapter_pairs: Dict[int, List[Dict]] = {}
    chapter_index: Dict[int, Dict[Tuple[str, str], Dict]] = {}

    for ch in chapters:
        fpath = _find_interactions_file(
            interactions_dir=interactions_dir, book_id=book_id, chapter=ch
        )
        if fpath is None:
            chapter_pairs[ch] = []
            chapter_index[ch] = {}
            continue
        pairs = process_chapter_file(str(fpath))
        chapter_pairs[ch] = pairs
        idx: Dict[Tuple[str, str], Dict] = {}
        for p in pairs:
            idx[_pair_key(p["char_a"], p["char_b"])] = p
        chapter_index[ch] = idx

    return chapter_pairs, chapter_index


def analyze_book(
    *,
    gt_csv: str,
    interactions_dir: str,
    book_id: Optional[str] = None,
    fuzzy_threshold: float = 80.0,
) -> Dict:
    """Measure exact vs fuzzy vs unmatched for one GT CSV."""
    _reconfigure_stdio_utf8()
    gt_cells = _load_gt_cells(gt_csv)
    if not gt_cells:
        raise ValueError(f"No valid [0,1] GT cells in {gt_csv}")

    if book_id is None:
        book_id = _book_id_from_filename(Path(gt_csv))

    chapters = sorted({int(x["chapter"]) for x in gt_cells})
    chapter_pairs, chapter_index = _build_chapter_index(
        interactions_dir=interactions_dir, book_id=str(book_id), chapters=chapters
    )

    counts = Counter()
    fuzzy_best_scores: List[float] = []
    unmatched_examples: List[Dict] = []

    for ex in gt_cells:
        ch = int(ex["chapter"])
        a = str(ex["char_a"])
        b = str(ex["char_b"])

        # 1) exact normalized pair match
        if _pair_key(a, b) in chapter_index.get(ch, {}):
            counts["exact"] += 1
            continue

        # 2) fuzzy match against same-chapter observed pairs
        best_score = -1.0
        best_pair = None
        a_vars = _name_variants(a)
        b_vars = _name_variants(b)
        for p in chapter_pairs.get(ch, []):
            pa = _name_variants(p["char_a"])
            pb = _name_variants(p["char_b"])

            best_here = -1.0
            for av in a_vars:
                for bv in b_vars:
                    for pav in pa:
                        for pbv in pb:
                            s1 = (_fuzzy_ratio(av, pav) + _fuzzy_ratio(bv, pbv)) / 2.0
                            s2 = (_fuzzy_ratio(av, pbv) + _fuzzy_ratio(bv, pav)) / 2.0
                            best_here = max(best_here, s1, s2)
            if best_here > best_score:
                best_score = best_here
                best_pair = p

        if best_pair is not None and best_score >= fuzzy_threshold:
            counts["fuzzy"] += 1
            fuzzy_best_scores.append(best_score)
        else:
            counts["unmatched"] += 1
            if len(unmatched_examples) < 50:
                unmatched_examples.append(
                    {"chapter": ch, "char_a": a, "char_b": b, "best_fuzzy": best_score}
                )

    total = len(gt_cells)
    out = {
        "book_id": str(book_id),
        "gt_csv": str(gt_csv),
        "n_cells": total,
        "exact": int(counts["exact"]),
        "fuzzy": int(counts["fuzzy"]),
        "unmatched": int(counts["unmatched"]),
        "exact_rate": float(counts["exact"] / total),
        "fuzzy_rate": float(counts["fuzzy"] / total),
        "unmatched_rate": float(counts["unmatched"] / total),
        "fuzzy_best_score_mean": float(sum(fuzzy_best_scores) / len(fuzzy_best_scores))
        if fuzzy_best_scores
        else None,
        "fuzzy_best_score_min": float(min(fuzzy_best_scores)) if fuzzy_best_scores else None,
        "fuzzy_best_score_max": float(max(fuzzy_best_scores)) if fuzzy_best_scores else None,
        "unmatched_examples": unmatched_examples,
    }
    return out


def analyze_dir(
    *,
    gt_dir: str,
    interactions_dir: str,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    fuzzy_threshold: float = 80.0,
) -> Dict:
    _reconfigure_stdio_utf8()
    gt_paths = sorted(Path(gt_dir).glob("*.csv"))
    if not gt_paths:
        raise FileNotFoundError(f"No GT CSVs found in {gt_dir}")

    per_book: List[Dict] = []
    for p in gt_paths:
        try:
            per_book.append(
                analyze_book(
                    gt_csv=str(p),
                    interactions_dir=interactions_dir,
                    fuzzy_threshold=fuzzy_threshold,
                )
            )
        except Exception as e:
            per_book.append(
                {
                    "book_id": None,
                    "gt_csv": str(p),
                    "error": str(e),
                }
            )

    # Aggregate weighted by number of GT cells per book
    tot_cells = sum(int(b.get("n_cells", 0) or 0) for b in per_book if "n_cells" in b)
    agg = defaultdict(int)
    for b in per_book:
        if "n_cells" not in b:
            continue
        agg["n_cells"] += int(b["n_cells"])
        agg["exact"] += int(b["exact"])
        agg["fuzzy"] += int(b["fuzzy"])
        agg["unmatched"] += int(b["unmatched"])

    summary = {
        "gt_dir": gt_dir,
        "interactions_dir": interactions_dir,
        "fuzzy_threshold": fuzzy_threshold,
        "n_books": len(gt_paths),
        "n_cells": int(agg["n_cells"]),
        "exact": int(agg["exact"]),
        "fuzzy": int(agg["fuzzy"]),
        "unmatched": int(agg["unmatched"]),
        "exact_rate": float(agg["exact"] / tot_cells) if tot_cells else None,
        "fuzzy_rate": float(agg["fuzzy"] / tot_cells) if tot_cells else None,
        "unmatched_rate": float(agg["unmatched"] / tot_cells) if tot_cells else None,
        "per_book": per_book,
    }

    if output_json:
        outp = Path(output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if output_csv:
        rows = []
        for b in per_book:
            if "n_cells" not in b:
                continue
            rows.append(
                {
                    "book_id": b["book_id"],
                    "gt_csv": Path(b["gt_csv"]).name,
                    "n_cells": b["n_cells"],
                    "exact": b["exact"],
                    "fuzzy": b["fuzzy"],
                    "unmatched": b["unmatched"],
                    "exact_rate": b["exact_rate"],
                    "fuzzy_rate": b["fuzzy_rate"],
                    "unmatched_rate": b["unmatched_rate"],
                    "fuzzy_best_score_mean": b["fuzzy_best_score_mean"],
                }
            )
        df = pd.DataFrame(rows)
        outp = Path(output_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)

    return summary


def main() -> None:
    _ensure_src_on_path()
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser(description="Analyze GT↔interactions matching effectiveness")
    ap.add_argument("--gt-csv", default=None, help="Analyze one GT CSV")
    ap.add_argument(
        "--gt-dir",
        default="data/dataset/ground_truth",
        help="Analyze all GT CSVs in this dir",
    )
    ap.add_argument(
        "--interactions-dir",
        default="data/dataset/interactions",
        help="Root of interaction results (e.g. data/dataset/interactions)",
    )
    ap.add_argument("--book-id", default=None, help="Optional book_id override")
    ap.add_argument("--threshold", type=float, default=80.0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    if args.gt_csv:
        res = analyze_book(
            gt_csv=args.gt_csv,
            interactions_dir=args.interactions_dir,
            book_id=args.book_id,
            fuzzy_threshold=args.threshold,
        )
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "unmatched_examples"}, indent=2))
        return

    res = analyze_dir(
        gt_dir=args.gt_dir,
        interactions_dir=args.interactions_dir,
        output_json=args.out_json,
        output_csv=args.out_csv,
        fuzzy_threshold=args.threshold,
    )
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("per_book",)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

