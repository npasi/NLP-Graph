"""Step 5: book-level inference → GT-style CSV with carry-forward and special values."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

from longformer_pipeline.build_dataset import (
    _CHAPTER_COL_RE,
    _fuzzy_ratio,
    _name_variants,
    _pair_key,
)
from longformer_pipeline.input_builder import (
    add_entity_markers,
    build_longformer_input,
    process_chapter_file,
)
from longformer_pipeline.scorer import LongformerScorer

_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d+)_proper_interactions\.txt$")


def _ensure_src_on_path() -> None:
    this_file = Path(__file__).resolve()
    src_dir = this_file.parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def discover_chapter_files(interactions_dir: str, book_id: str) -> Dict[int, Path]:
    """Map chapter index (0-based, matches GT ``chapter_1`` → 0) to interaction file.

    Resolution order matches ``build_dataset._find_interactions_file`` per chapter.
    """
    root = Path(interactions_dir)
    found: Dict[int, Path] = {}
    dirs_in_order = [
        root / str(book_id),
        root / f"results_{book_id}" / "chapters",
        root / str(book_id) / "chapters",
        root,
    ]
    for d in dirs_in_order:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("chapter_*_proper_interactions.txt")):
            m = _CHAPTER_FILE_RE.match(f.name)
            if not m:
                continue
            k = int(m.group(1))
            if k not in found:
                found[k] = f
    return dict(sorted(found.items()))


def _read_title_author(gt_csv: Optional[str]) -> Tuple[str, str]:
    if not gt_csv:
        return "", ""
    df = pd.read_csv(gt_csv, nrows=1)
    t = str(df.loc[0, "title"]).strip() if "title" in df.columns else ""
    a = str(df.loc[0, "author"]).strip() if "author" in df.columns else ""
    return t, a


def _load_gt_minus10(
    gt_csv: str,
    *,
    n_chapters: int,
) -> List[Tuple[str, str, int]]:
    """(character_1, character_2, chapter_0based) for cells marked -10."""
    df = pd.read_csv(gt_csv)
    out: List[Tuple[str, str, int]] = []
    for _, row in df.iterrows():
        a = str(row["character_1"]).strip()
        b = str(row["character_2"]).strip()
        for col in df.columns:
            m = _CHAPTER_COL_RE.match(str(col).strip())
            if not m:
                continue
            ch = int(m.group(1)) - 1
            if ch < 0 or ch >= n_chapters:
                continue
            try:
                val = float(row[col])
            except Exception:
                continue
            if val == -10.0:
                out.append((a, b, ch))
    return out


def _match_gt_pair_to_row_key(
    char_a: str,
    char_b: str,
    row_keys: List[Tuple[str, str, str, str]],
    *,
    fuzzy_threshold: float = 80.0,
) -> Optional[Tuple[str, str]]:
    """Match GT names to an inference row; ``row_keys`` entries are (pk_a, pk_b, disp_a, disp_b)."""
    target = _pair_key(char_a, char_b)
    for pka, pkb, _, _ in row_keys:
        if (pka, pkb) == target:
            return (pka, pkb)

    a_vars = _name_variants(char_a)
    b_vars = _name_variants(char_b)
    best_score = -1.0
    best: Optional[Tuple[str, str, str, str]] = None
    for pka, pkb, da, db in row_keys:
        pa = _name_variants(da)
        pb = _name_variants(db)
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
            best = (pka, pkb, da, db)
    if best is not None and best_score >= fuzzy_threshold:
        return (best[0], best[1])
    return None


def _reconfigure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows when downstream code prints non-ASCII."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def predict_book(
    book_id: str,
    interactions_dir: str,
    model_dir: str,
    output_csv: str,
    gt_csv: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 8,
    norm: Literal["sigmoid", "clip"] = "sigmoid",
    *,
    fuzzy_threshold: float = 80.0,
    scorer: Optional["LongformerScorer"] = None,
) -> None:
    """Run Longformer on all chapter interaction files for ``book_id`` and write a GT-style CSV.

    Per-chapter scores use ``-1`` when the pair never appears in that chapter's interaction file.
    Carry-forward copies a previous chapter's score in ``[0, 1]`` into later ``-1`` cells.
    If ``gt_csv`` is set, cells marked ``-10`` in that GT are copied into the output (overwriting).
    """
    _reconfigure_stdio_utf8()
    _ensure_src_on_path()

    chapter_files = discover_chapter_files(interactions_dir, book_id)
    if not chapter_files:
        raise FileNotFoundError(
            f"No chapter_*_proper_interactions.txt found under {interactions_dir!r} "
            f"for book_id={book_id!r}"
        )

    max_ch = max(chapter_files.keys())
    n_chapters = max_ch + 1

    print(f"[infer] book_id={book_id} chapters indexed 0..{max_ch} -> n_chapters={n_chapters}")

    if scorer is None:
        scorer = LongformerScorer(model_dir, device=device, norm=norm)

    # pair_key -> list of scores per chapter (init -1)
    chapter_scores: Dict[int, Dict[Tuple[str, str], float]] = {}
    display_for_pk: Dict[Tuple[str, str], Tuple[str, str]] = {}
    prev_cache: Dict[Tuple[str, str], float] = {}

    n_predicted = 0
    for ch in range(n_chapters):
        path = chapter_files.get(ch)
        if path is None:
            chapter_scores[ch] = {}
            print(f"[infer] Chapter {ch}: no file, skipping")
            continue
        print(f"[infer] Chapter {ch}: {path}")
        rows = process_chapter_file(str(path))
        texts: List[str] = []
        pks: List[Tuple[str, str]] = []
        for r in rows:
            pk = _pair_key(r["char_a"], r["char_b"])
            prev = prev_cache.get(pk)
            prev_score = prev if (prev is not None and 0.0 <= prev <= 1.0) else None
            marked = add_entity_markers(r["raw_text"], r["char_a"], r["char_b"])
            lf = build_longformer_input(
                r["char_a"],
                r["char_b"],
                marked,
                prev_score=prev_score,
            )
            texts.append(lf)
            pks.append(pk)
        batch_scores = scorer.score_batch(texts, batch_size=batch_size)
        idx: Dict[Tuple[str, str], float] = {}
        for r, pk, s in zip(rows, pks, batch_scores, strict=True):
            idx[pk] = float(s)
            if 0.0 <= float(s) <= 1.0:
                prev_cache[pk] = float(s)
            if pk not in display_for_pk:
                display_for_pk[pk] = (r["char_a"], r["char_b"])
        chapter_scores[ch] = idx
        n_predicted += sum(1 for s in batch_scores if s >= 0.0)

    all_keys = set()
    for ch in range(n_chapters):
        all_keys.update(chapter_scores.get(ch, {}).keys())

    # Build score matrix [pair][ch]
    matrix: Dict[Tuple[str, str], List[float]] = {}
    for pk in all_keys:
        matrix[pk] = [
            chapter_scores.get(ch, {}).get(pk, -1.0) for ch in range(n_chapters)
        ]

    carried = 0
    for pk in matrix:
        row = matrix[pk]
        for ch in range(1, n_chapters):
            if row[ch] == -1.0:
                prev = row[ch - 1]
                if 0.0 <= prev <= 1.0:
                    row[ch] = prev
                    carried += 1

    n_minus10_applied = 0
    if gt_csv:
        minus_cells = _load_gt_minus10(gt_csv, n_chapters=n_chapters)
        row_key_list = [
            (pk[0], pk[1], display_for_pk[pk][0], display_for_pk[pk][1])
            for pk in sorted(matrix.keys())
        ]
        for ga, gb, ch in minus_cells:
            mk = _match_gt_pair_to_row_key(
                ga, gb, row_key_list, fuzzy_threshold=fuzzy_threshold
            )
            if mk is None:
                print(
                    f"[infer] WARNING: GT -10 for ({ga}, {gb}) ch{ch} - no matching inference row"
                )
                continue
            if mk not in matrix:
                continue
            matrix[mk][ch] = -10.0
            n_minus10_applied += 1

    title, author = _read_title_author(gt_csv)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chap_cols = [f"chapter_{j}" for j in range(1, n_chapters + 1)]
    records: List[Dict] = []
    for pk in sorted(matrix.keys()):
        da, db = display_for_pk.get(pk, (pk[0], pk[1]))
        rec: Dict = {
            "title": title,
            "author": author,
            "character_1": da,
            "character_2": db,
        }
        for j, val in enumerate(matrix[pk], start=1):
            rec[f"chapter_{j}"] = val
        records.append(rec)

    df_out = pd.DataFrame.from_records(records)
    col_order = ["title", "author", "character_1", "character_2"] + chap_cols
    df_out = df_out[[c for c in col_order if c in df_out.columns]]
    df_out.to_csv(out_path, index=False)

    n_minus1 = sum(
        1
        for pk in matrix
        for v in matrix[pk]
        if v == -1.0
    )
    n_minus10_out = sum(
        1 for pk in matrix for v in matrix[pk] if v == -10.0
    )

    print(f"[infer] Wrote {out_path} ({len(records)} pairs × {n_chapters} chapters)")
    print(
        f"[infer] Summary: predicted_cells≈{n_predicted}, carry_forwards={carried}, "
        f"cells==-1: {n_minus1}, cells==-10: {n_minus10_out} "
        f"(GT -10 applied: {n_minus10_applied})"
    )


def main() -> None:
    _ensure_src_on_path()
    ap = argparse.ArgumentParser(description="Longformer book inference → GT-style CSV")
    ap.add_argument("book_id", help="Numeric book id (e.g. 22)")
    ap.add_argument(
        "--interactions-dir",
        default="data/dataset/interactions",
        help="Root containing results_<id>/chapters/ or layouts from build_dataset",
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Fine-tuned model directory (tokenizer + weights)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/predictions/<book_id>_predicted.csv)",
    )
    ap.add_argument(
        "--gt-csv",
        default=None,
        help="Optional GT CSV to copy -10 cells into the prediction",
    )
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--norm",
        choices=("sigmoid", "clip"),
        default="sigmoid",
        help="Must match training (default: sigmoid)",
    )
    args = ap.parse_args()

    out = args.output
    if not out:
        out = str(
            Path("data") / "predictions" / f"{args.book_id}_predicted.csv"
        )

    predict_book(
        book_id=str(args.book_id),
        interactions_dir=args.interactions_dir,
        model_dir=args.model_dir,
        output_csv=out,
        gt_csv=args.gt_csv,
        device=args.device,
        batch_size=args.batch_size,
        norm=args.norm,
    )


if __name__ == "__main__":
    main()
