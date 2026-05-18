"""Batch: one graph PNG per chapter for each model (test/train books)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from longformer_pipeline.plot_affinity_graphs import (
    _cell_in_unit_interval,
    _chapter_cols,
    plot_chapter_graph,
)


def _model_dirs(split: str, pred_root: Path) -> Dict[str, Path]:
    if split == "test":
        return {
            "pretrained_base": pred_root / "pretrained_base",
            "old_1ep_MSE": pred_root / "prevprefix_test_1ep",
            "new_3ep_MSE": pred_root / "prevprefix_test_3ep_mse",
            "pearson_2ep": pred_root / "prevprefix_test_pearson_2ep",
        }
    return {
        "pretrained_base": pred_root / "pretrained_base",
        "old_1ep_MSE": pred_root / "prevprefix_full_1ep",
        "new_3ep_MSE": pred_root / "prevprefix_full_3ep_mse",
        "pearson_2ep": pred_root / "prevprefix_full_pearson_2ep",
    }


def run_book(
    *,
    book_id: str,
    chapters: List[str],
    out_root: Path,
    models: List[Tuple[str, Path]],
    top_k_edges: int,
    max_nodes: int,
    seed: int,
    min_edges: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "book_id": book_id,
        "chapters": chapters,
        "sources": {},
    }

    for model_name, pred_csv in models:
        if not pred_csv.is_file():
            print(f"[all-ch] SKIP {book_id} {model_name}: missing {pred_csv}")
            summary["sources"][model_name] = {"skipped": True, "reason": "missing csv"}
            continue

        out_dir = out_root / model_name / "per_chapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        skipped: List[str] = []
        df = pd.read_csv(pred_csv)

        for col in chapters:
            out_png = out_dir / f"graph_pred_{col}.png"
            n_valid = 0
            if col in df.columns:
                for _, row in df.iterrows():
                    if _cell_in_unit_interval(row[col]):
                        n_valid += 1
            if n_valid < min_edges:
                skipped.append(col)
                continue
            ok = plot_chapter_graph(
                pred_csv,
                col,
                out_png,
                title_label=model_name,
                file_prefix="graph_pred",
                top_k_edges=top_k_edges,
                max_nodes=max_nodes,
                seed=seed,
            )
            if ok:
                written.append(col)
            else:
                skipped.append(col)

        summary["sources"][model_name] = {
            "csv": str(pred_csv),
            "out_dir": str(out_dir),
            "written": written,
            "skipped": skipped,
            "n_written": len(written),
        }
        print(f"[all-ch] {book_id} {model_name}: {len(written)}/{len(chapters)} chapters -> {out_dir}")

    meta_path = out_root / "per_chapter_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-ids", default="95,99,110")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--gt-dir", default="data/dataset/ground_truth")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--out-root", default="data/share_with_friend_clean/graphs")
    ap.add_argument("--top-k-edges", type=int, default=10)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--min-edges", type=int, default=2, help="Min valid pair scores in chapter to plot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys (default: all). E.g. pearson_2ep",
    )
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    out_base = Path(args.out_root) / args.split
    model_dirs = _model_dirs(args.split, Path(args.pred_root))
    if args.models:
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        model_dirs = {k: v for k, v in model_dirs.items() if k in want}

    for bid in [x.strip() for x in args.book_ids.split(",") if x.strip()]:
        gt_csv = next(gt_dir.glob(f"{bid}_*.csv"), None)
        if gt_csv is None:
            print(f"[all-ch] SKIP book {bid}: no GT csv")
            continue
        gt_df = pd.read_csv(gt_csv)
        chapters = _chapter_cols(gt_df)
        models = [(n, d / f"{bid}_predicted.csv") for n, d in model_dirs.items()]
        run_book(
            book_id=bid,
            chapters=chapters,
            out_root=out_base / bid,
            models=models,
            top_k_edges=int(args.top_k_edges),
            max_nodes=int(args.max_nodes),
            seed=int(args.seed),
            min_edges=int(args.min_edges),
        )


if __name__ == "__main__":
    main()
