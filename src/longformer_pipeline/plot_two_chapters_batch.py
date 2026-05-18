"""Batch: predicted affinity graphs for two consecutive chapters (transition from GT CSV)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from longformer_pipeline.plot_affinity_graphs import (
    _best_transition,
    _chapter_cols,
    plot_two_chapters,
)


def run_book(
    *,
    book_id: str,
    gt_csv: Path,
    out_root: Path,
    models: list[tuple[str, Path]],
    top_k_edges: int,
    seed: int,
) -> None:
    gt_df = pd.read_csv(gt_csv)
    ch = _chapter_cols(gt_df)
    t0 = _best_transition(gt_df)
    if t0 is None:
        raise ValueError(f"No valid transition for book {book_id}")
    col_a, col_b = ch[t0], ch[t0 + 1]

    for model_name, pred_csv in models:
        if not pred_csv.is_file():
            print(f"[batch] SKIP {book_id} {model_name}: missing {pred_csv}")
            continue
        out_dir = out_root / model_name / "two_chapters_change"
        meta = plot_two_chapters(
            pred_csv,
            out_dir,
            col_a=col_a,
            col_b=col_b,
            file_prefix="graph_pred",
            title_label=model_name,
            top_k_edges=top_k_edges,
            seed=seed,
            layout_df=gt_df,
        )
        meta["book_id"] = book_id
        meta["layout_csv"] = str(gt_csv)
        meta["pred_csv"] = str(pred_csv)
        meta["transition_index_0based"] = int(t0)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[batch] {book_id} {model_name} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-ids", default="95,99,110")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--gt-dir", default="data/dataset/ground_truth")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument(
        "--out-root",
        default="data/share_with_friend_clean/graphs",
    )
    ap.add_argument("--top-k-edges", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys (default: all). E.g. pearson_2ep",
    )
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    out_base = Path(args.out_root) / args.split

    if args.split == "test":
        model_dirs = {
            "pretrained_base": Path(args.pred_root) / "pretrained_base",
            "old_1ep_MSE": Path(args.pred_root) / "prevprefix_test_1ep",
            "new_3ep_MSE": Path(args.pred_root) / "prevprefix_test_3ep_mse",
            "pearson_2ep": Path(args.pred_root) / "prevprefix_test_pearson_2ep",
        }
    else:
        model_dirs = {
            "pretrained_base": Path(args.pred_root) / "pretrained_base",
            "old_1ep_MSE": Path(args.pred_root) / "prevprefix_full_1ep",
            "new_3ep_MSE": Path(args.pred_root) / "prevprefix_full_3ep_mse",
            "pearson_2ep": Path(args.pred_root) / "prevprefix_full_pearson_2ep",
        }
    if getattr(args, "models", None):
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        model_dirs = {k: v for k, v in model_dirs.items() if k in want}

    for bid in [x.strip() for x in args.book_ids.split(",") if x.strip()]:
        gt_csv = next(gt_dir.glob(f"{bid}_*.csv"), None)
        if gt_csv is None:
            print(f"[batch] SKIP book {bid}: no GT csv")
            continue
        models = [(n, d / f"{bid}_predicted.csv") for n, d in model_dirs.items()]
        run_book(
            book_id=bid,
            gt_csv=gt_csv,
            out_root=out_base / bid,
            models=models,
            top_k_edges=int(args.top_k_edges),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
