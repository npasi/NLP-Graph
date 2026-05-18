"""Plot two-chapter predicted graphs at the GT transition with largest mean |Δ| on top-K union edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from longformer_pipeline.plot_affinity_graphs import (
    _chapter_cols,
    _chapter_weights,
    _top_edges_union,
    plot_two_chapters,
)


def best_transition_topk(df: pd.DataFrame, top_k: int = 10) -> Tuple[str, str, int, float, float]:
    ch = _chapter_cols(df)
    best: Tuple[float, float, str, str, int] | None = None
    for i in range(len(ch) - 1):
        ca, cb = ch[i], ch[i + 1]
        wa, wb = _chapter_weights(df, ca), _chapter_weights(df, cb)
        edges = _top_edges_union(wa, wb, top_k)
        deltas = [abs(b - a) for _k, a, b in edges if np.isfinite(a) and np.isfinite(b)]
        if not deltas:
            continue
        mean_d = float(np.mean(deltas))
        max_d = float(np.max(deltas))
        if best is None or mean_d > best[0]:
            best = (mean_d, max_d, ca, cb, i)
    if best is None:
        raise ValueError("No valid transition")
    mean_d, max_d, ca, cb, t0 = best
    return ca, cb, t0, mean_d, max_d


def _test_models(pred_root: Path, bid: str) -> List[Tuple[str, Path]]:
    return [
        ("pretrained_base", pred_root / "pretrained_base" / f"{bid}_predicted.csv"),
        ("old_1ep_MSE", pred_root / "prevprefix_test_1ep" / f"{bid}_predicted.csv"),
        ("new_3ep_MSE", pred_root / "prevprefix_test_3ep_mse" / f"{bid}_predicted.csv"),
        ("pearson_2ep", pred_root / "prevprefix_test_pearson_2ep" / f"{bid}_predicted.csv"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-ids", default="95,99,110")
    ap.add_argument("--gt-dir", default="data/dataset/ground_truth")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--out-root", default="data/share_with_friend_clean/graphs/test")
    ap.add_argument("--subdir", default="two_chapters_max_change")
    ap.add_argument("--top-k-edges", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_root = Path(args.pred_root)
    summary: Dict[str, dict] = {}

    for bid in [x.strip() for x in args.book_ids.split(",") if x.strip()]:
        gt_csv = next(gt_dir.glob(f"{bid}_*.csv"))
        gt_df = pd.read_csv(gt_csv)
        ca, cb, t0, mean_d, max_d = best_transition_topk(gt_df, int(args.top_k_edges))
        book_root = Path(args.out_root) / bid
        print(f"\n[max-change] book {bid}: {ca} -> {cb}  mean|Δ|={mean_d:.4f}  max|Δ|={max_d:.4f}")

        book_summary = {
            "book_id": bid,
            "gt_csv": str(gt_csv),
            "chapter_a": ca,
            "chapter_b": cb,
            "mean_abs_delta_topk": mean_d,
            "max_abs_delta_topk": max_d,
            "transition_index_0based": t0,
            "models": {},
        }

        for name, pred_csv in _test_models(pred_root, bid):
            if not pred_csv.is_file():
                print(f"  SKIP {name}: missing {pred_csv}")
                continue
            out_dir = book_root / name / args.subdir
            plot_two_chapters(
                pred_csv, out_dir, col_a=ca, col_b=cb,
                file_prefix="graph_pred", title_label=name,
                top_k_edges=int(args.top_k_edges), seed=int(args.seed),
                layout_df=gt_df,
            )
            book_summary["models"][name] = {
                "pred_csv": str(pred_csv),
                "out_dir": str(out_dir),
                "png_a": str(out_dir / f"graph_pred_{ca}.png"),
                "png_b": str(out_dir / f"graph_pred_{cb}.png"),
            }
            print(f"  {name} -> {out_dir}")

        meta_path = book_root / f"{args.subdir}_meta.json"
        meta_path.write_text(json.dumps(book_summary, indent=2), encoding="utf-8")
        summary[bid] = book_summary

    index_path = Path(args.out_root) / "max_change_transitions_index.json"
    index_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[index] {index_path}")


if __name__ == "__main__":
    main()
