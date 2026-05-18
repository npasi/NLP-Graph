"""Diagnose whether the fine-tuned model compresses chapter-to-chapter score variation.

For each book we build aligned matrices (predicted vs GT) over the same characters
and chapters. Then we compute, only on cells where BOTH GT_t and GT_{t+1} are valid
(i.e. in [0,1], not -1/-10), the absolute delta |x_{t+1} - x_t| for both predicted
and GT. This tells us if the model is producing flatter trajectories than reality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def _reconfigure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _chapter_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        s = str(c).strip().lower()
        if s.startswith("chapter_"):
            try:
                int(s.split("_", 1)[1])
                cols.append(c)
            except Exception:
                pass
    cols.sort(key=lambda x: int(str(x).strip().split("_", 1)[1]))
    return cols


def _key(a: str, b: str) -> Tuple[str, str]:
    a, b = str(a).strip(), str(b).strip()
    return (a, b) if a <= b else (b, a)


def _load_matrix(path: Path) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str]]:
    df = pd.read_csv(path)
    if "character_1" not in df.columns or "character_2" not in df.columns:
        raise ValueError(f"Missing character columns in {path}")
    ch_cols = _chapter_cols(df)
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for _, row in df.iterrows():
        k = _key(row["character_1"], row["character_2"])
        vals = row[ch_cols].to_numpy(dtype=float)
        out[k] = vals
    return out, ch_cols


def diagnose_book(
    pred_path: Path, gt_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (gt_deltas, pred_deltas, gt_vals, pred_vals, n_pairs).

    Deltas are computed on transitions where BOTH gt_t and gt_{t+1} are in [0,1].
    Values are computed on cells where gt is in [0,1].
    """
    pred_map, pred_cols = _load_matrix(pred_path)
    gt_map, gt_cols = _load_matrix(gt_path)
    common_keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    n_chapters = min(len(pred_cols), len(gt_cols))
    if n_chapters < 2 or not common_keys:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    gt_deltas: List[float] = []
    pred_deltas: List[float] = []
    gt_vals: List[float] = []
    pred_vals: List[float] = []
    for k in common_keys:
        g = gt_map[k][:n_chapters]
        p = pred_map[k][:n_chapters]
        valid_cell = (g >= 0) & (g <= 1) & np.isfinite(g) & np.isfinite(p) & (p >= 0)
        gt_vals.extend(g[valid_cell].tolist())
        pred_vals.extend(p[valid_cell].tolist())
        for t in range(n_chapters - 1):
            gA, gB = g[t], g[t + 1]
            pA, pB = p[t], p[t + 1]
            ok = (
                np.isfinite(gA)
                and np.isfinite(gB)
                and 0 <= gA <= 1
                and 0 <= gB <= 1
                and np.isfinite(pA)
                and np.isfinite(pB)
                and pA >= 0
                and pB >= 0
            )
            if ok:
                gt_deltas.append(abs(gB - gA))
                pred_deltas.append(abs(pB - pA))
    return (
        np.array(gt_deltas),
        np.array(pred_deltas),
        np.array(gt_vals),
        np.array(pred_vals),
        len(common_keys),
    )


def diagnose(predictions_dir: str, gt_dir: str, output_dir: str) -> None:
    pred_root = Path(predictions_dir)
    gt_root = Path(gt_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    gt_files = list(gt_root.glob("*_affinity*.csv"))
    gt_by_id: Dict[str, Path] = {}
    for p in gt_files:
        bid = p.stem.split("_", 1)[0]
        gt_by_id.setdefault(bid, p)

    pred_files = sorted(pred_root.glob("*_predicted.csv"))
    rows: List[Dict[str, float]] = []
    all_gt_deltas: List[np.ndarray] = []
    all_pred_deltas: List[np.ndarray] = []
    all_gt_vals: List[np.ndarray] = []
    all_pred_vals: List[np.ndarray] = []

    for pred_path in pred_files:
        bid = pred_path.stem.split("_", 1)[0]
        gt_path = gt_by_id.get(bid)
        if gt_path is None:
            print(f"[diag] book {bid}: no GT file -> skip")
            continue
        gd, pd_, gv, pv, npairs = diagnose_book(pred_path, gt_path)
        if gd.size == 0:
            print(f"[diag] book {bid}: no valid transitions -> skip")
            continue
        all_gt_deltas.append(gd)
        all_pred_deltas.append(pd_)
        all_gt_vals.append(gv)
        all_pred_vals.append(pv)
        rows.append(
            {
                "book_id": bid,
                "n_pairs": float(npairs),
                "n_transitions": float(gd.size),
                "gt_mean_abs_delta": float(np.mean(gd)),
                "pred_mean_abs_delta": float(np.mean(pd_)),
                "ratio_pred_over_gt": float(np.mean(pd_) / max(np.mean(gd), 1e-9)),
                "gt_std_value": float(np.std(gv)) if gv.size > 0 else float("nan"),
                "pred_std_value": float(np.std(pv)) if pv.size > 0 else float("nan"),
                "gt_mean_value": float(np.mean(gv)) if gv.size > 0 else float("nan"),
                "pred_mean_value": float(np.mean(pv)) if pv.size > 0 else float("nan"),
                "mae_value": float(np.mean(np.abs(gv - pv))) if gv.size > 0 else float("nan"),
            }
        )

    if not rows:
        print("[diag] no rows produced; nothing to do")
        return

    table = pd.DataFrame(rows).sort_values("book_id", key=lambda s: s.astype(int) if s.str.isnumeric().all() else s)
    csv_path = out_root / "score_compression_per_book.csv"
    table.to_csv(csv_path, index=False)
    print(f"[diag] wrote {csv_path}")

    gd = np.concatenate(all_gt_deltas)
    pd_ = np.concatenate(all_pred_deltas)
    gv = np.concatenate(all_gt_vals)
    pv = np.concatenate(all_pred_vals)

    print()
    print("=" * 70)
    print("AGGREGATE DIAGNOSTIC (across all books)")
    print("=" * 70)
    print(f"  Transitions analyzed (both endpoints valid in GT): {gd.size:,}")
    print(f"  Aligned valid cells:                                {gv.size:,}")
    print()
    print("  Chapter-to-chapter |Δ|:")
    print(
        f"    GT     : mean={np.mean(gd):.4f}  median={np.median(gd):.4f}  "
        f"p90={np.percentile(gd, 90):.4f}  max={np.max(gd):.4f}"
    )
    print(
        f"    PRED   : mean={np.mean(pd_):.4f}  median={np.median(pd_):.4f}  "
        f"p90={np.percentile(pd_, 90):.4f}  max={np.max(pd_):.4f}"
    )
    compression = np.mean(pd_) / max(np.mean(gd), 1e-9)
    print(f"    RATIO  : pred/gt = {compression:.3f}  (1.0 = no compression)")
    print()
    print("  Score values (where GT in [0,1]):")
    print(
        f"    GT     : mean={np.mean(gv):.3f}  std={np.std(gv):.3f}  "
        f"min={np.min(gv):.3f}  max={np.max(gv):.3f}"
    )
    print(
        f"    PRED   : mean={np.mean(pv):.3f}  std={np.std(pv):.3f}  "
        f"min={np.min(pv):.3f}  max={np.max(pv):.3f}"
    )
    print(f"    STD ratio pred/gt = {np.std(pv) / max(np.std(gv), 1e-9):.3f}")
    print(f"    MAE(pred, gt)     = {np.mean(np.abs(gv - pv)):.4f}")
    print("=" * 70)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="black")
    ax.set_facecolor("black")
    bins = np.linspace(0, 1, 41)
    ax.hist(gd, bins=bins, alpha=0.55, color="#FFD400", label="GT |Δ|")
    ax.hist(pd_, bins=bins, alpha=0.55, color="#4FC3F7", label="Pred |Δ|")
    ax.set_yscale("log")
    ax.set_xlabel("|Δ| chapter→chapter", color="white")
    ax.set_ylabel("count (log)", color="white")
    ax.set_title("Distribution of |Δ| per transition: GT vs Predicted", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    p1 = out_root / "delta_distribution_gt_vs_pred.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150, facecolor="black")
    plt.close()
    print(f"[diag] wrote {p1}")

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="black")
    ax.set_facecolor("black")
    bins = np.linspace(0, 1, 41)
    ax.hist(gv, bins=bins, alpha=0.55, color="#FFD400", label="GT score")
    ax.hist(pv, bins=bins, alpha=0.55, color="#4FC3F7", label="Pred score")
    ax.set_xlabel("score in [0,1]", color="white")
    ax.set_ylabel("count", color="white")
    ax.set_title("Distribution of values: GT vs Predicted", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    p2 = out_root / "value_distribution_gt_vs_pred.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150, facecolor="black")
    plt.close()
    print(f"[diag] wrote {p2}")


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--gt-dir", default="data/dataset/ground_truth")
    ap.add_argument("--output-dir", default="data/evaluation/diagnostics/score_compression")
    args = ap.parse_args()
    diagnose(args.predictions_dir, args.gt_dir, args.output_dir)


if __name__ == "__main__":
    main()
