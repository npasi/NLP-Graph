"""Step 6: compare predicted CSV vs ground-truth CSV — metrics and plots."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from longformer_pipeline.build_dataset import _CHAPTER_COL_RE


def _ensure_src_on_path() -> None:
    this_file = Path(__file__).resolve()
    src_dir = this_file.parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _reconfigure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _chapter_columns(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if _CHAPTER_COL_RE.match(str(c).strip())],
        key=lambda c: int(_CHAPTER_COL_RE.match(str(c).strip()).group(1)),
    )


def _cell_in_unit_interval(val) -> bool:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return False
    if math.isnan(x):
        return False
    return 0.0 <= x <= 1.0


def _pair_fuzzy_score(a1: str, a2: str, b1: str, b2: str) -> float:
    s1 = (fuzz.token_set_ratio(a1, b1) + fuzz.token_set_ratio(a2, b2)) / 2.0
    s2 = (fuzz.token_set_ratio(a1, b2) + fuzz.token_set_ratio(a2, b1)) / 2.0
    return max(s1, s2)


def _find_pred_row(
    gt_c1: str,
    gt_c2: str,
    pred_df: pd.DataFrame,
    *,
    fuzzy_threshold: float = 80.0,
) -> Optional[pd.Series]:
    g1 = str(gt_c1).strip()
    g2 = str(gt_c2).strip()

    best_idx: Optional[int] = None
    best_fuzz = -1.0

    for idx, row in pred_df.iterrows():
        p1 = str(row["character_1"]).strip()
        p2 = str(row["character_2"]).strip()
        if (g1, g2) == (p1, p2) or (g1, g2) == (p2, p1):
            return row
        fs = _pair_fuzzy_score(g1, g2, p1, p2)
        if fs > best_fuzz:
            best_fuzz = fs
            best_idx = idx

    if best_idx is not None and best_fuzz > fuzzy_threshold:
        return pred_df.loc[best_idx]
    return None


def align_csvs(predicted_csv: str, gt_csv: str) -> List[Tuple[float, float]]:
    """Align GT pairs to predictions; return (pred_score, gt_score) for comparable cells."""
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(gt_csv)

    print(f"[align] Predicted CSV: {predicted_csv} ({len(pred_df)} pairs)")
    print(f"[align] GT CSV: {gt_csv} ({len(gt_df)} pairs)")

    pred_ch = _chapter_columns(pred_df)
    gt_ch = _chapter_columns(gt_df)
    common_ch = [c for c in gt_ch if c in pred_ch]
    if not common_ch:
        raise ValueError("No common chapter_* columns between predicted and GT CSVs.")

    aligned: List[Tuple[float, float]] = []
    unmatched_pairs: List[str] = []
    n_matched = 0

    for _, gt_row in gt_df.iterrows():
        g1 = str(gt_row["character_1"]).strip()
        g2 = str(gt_row["character_2"]).strip()
        prow = _find_pred_row(g1, g2, pred_df)
        if prow is None:
            unmatched_pairs.append(f"{g1} <-> {g2}")
            continue
        n_matched += 1
        for col in common_ch:
            if not _cell_in_unit_interval(gt_row[col]):
                continue
            if not _cell_in_unit_interval(prow[col]):
                continue
            aligned.append((float(prow[col]), float(gt_row[col])))

    print(f"[align] Matched pairs: {n_matched}/{len(gt_df)}")
    print(f"[align] Comparable cells (both in [0,1]): {len(aligned)}")
    if unmatched_pairs:
        print(
            f"[align] WARNING: {len(unmatched_pairs)} GT pairs not found in predictions:"
        )
        for p in unmatched_pairs:
            print(f"[align]   {p}")

    assert len(aligned) > 0, "No comparable cells found! Check pair matching."
    return aligned


def compute_metrics(aligned: List[Tuple[float, float]]) -> Dict:
    preds = [p for p, g in aligned]
    gts = [g for p, g in aligned]

    mse = float(mean_squared_error(gts, preds))
    try:
        rmse = float(mean_squared_error(gts, preds, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mse))

    pr = pearsonr(gts, preds)
    sp = spearmanr(gts, preds)

    pred_mean = sum(preds) / len(preds)
    gt_mean = sum(gts) / len(gts)
    pred_std = (sum((p - pred_mean) ** 2 for p in preds) / len(preds)) ** 0.5
    gt_std = (sum((g - gt_mean) ** 2 for g in gts) / len(gts)) ** 0.5

    metrics = {
        "n_cells": len(aligned),
        "mse": mse,
        "rmse": rmse,
        "mae": float(mean_absolute_error(gts, preds)),
        "pearson": float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0]),
        "pearson_pvalue": float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1]),
        "spearman": float(sp.statistic) if hasattr(sp, "statistic") else float(sp[0]),
        "spearman_pvalue": float(sp.pvalue) if hasattr(sp, "pvalue") else float(sp[1]),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "gt_mean": gt_mean,
        "gt_std": gt_std,
    }

    print(f"\n{'=' * 60}")
    print("  EVALUATION METRICS")
    print(f"{'=' * 60}")
    print(f"  Cells compared:    {metrics['n_cells']}")
    print(f"  MSE:               {metrics['mse']:.4f}")
    print(f"  RMSE:              {metrics['rmse']:.4f}")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(
        f"  Pearson:           {metrics['pearson']:.4f} "
        f"(p={metrics['pearson_pvalue']:.2e})"
    )
    print(
        f"  Spearman:          {metrics['spearman']:.4f} "
        f"(p={metrics['spearman_pvalue']:.2e})"
    )
    print(f"  Pred mean±std:     {metrics['pred_mean']:.3f} ± {metrics['pred_std']:.3f}")
    print(f"  GT mean±std:       {metrics['gt_mean']:.3f} ± {metrics['gt_std']:.3f}")
    print(f"{'=' * 60}")

    return metrics


def plot_scatter(
    aligned: List[Tuple[float, float]],
    output_path: str,
    title: str = "",
) -> None:
    preds = [p for p, g in aligned]
    gts = [g for p, g in aligned]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(gts, preds, alpha=0.5, s=30, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Ground Truth Score", fontsize=12)
    ax.set_ylabel("Predicted Score", fontsize=12)
    ax.set_title(title or "Predicted vs Ground Truth", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    mse = mean_squared_error(gts, preds)
    pr = pearsonr(gts, preds)
    r = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
    ax.text(
        0.05,
        0.95,
        f"MSE={mse:.4f}\nPearson={r:.4f}\nn={len(aligned)}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[plot] Scatter plot saved to: {output_path}")
    plt.close()


def plot_error_distribution(aligned: List[Tuple[float, float]], output_path: str) -> None:
    errors = [p - g for p, g in aligned]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Error (Predicted - GT)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)

    mean_err = sum(errors) / len(errors)
    ax.axvline(
        mean_err,
        color="orange",
        linewidth=1,
        linestyle="--",
        label=f"Mean error: {mean_err:.4f}",
    )
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[plot] Error distribution saved to: {output_path}")
    plt.close()


def _mean_abs_deltas(series: List[float]) -> List[float]:
    """Mean absolute delta per transition i->i+1, using only consecutive valid points (>=0)."""
    if not series or len(series) < 2:
        return []
    out: List[float] = []
    for a, b in zip(series[:-1], series[1:], strict=True):
        if a >= 0 and b >= 0:
            out.append(abs(b - a))
        else:
            out.append(float("nan"))
    return out


def plot_avg_change(
    predicted_csv: str,
    gt_csv: str,
    output_path: str,
) -> None:
    """Plot average absolute change across chapters (Pred vs GT)."""
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(gt_csv)
    pred_ch = _chapter_columns(pred_df)
    gt_ch = _chapter_columns(gt_df)
    common_ch = [c for c in gt_ch if c in pred_ch]
    if len(common_ch) < 2:
        print("[change] Not enough chapter columns for change plot.")
        return

    # Collect deltas for pairs that exist in both (matched by GT→pred)
    deltas_pred: List[List[float]] = []
    deltas_gt: List[List[float]] = []

    for _, gt_row in gt_df.iterrows():
        g1 = str(gt_row["character_1"]).strip()
        g2 = str(gt_row["character_2"]).strip()
        prow = _find_pred_row(g1, g2, pred_df)
        if prow is None:
            continue
        pred_scores = [float(prow[c]) for c in common_ch]
        gt_scores = [float(gt_row[c]) for c in common_ch]
        deltas_pred.append(_mean_abs_deltas(pred_scores))
        deltas_gt.append(_mean_abs_deltas(gt_scores))

    if not deltas_pred:
        print("[change] No matched pairs for change plot.")
        return

    # Mean ignoring NaNs
    dp = np.array(deltas_pred, dtype=float)
    dg = np.array(deltas_gt, dtype=float)
    mean_pred = np.nanmean(dp, axis=0)
    mean_gt = np.nanmean(dg, axis=0)

    x = list(range(1, len(common_ch)))  # transition index: 1 means ch1->ch2
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x, mean_pred, "b-o", label="Predicted mean |Δ|", markersize=5)
    ax.plot(x, mean_gt, "r--s", label="GT mean |Δ|", markersize=5)
    ax.set_xlabel("Chapter transition (i means chapter_i -> chapter_{i+1})")
    ax.set_ylabel("Mean |Δ score|")
    ax.set_title("Average score change across chapters")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] Avg change plot saved to: {output_path}")


def plot_top_changers(
    predicted_csv: str,
    gt_csv: str,
    output_path: str,
    *,
    top_k: int = 5,
) -> None:
    """Plot the top-K predicted pairs with the largest mean |Δ| across chapters, with GT overlay."""
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(gt_csv)
    pred_ch = _chapter_columns(pred_df)
    gt_ch = _chapter_columns(gt_df)
    common_ch = [c for c in gt_ch if c in pred_ch]
    if len(common_ch) < 2:
        print("[change] Not enough chapter columns for top changers.")
        return

    matched = _match_pairs_for_arcs(predicted_csv, gt_csv)  # (pair_name, pred_scores, gt_scores)
    scored: List[Tuple[float, str, List[float], List[float]]] = []
    for pair_name, pred_scores, gt_scores in matched:
        deltas = _mean_abs_deltas(pred_scores)
        # compute mean over non-NaN transitions
        vals = [d for d in deltas if not (isinstance(d, float) and math.isnan(d))]
        if not vals:
            continue
        scored.append((float(sum(vals) / len(vals)), pair_name, pred_scores, gt_scores))

    if not scored:
        print("[change] No pairs with valid deltas for top changers.")
        return

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    n = len(top)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    chapters = list(range(1, len(common_ch) + 1))
    for ax, (mean_abs_delta, pair_name, pred_scores, gt_scores) in zip(axes, top, strict=True):
        valid_pred = [(ch, s) for ch, s in zip(chapters, pred_scores) if s >= 0]
        valid_gt = [(ch, s) for ch, s in zip(chapters, gt_scores) if s >= 0]

        if valid_pred:
            ax.plot(*zip(*valid_pred), "b-o", label="Predicted", markersize=5)
        if valid_gt:
            ax.plot(*zip(*valid_gt), "r--s", label="GT", markersize=5)

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{pair_name}  (pred mean |Δ| = {mean_abs_delta:.3f})")
        ax.legend()

    axes[-1].set_xlabel("Chapter")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] Top changers plot saved to: {output_path}")


def _sanitize_filename(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:120] if len(s) > 120 else s


def _match_pairs_for_arcs(
    predicted_csv: str,
    gt_csv: str,
) -> List[Tuple[str, List[float], List[float]]]:
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(gt_csv)
    pred_ch = _chapter_columns(pred_df)
    gt_ch = _chapter_columns(gt_df)
    common_ch = [c for c in gt_ch if c in pred_ch]

    out: List[Tuple[str, List[float], List[float]]] = []
    for _, gt_row in gt_df.iterrows():
        g1 = str(gt_row["character_1"]).strip()
        g2 = str(gt_row["character_2"]).strip()
        prow = _find_pred_row(g1, g2, pred_df)
        if prow is None:
            continue
        pred_scores = [float(prow[c]) for c in common_ch]
        gt_scores = [float(gt_row[c]) for c in common_ch]
        pair_name = f"{g1} vs {g2}"
        out.append((pair_name, pred_scores, gt_scores))
    return out


def plot_pair_arcs(
    predicted_csv: str,
    gt_csv: str,
    output_dir: str,
    max_pairs: int = 10,
) -> None:
    """Plot predicted vs GT trajectories for the first ``max_pairs`` matched GT rows."""
    matched_pairs = _match_pairs_for_arcs(predicted_csv, gt_csv)
    os.makedirs(output_dir, exist_ok=True)

    for i, (pair_name, pred_scores, gt_scores) in enumerate(matched_pairs[:max_pairs]):
        fig, ax = plt.subplots(figsize=(10, 4))

        chapters = list(range(1, len(pred_scores) + 1))

        valid_pred = [(ch, s) for ch, s in zip(chapters, pred_scores) if s >= 0]
        valid_gt = [(ch, s) for ch, s in zip(chapters, gt_scores) if s >= 0]

        if valid_pred:
            ax.plot(*zip(*valid_pred), "b-o", label="Predicted", markersize=6)
        if valid_gt:
            ax.plot(*zip(*valid_gt), "r--s", label="GT", markersize=6)

        ax.set_xlabel("Chapter")
        ax.set_ylabel("Score")
        ax.set_title(pair_name)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        safe = _sanitize_filename(pair_name)
        out_path = os.path.join(output_dir, f"arc_{i + 1}_{safe}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[arcs] Saved arc plot: {out_path}")


def _metrics_for_json(m: Dict) -> Dict:
    def fix(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    return {k: fix(v) for k, v in m.items()}


def evaluate(
    predicted_csv: str,
    gt_csv: str,
    output_dir: str = "data/evaluation",
) -> Dict:
    _reconfigure_stdio_utf8()
    _ensure_src_on_path()
    print(f"\n{'=' * 60}")
    print("  EVALUATION")
    print(f"  Predicted: {predicted_csv}")
    print(f"  GT: {gt_csv}")
    print(f"{'=' * 60}\n")

    os.makedirs(output_dir, exist_ok=True)

    aligned = align_csvs(predicted_csv, gt_csv)
    metrics = compute_metrics(aligned)

    plot_scatter(aligned, os.path.join(output_dir, "scatter.png"))
    plot_error_distribution(aligned, os.path.join(output_dir, "error_dist.png"))
    plot_pair_arcs(predicted_csv, gt_csv, output_dir)
    plot_avg_change(predicted_csv, gt_csv, os.path.join(output_dir, "avg_change.png"))
    plot_top_changers(
        predicted_csv,
        gt_csv,
        os.path.join(output_dir, "top5_changers.png"),
        top_k=5,
    )

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_metrics_for_json(metrics), f, indent=2)
    print(f"\n[eval] Metrics saved to: {metrics_path}")

    print(f"\n{'=' * 60}")
    print("  ASSESSMENT")
    print(f"{'=' * 60}")
    mae = metrics["mae"]
    pear = metrics["pearson"]
    if math.isnan(pear):
        print(f"  FAIL — Pearson is NaN (constant inputs?); MAE={mae:.4f}")
    elif mae < 0.15 and pear > 0.6:
        print("  PASS — MAE < 0.15 and Pearson > 0.6")
    elif mae < 0.20 and pear > 0.4:
        print("  MARGINAL — needs improvement")
    else:
        print(f"  FAIL — MAE={mae:.4f}, Pearson={pear:.4f}")
    print(f"{'=' * 60}")

    return metrics


def main() -> None:
    _reconfigure_stdio_utf8()
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(
        description="Evaluate predicted vs GT sentiment scores",
    )
    parser.add_argument("--predicted", required=True, help="Path to predicted CSV")
    parser.add_argument("--gt", required=True, help="Path to GT CSV")
    parser.add_argument("--output-dir", default="data/evaluation")
    args = parser.parse_args()

    evaluate(args.predicted, args.gt, args.output_dir)


if __name__ == "__main__":
    main()
