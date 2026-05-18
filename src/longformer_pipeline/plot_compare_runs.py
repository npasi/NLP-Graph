"""Overlay all-books mean |Δ predicted score| curves for two runs (fine-tuned vs pretrained)."""

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
    cols = []
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


def _mean_abs_delta_per_transition(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        return np.array([], dtype=float)
    a = scores[:, :-1]
    b = scores[:, 1:]
    valid = (a >= 0) & (b >= 0) & np.isfinite(a) & np.isfinite(b)
    diff = np.abs(b - a)
    diff = np.where(valid, diff, np.nan)
    return np.nanmean(diff, axis=0)


def _stack_curves(predictions_dir: str) -> Tuple[np.ndarray, int]:
    root = Path(predictions_dir)
    paths = sorted(root.glob("*_predicted.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_predicted.csv in {root}")
    curves: Dict[str, np.ndarray] = {}
    max_len = 0
    for p in paths:
        df = pd.read_csv(p)
        ch = _chapter_cols(df)
        if not ch:
            continue
        mat = df[ch].to_numpy(dtype=float)
        c = _mean_abs_delta_per_transition(mat)
        if c.size == 0:
            continue
        bid = p.stem.split("_", 1)[0]
        curves[bid] = c
        max_len = max(max_len, int(c.size))
    if not curves:
        raise ValueError(f"No valid curves in {root}")
    bids = sorted(curves.keys(), key=lambda x: int(x) if x.isdigit() else x)
    stacked = np.full((len(bids), max_len), np.nan, dtype=float)
    for i, bid in enumerate(bids):
        c = curves[bid]
        stacked[i, : c.size] = c
    return stacked, max_len


def plot_compare(
    finetuned_dir: str,
    pretrained_dir: str,
    output_path: str,
    yscale: str = "log",
) -> None:
    ft_stack, ft_len = _stack_curves(finetuned_dir)
    pt_stack, pt_len = _stack_curves(pretrained_dir)
    max_len = max(ft_len, pt_len)

    def _pad(stack: np.ndarray, n: int) -> np.ndarray:
        if stack.shape[1] == n:
            return stack
        out = np.full((stack.shape[0], n), np.nan, dtype=float)
        out[:, : stack.shape[1]] = stack
        return out

    ft = _pad(ft_stack, max_len)
    pt = _pad(pt_stack, max_len)
    ft_mean = np.nanmean(ft, axis=0)
    pt_mean = np.nanmean(pt, axis=0)

    x = np.arange(1, max_len + 1)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
    ax.set_facecolor("black")

    purple = "#B266FF"
    azzurro = "#4FC3F7"
    white = "#FFFFFF"

    eps = 1e-6
    if yscale == "log":
        ft_plot = np.where(np.isfinite(ft) & (ft > 0), ft, np.nan)
        pt_plot = np.where(np.isfinite(pt) & (pt > 0), pt, np.nan)
        ft_mean_plot = np.where(np.isfinite(ft_mean) & (ft_mean > 0), ft_mean, np.nan)
        pt_mean_plot = np.where(np.isfinite(pt_mean) & (pt_mean > 0), pt_mean, np.nan)
    else:
        ft_plot, pt_plot = ft, pt
        ft_mean_plot, pt_mean_plot = ft_mean, pt_mean

    for row in ft_plot:
        ax.plot(x, row, color=purple, alpha=0.25, linewidth=0.9)
    for row in pt_plot:
        ax.plot(x, row, color=purple, alpha=0.25, linewidth=0.9)

    ax.plot(x, ft_mean_plot, color=azzurro, linewidth=2.6, label="Fine-tuned (mean)")
    ax.plot(x, pt_mean_plot, color=white, linewidth=2.6, label="Pretrained (mean)")

    title_extra = " [log y]" if yscale == "log" else ""
    ax.set_title(
        "average error of predicted confidence score against chapter "
        f"(all books plotted) - fine-tuned vs pretrained{title_extra}",
        color="white",
    )
    ax.set_xlabel("chapter", color="white")
    ax.set_ylabel("Mean |Δ predicted score| across pairs", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    if yscale == "log":
        ax.set_yscale("log")
        finite_min = np.nanmin(np.where(ft_plot > 0, ft_plot, np.nan))
        if not np.isfinite(finite_min) or finite_min <= 0:
            finite_min = eps
        ax.set_ylim(bottom=max(finite_min * 0.5, eps))
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25, color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp, dpi=160, facecolor="black")
    plt.close()
    print(f"[plot] Wrote: {outp}  (yscale={yscale})")
    print(
        f"[plot] FT books: {ft.shape[0]}, PT books: {pt.shape[0]}, "
        f"transitions: {max_len}"
    )
    print(
        f"[plot] FT mean range: [{np.nanmin(ft_mean):.5f}, {np.nanmax(ft_mean):.5f}] | "
        f"PT mean range: [{np.nanmin(pt_mean):.5f}, {np.nanmax(pt_mean):.5f}]"
    )


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned-dir", required=True)
    ap.add_argument("--pretrained-dir", required=True)
    ap.add_argument(
        "--output",
        default="data/evaluation/compare/all_books_avg_change_compare_log.png",
    )
    ap.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="log",
        help="Y-axis scale (log handles the order-of-magnitude gap between FT and PT)",
    )
    args = ap.parse_args()
    plot_compare(
        args.finetuned_dir,
        args.pretrained_dir,
        args.output,
        yscale=args.yscale,
    )


if __name__ == "__main__":
    main()
