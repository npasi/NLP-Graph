"""Plot average chapter-to-chapter confidence change for all books in one chart.

Confidence = predicted score in [0,1] from Step 5 CSVs.
For each book and each transition (chapter_i -> chapter_{i+1}), we compute:
  mean(|score_{i+1} - score_i|) across all pairs where both scores are valid (>=0).
Then plot all books on a single figure.
"""

from __future__ import annotations

import argparse
import math
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
    """scores: shape [n_pairs, n_chapters], with invalid marked as <0 or NaN."""
    if scores.shape[1] < 2:
        return np.array([], dtype=float)
    a = scores[:, :-1]
    b = scores[:, 1:]
    valid = (a >= 0) & (b >= 0) & np.isfinite(a) & np.isfinite(b)
    diff = np.abs(b - a)
    diff = np.where(valid, diff, np.nan)
    return np.nanmean(diff, axis=0)  # length n_chapters-1


def compute_book_curve(pred_csv: str) -> Tuple[str, np.ndarray]:
    df = pd.read_csv(pred_csv)
    ch = _chapter_cols(df)
    if not ch:
        raise ValueError(f"No chapter_* columns in {pred_csv}")
    mat = df[ch].to_numpy(dtype=float)
    curve = _mean_abs_delta_per_transition(mat)
    book_id = Path(pred_csv).stem.split("_", 1)[0]
    return book_id, curve


def plot_all_books(predictions_dir: str, output_path: str) -> None:
    root = Path(predictions_dir)
    paths = sorted(root.glob("*_predicted.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_predicted.csv found in {root}")

    curves: Dict[str, np.ndarray] = {}
    max_len = 0
    for p in paths:
        bid, curve = compute_book_curve(str(p))
        if curve.size == 0:
            continue
        curves[bid] = curve
        max_len = max(max_len, int(curve.size))

    if not curves:
        raise ValueError("No valid curves computed (need at least 2 chapters).")

    # Stack with NaN padding
    bids = sorted(curves.keys(), key=lambda x: int(x) if x.isdigit() else x)
    stacked = np.full((len(bids), max_len), np.nan, dtype=float)
    for i, bid in enumerate(bids):
        c = curves[bid]
        stacked[i, : c.size] = c

    mean_curve = np.nanmean(stacked, axis=0)

    x = np.arange(1, max_len + 1)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
    ax.set_facecolor("black")

    purple = "#B266FF"
    yellow = "#FFD400"

    for i, _bid in enumerate(bids):
        y = stacked[i]
        ax.plot(x, y, color=purple, alpha=0.35, linewidth=1)

    ax.plot(x, mean_curve, color=yellow, linewidth=2.6, label="Mean across books")

    ax.set_title(
        "average error of predicted confidence score against chapter (all books plotted)",
        color="white",
    )
    ax.set_xlabel("chapter", color="white")
    ax.set_ylabel("Mean |Δ predicted score| across pairs", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp, dpi=160, facecolor="black")
    plt.close()
    print(f"[plot] Wrote: {outp}")
    print(f"[plot] Books plotted: {len(bids)}")
    print(f"[plot] Max transitions: {max_len}")


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing <book_id>_predicted.csv files",
    )
    ap.add_argument(
        "--output",
        default="data/evaluation/compare/all_books_avg_change_ft.png",
        help="Output PNG path",
    )
    args = ap.parse_args()
    plot_all_books(args.predictions_dir, args.output)


if __name__ == "__main__":
    main()

