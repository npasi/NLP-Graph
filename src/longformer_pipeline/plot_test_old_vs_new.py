"""Compare old (1ep MSE) vs new (3ep MSE) predictions on the test set.

Produces:
  1) A per-book overlay of |delta predicted score| across chapter transitions
     (one figure per book, both runs superimposed)
  2) An aggregate plot (mean across books) with both runs superimposed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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


def _curve_for_book(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    ch = _chapter_cols(df)
    if not ch:
        return np.array([])
    mat = df[ch].to_numpy(dtype=float)
    return _mean_abs_delta_per_transition(mat)


def _style_dark(ax):
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def plot_per_book(old_dir: Path, new_dir: Path, out_dir: Path, book_ids: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    color_old = "#FFD400"
    color_new = "#4FC3F7"

    for bid in book_ids:
        f_old = old_dir / f"{bid}_predicted.csv"
        f_new = new_dir / f"{bid}_predicted.csv"
        if not f_old.exists() or not f_new.exists():
            print(f"[skip] {bid}: missing prediction csv (old={f_old.exists()}, new={f_new.exists()})")
            continue
        c_old = _curve_for_book(f_old)
        c_new = _curve_for_book(f_new)
        if c_old.size == 0 and c_new.size == 0:
            print(f"[skip] {bid}: empty curves")
            continue
        n = max(c_old.size, c_new.size)
        x = np.arange(1, n + 1)
        co = np.full(n, np.nan)
        cn = np.full(n, np.nan)
        co[: c_old.size] = c_old
        cn[: c_new.size] = c_new

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
        _style_dark(ax)
        ax.plot(x, co, color=color_old, linewidth=2.4, marker="o", label="Old (1ep MSE)")
        ax.plot(x, cn, color=color_new, linewidth=2.4, marker="s", label="New (3ep MSE)")
        ax.set_title(
            f"book {bid} - mean |delta predicted score| across chapter transitions",
            color="white",
        )
        ax.set_xlabel("chapter transition (i -> i+1)", color="white")
        ax.set_ylabel("mean |delta predicted score|", color="white")
        ax.grid(True, alpha=0.25, color="white")
        ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
        outp = out_dir / f"book_{bid}_old_vs_new.png"
        plt.tight_layout()
        plt.savefig(outp, dpi=160, facecolor="black")
        plt.close()
        print(f"[plot] {outp}")


def plot_aggregate(old_dir: Path, new_dir: Path, out_path: Path, book_ids: List[str]) -> None:
    plt.style.use("dark_background")

    def _stack(d: Path) -> np.ndarray:
        curves: Dict[str, np.ndarray] = {}
        max_len = 0
        for bid in book_ids:
            p = d / f"{bid}_predicted.csv"
            if not p.exists():
                continue
            c = _curve_for_book(p)
            if c.size == 0:
                continue
            curves[bid] = c
            max_len = max(max_len, c.size)
        if not curves:
            return np.zeros((0, 0))
        out = np.full((len(curves), max_len), np.nan)
        for i, bid in enumerate(sorted(curves.keys(), key=lambda x: int(x))):
            c = curves[bid]
            out[i, : c.size] = c
        return out

    old_stack = _stack(old_dir)
    new_stack = _stack(new_dir)
    n = max(old_stack.shape[1] if old_stack.size else 0, new_stack.shape[1] if new_stack.size else 0)
    if n == 0:
        print("[plot] No data for aggregate")
        return

    def _pad(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return np.zeros((0, n))
        if a.shape[1] == n:
            return a
        out = np.full((a.shape[0], n), np.nan)
        out[:, : a.shape[1]] = a
        return out

    old_p = _pad(old_stack)
    new_p = _pad(new_stack)
    old_mean = np.nanmean(old_p, axis=0) if old_p.size else np.full(n, np.nan)
    new_mean = np.nanmean(new_p, axis=0) if new_p.size else np.full(n, np.nan)

    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
    _style_dark(ax)

    purple = "#B266FF"
    color_old = "#FFD400"
    color_new = "#4FC3F7"

    for row in old_p:
        ax.plot(x, row, color=purple, alpha=0.30, linewidth=1.0)
    for row in new_p:
        ax.plot(x, row, color=purple, alpha=0.30, linewidth=1.0)

    ax.plot(x, old_mean, color=color_old, linewidth=2.6, label="Old mean (1ep MSE)")
    ax.plot(x, new_mean, color=color_new, linewidth=2.6, label="New mean (3ep MSE)")

    ax.set_title(
        f"test set ({old_p.shape[0]} books) - mean |delta predicted score| across chapter transitions",
        color="white",
    )
    ax.set_xlabel("chapter transition (i -> i+1)", color="white")
    ax.set_ylabel("mean |delta predicted score|", color="white")
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor="black")
    plt.close()
    print(f"[plot] {out_path}")
    print(
        f"[plot] old mean range [{np.nanmin(old_mean):.5f}, {np.nanmax(old_mean):.5f}] | "
        f"new mean range [{np.nanmin(new_mean):.5f}, {np.nanmax(new_mean):.5f}]"
    )


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", default="data/predictions/prevprefix_test_1ep")
    ap.add_argument("--new-dir", default="data/predictions/prevprefix_test_3ep_mse")
    ap.add_argument("--out-dir", default="data/evaluation/compare/test_old_vs_new")
    ap.add_argument("--book-ids", default="95,99,110")
    args = ap.parse_args()

    book_ids = [b.strip() for b in args.book_ids.split(",") if b.strip()]
    out_dir = Path(args.out_dir)
    plot_per_book(Path(args.old_dir), Path(args.new_dir), out_dir, book_ids)
    plot_aggregate(
        Path(args.old_dir),
        Path(args.new_dir),
        out_dir / "test_aggregate_old_vs_new.png",
        book_ids,
    )


if __name__ == "__main__":
    main()
