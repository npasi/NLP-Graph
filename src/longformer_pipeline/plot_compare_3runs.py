"""Overlay all-books mean |delta predicted score| curves for THREE runs.

Reads one or more prediction directories per model and computes the
intersection of book IDs across all three models, then plots:
  - thin purple lines for each individual book/model curve
  - one bold mean line per model

Designed as a 3-way extension of plot_compare_runs.py (FT vs PT).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def _curve(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    ch = _chapter_cols(df)
    if not ch:
        return np.array([])
    mat = df[ch].to_numpy(dtype=float)
    return _mean_abs_delta_per_transition(mat)


def _ids_in_dirs(dirs: Sequence[Path]) -> List[str]:
    ids: List[str] = []
    seen = set()
    for d in dirs:
        for p in sorted(d.glob("*_predicted.csv")):
            bid = p.stem.split("_", 1)[0]
            if bid not in seen:
                seen.add(bid)
                ids.append(bid)
    return ids


def _curves_for_model(dirs: Sequence[Path], book_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """For each book id, find the first prediction csv across the given dirs."""
    out: Dict[str, np.ndarray] = {}
    for bid in book_ids:
        for d in dirs:
            p = d / f"{bid}_predicted.csv"
            if p.exists():
                c = _curve(p)
                if c.size > 0:
                    out[bid] = c
                break
    return out


def _stack(curves: Dict[str, np.ndarray], book_ids: Sequence[str]) -> Tuple[np.ndarray, int]:
    if not curves:
        return np.zeros((0, 0)), 0
    max_len = max(c.size for c in curves.values())
    out = np.full((len(book_ids), max_len), np.nan)
    for i, bid in enumerate(book_ids):
        c = curves.get(bid)
        if c is not None and c.size:
            out[i, : c.size] = c
    return out, max_len


def plot_3way(
    new_dirs: Sequence[Path],
    old_dirs: Sequence[Path],
    pre_dirs: Sequence[Path],
    output_path: Path,
    yscale: str = "log",
) -> None:
    new_ids = set(_ids_in_dirs(new_dirs))
    old_ids = set(_ids_in_dirs(old_dirs))
    pre_ids = set(_ids_in_dirs(pre_dirs))
    common = sorted(new_ids & old_ids & pre_ids, key=lambda x: int(x) if x.isdigit() else x)
    print(f"[plot] new_ids: {sorted(new_ids, key=lambda x: int(x))}")
    print(f"[plot] old_ids: {sorted(old_ids, key=lambda x: int(x))}")
    print(f"[plot] pre_ids: {sorted(pre_ids, key=lambda x: int(x))}")
    print(f"[plot] intersection ({len(common)}): {common}")
    if not common:
        raise ValueError("No book id is present in all three model directories")

    new_c = _curves_for_model(new_dirs, common)
    old_c = _curves_for_model(old_dirs, common)
    pre_c = _curves_for_model(pre_dirs, common)

    new_st, n1 = _stack(new_c, common)
    old_st, n2 = _stack(old_c, common)
    pre_st, n3 = _stack(pre_c, common)
    n = max(n1, n2, n3)

    def _pad(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return np.zeros((len(common), n)) * np.nan
        if a.shape[1] == n:
            return a
        out = np.full((a.shape[0], n), np.nan)
        out[:, : a.shape[1]] = a
        return out

    new_p = _pad(new_st)
    old_p = _pad(old_st)
    pre_p = _pad(pre_st)

    new_mean = np.nanmean(new_p, axis=0)
    old_mean = np.nanmean(old_p, axis=0)
    pre_mean = np.nanmean(pre_p, axis=0)

    x = np.arange(1, n + 1)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="black")
    ax.set_facecolor("black")

    purple = "#B266FF"
    green = "#7CFC00"
    yellow = "#FFD400"
    azzurro = "#4FC3F7"

    if yscale == "log":
        def _pos(a):
            return np.where(np.isfinite(a) & (a > 0), a, np.nan)
        new_show, old_show, pre_show = _pos(new_p), _pos(old_p), _pos(pre_p)
        new_m_show, old_m_show, pre_m_show = _pos(new_mean), _pos(old_mean), _pos(pre_mean)
    else:
        new_show, old_show, pre_show = new_p, old_p, pre_p
        new_m_show, old_m_show, pre_m_show = new_mean, old_mean, pre_mean

    for row in new_show:
        ax.plot(x, row, color=purple, alpha=0.22, linewidth=0.8)
    for row in old_show:
        ax.plot(x, row, color=purple, alpha=0.22, linewidth=0.8)
    for row in pre_show:
        ax.plot(x, row, color=purple, alpha=0.22, linewidth=0.8)

    ax.plot(x, new_m_show, color=green, linewidth=2.8, label="New FT (3ep MSE) - mean")
    ax.plot(x, old_m_show, color=azzurro, linewidth=2.8, label="Old FT (1ep MSE) - mean")
    ax.plot(x, pre_m_show, color=yellow, linewidth=2.8, label="Pretrained base (no FT) - mean")

    suffix = " [log y]" if yscale == "log" else ""
    ax.set_title(
        f"average |delta predicted score| across chapters - 3-way model comparison "
        f"({len(common)} common books){suffix}",
        color="white",
    )
    ax.set_xlabel("chapter transition (i -> i+1)", color="white")
    ax.set_ylabel("Mean |delta predicted score| across pairs", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    if yscale == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25, color="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white", loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, facecolor="black")
    plt.close()
    print(f"[plot] Wrote: {output_path}  (yscale={yscale})")
    print(
        f"[plot] new mean range [{np.nanmin(new_mean):.5f}, {np.nanmax(new_mean):.5f}] | "
        f"old mean range [{np.nanmin(old_mean):.5f}, {np.nanmax(old_mean):.5f}] | "
        f"pre mean range [{np.nanmin(pre_mean):.5f}, {np.nanmax(pre_mean):.5f}]"
    )


def _split(arg: str) -> List[Path]:
    return [Path(s.strip()) for s in arg.split(",") if s.strip()]


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--new-dirs",
        default="data/predictions/prevprefix_full_3ep_mse,data/predictions/prevprefix_test_3ep_mse",
        help="Comma-separated dirs for the NEW fine-tuned model (3ep MSE)",
    )
    ap.add_argument(
        "--old-dirs",
        default="data/predictions/prevprefix_full_1ep",
        help="Comma-separated dirs for the OLD fine-tuned model (1ep MSE)",
    )
    ap.add_argument(
        "--pre-dirs",
        default="data/predictions/pretrained_base",
        help="Comma-separated dirs for the PRETRAINED base model",
    )
    ap.add_argument(
        "--output",
        default="data/evaluation/compare/all_books_avg_change_compare_3way_log.png",
    )
    ap.add_argument("--yscale", choices=["linear", "log"], default="log")
    args = ap.parse_args()

    plot_3way(
        new_dirs=_split(args.new_dirs),
        old_dirs=_split(args.old_dirs),
        pre_dirs=_split(args.pre_dirs),
        output_path=Path(args.output),
        yscale=args.yscale,
    )


if __name__ == "__main__":
    main()
