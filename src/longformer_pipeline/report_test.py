"""Aggregate and print test-set metrics for old vs new runs, write CSV + JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _reconfigure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _load(metrics_dir: Path, bid: str) -> Dict[str, Any]:
    p = metrics_dir / bid / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", default="data/evaluation/prevprefix_test_1ep")
    ap.add_argument("--new-dir", default="data/evaluation/prevprefix_test_3ep_mse")
    ap.add_argument("--book-ids", default="95,99,110")
    ap.add_argument("--out-csv", default="data/evaluation/compare/test_metrics_old_vs_new.csv")
    ap.add_argument("--out-json", default="data/evaluation/compare/test_metrics_old_vs_new.json")
    args = ap.parse_args()

    book_ids = [b.strip() for b in args.book_ids.split(",") if b.strip()]
    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)

    rows: List[Dict[str, Any]] = []
    cumul_old = {"mse": [], "mae": [], "pearson": [], "pred_std": []}
    cumul_new = {"mse": [], "mae": [], "pearson": [], "pred_std": []}

    for bid in book_ids:
        o = _load(old_dir, bid)
        n = _load(new_dir, bid)
        if not o or not n:
            print(f"[skip] {bid}: missing metrics (old={bool(o)}, new={bool(n)})")
            continue
        rows.append(
            {
                "book_id": bid,
                "n_cells": o.get("n_cells"),
                "old_mse": o.get("mse"),
                "new_mse": n.get("mse"),
                "old_mae": o.get("mae"),
                "new_mae": n.get("mae"),
                "old_pearson": o.get("pearson"),
                "new_pearson": n.get("pearson"),
                "old_pred_std": o.get("pred_std"),
                "new_pred_std": n.get("pred_std"),
                "gt_std": o.get("gt_std"),
            }
        )
        for k in cumul_old:
            cumul_old[k].append(o.get(k))
            cumul_new[k].append(n.get(k))

    def _fmt(x):
        return "n/a" if x is None else f"{x:+.4f}" if isinstance(x, float) and -1 <= x <= 1 and abs(x) < 1 else f"{x:.4f}"

    border = "=" * 92
    print(border)
    print("  TEST SET COMPARISON: OLD (1ep MSE) vs NEW (3ep MSE)")
    print(border)
    header = f"  {'book':6} {'n':>6} {'oldMAE':>8} {'newMAE':>8} {'dMAE':>8} {'oldMSE':>8} {'newMSE':>8} {'oldR':>7} {'newR':>7} {'oldStd':>8} {'newStd':>8} {'gtStd':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        d_mae = (r["new_mae"] - r["old_mae"]) if (r["new_mae"] is not None and r["old_mae"] is not None) else None
        print(
            f"  {r['book_id']:>6} {r['n_cells']:>6d} "
            f"{r['old_mae']:>8.4f} {r['new_mae']:>8.4f} "
            f"{d_mae:>+8.4f} "
            f"{r['old_mse']:>8.4f} {r['new_mse']:>8.4f} "
            f"{r['old_pearson']:>+7.3f} {r['new_pearson']:>+7.3f} "
            f"{r['old_pred_std']:>8.4f} {r['new_pred_std']:>8.4f} "
            f"{r['gt_std']:>7.3f}"
        )
    print("  " + "-" * (len(header) - 2))

    def _avg(xs: List[float]) -> float:
        ys = [x for x in xs if x is not None]
        return float(sum(ys) / len(ys)) if ys else float("nan")

    avg_old_mae = _avg(cumul_old["mae"])
    avg_new_mae = _avg(cumul_new["mae"])
    avg_old_mse = _avg(cumul_old["mse"])
    avg_new_mse = _avg(cumul_new["mse"])
    avg_old_r = _avg(cumul_old["pearson"])
    avg_new_r = _avg(cumul_new["pearson"])
    avg_old_std = _avg(cumul_old["pred_std"])
    avg_new_std = _avg(cumul_new["pred_std"])

    print("  AVG    ")
    print(
        f"  {'avg':>6} {'':>6} "
        f"{avg_old_mae:>8.4f} {avg_new_mae:>8.4f} "
        f"{(avg_new_mae - avg_old_mae):>+8.4f} "
        f"{avg_old_mse:>8.4f} {avg_new_mse:>8.4f} "
        f"{avg_old_r:>+7.3f} {avg_new_r:>+7.3f} "
        f"{avg_old_std:>8.4f} {avg_new_std:>8.4f}"
    )
    print(border)

    summary = {
        "books": rows,
        "averages": {
            "old": {
                "mae": avg_old_mae,
                "mse": avg_old_mse,
                "pearson": avg_old_r,
                "pred_std": avg_old_std,
            },
            "new": {
                "mae": avg_new_mae,
                "mse": avg_new_mse,
                "pearson": avg_new_r,
                "pred_std": avg_new_std,
            },
            "delta": {
                "mae": avg_new_mae - avg_old_mae,
                "mse": avg_new_mse - avg_old_mse,
                "pearson": avg_new_r - avg_old_r,
                "pred_std": avg_new_std - avg_old_std,
            },
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[report] JSON written: {out_json}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "book_id",
        "n_cells",
        "old_mae",
        "new_mae",
        "old_mse",
        "new_mse",
        "old_pearson",
        "new_pearson",
        "old_pred_std",
        "new_pred_std",
        "gt_std",
    ]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[report] CSV written: {out_csv}")


if __name__ == "__main__":
    main()
