"""3-way comparison: pretrained vs old (1ep MSE) vs new (3ep MSE) on a split.

Reads metrics.json from all_results/<split>/<model>/<book>/metrics.json
and prints a per-book + average comparison.
"""

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


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    _reconfigure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/evaluation/all_results")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--book-ids", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    book_ids = [b.strip() for b in args.book_ids.split(",") if b.strip()]
    root = Path(args.root) / args.split

    rows: List[Dict[str, Any]] = []
    for bid in book_ids:
        pre = _load(root / "pretrained_base" / bid / "metrics.json")
        old = _load(root / "old_1ep_MSE" / bid / "metrics.json")
        new = _load(root / "new_3ep_MSE" / bid / "metrics.json")
        rows.append({
            "book_id": bid,
            "n": (pre or old or new).get("n_cells"),
            "pre_mae": pre.get("mae"),
            "old_mae": old.get("mae"),
            "new_mae": new.get("mae"),
            "pre_mse": pre.get("mse"),
            "old_mse": old.get("mse"),
            "new_mse": new.get("mse"),
            "pre_R": pre.get("pearson"),
            "old_R": old.get("pearson"),
            "new_R": new.get("pearson"),
            "pre_std": pre.get("pred_std"),
            "old_std": old.get("pred_std"),
            "new_std": new.get("pred_std"),
            "gt_std": (pre or old or new).get("gt_std"),
        })

    border = "=" * 124
    print(border)
    print(f"  3-WAY COMPARISON ({args.split.upper()} SET)")
    print(border)
    hdr = f"  {'book':>5} {'n':>5} | {'preMAE':>7} {'oldMAE':>7} {'newMAE':>7} | {'preMSE':>7} {'oldMSE':>7} {'newMSE':>7} | {'preR':>7} {'oldR':>7} {'newR':>7} | {'preStd':>7} {'oldStd':>7} {'newStd':>7} {'gtStd':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _f(x, w=7, p=4):
        return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else " " * w

    def _fr(x, w=7, p=3):
        return f"{x:>+{w}.{p}f}" if isinstance(x, (int, float)) else " " * w

    sums = {k: [] for k in ["pre_mae","old_mae","new_mae","pre_mse","old_mse","new_mse","pre_R","old_R","new_R","pre_std","old_std","new_std"]}
    for r in rows:
        print(
            f"  {r['book_id']:>5} {r['n']:>5} | "
            f"{_f(r['pre_mae'])} {_f(r['old_mae'])} {_f(r['new_mae'])} | "
            f"{_f(r['pre_mse'])} {_f(r['old_mse'])} {_f(r['new_mse'])} | "
            f"{_fr(r['pre_R'])} {_fr(r['old_R'])} {_fr(r['new_R'])} | "
            f"{_f(r['pre_std'])} {_f(r['old_std'])} {_f(r['new_std'])} {_f(r['gt_std'])}"
        )
        for k in sums:
            v = r.get(k)
            if isinstance(v, (int, float)):
                sums[k].append(v)
    print("  " + "-" * (len(hdr) - 2))

    def _avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    a = {k: _avg(v) for k, v in sums.items()}
    print(
        f"  {'AVG':>5} {'':>5} | "
        f"{_f(a['pre_mae'])} {_f(a['old_mae'])} {_f(a['new_mae'])} | "
        f"{_f(a['pre_mse'])} {_f(a['old_mse'])} {_f(a['new_mse'])} | "
        f"{_fr(a['pre_R'])} {_fr(a['old_R'])} {_fr(a['new_R'])} | "
        f"{_f(a['pre_std'])} {_f(a['old_std'])} {_f(a['new_std'])}"
    )
    print(border)
    print(f"  WINNER MAE: {min(['pre','old','new'], key=lambda m: a[f'{m}_mae'])}")
    print(f"  WINNER MSE: {min(['pre','old','new'], key=lambda m: a[f'{m}_mse'])}")
    print(f"  WINNER R  : {max(['pre','old','new'], key=lambda m: a[f'{m}_R'])}")
    print(border)

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["book_id","n","pre_mae","old_mae","new_mae","pre_mse","old_mse","new_mse","pre_R","old_R","new_R","pre_std","old_std","new_std","gt_std"]
        with out.open("w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"[csv] {out}")


if __name__ == "__main__":
    main()
