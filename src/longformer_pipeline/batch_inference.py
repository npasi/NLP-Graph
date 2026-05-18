"""Batch inference over many books with the model loaded once.

Reuses ``inference.predict_book`` but passes in a single shared
``LongformerScorer`` to avoid 42x model loading overhead.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional


def _reconfigure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _ensure_src_on_path() -> None:
    this_file = Path(__file__).resolve()
    src_dir = this_file.parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _find_gt_csv(gt_dir: Path, book_id: str) -> Optional[Path]:
    for p in gt_dir.glob(f"{book_id}_*.csv"):
        return p
    return None


def _ids_from_interactions(interactions_dir: Path) -> List[str]:
    ids: List[str] = []
    for d in sorted(interactions_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^results_(\d+)$", d.name)
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else s)


def main() -> None:
    _reconfigure_stdio_utf8()
    _ensure_src_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--interactions-dir", default="data/dataset/interactions")
    ap.add_argument("--gt-dir", default="data/dataset/ground_truth")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--book-ids", default=None, help="Comma-separated, default=auto from interactions")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--norm",
        choices=("sigmoid", "clip"),
        default="sigmoid",
    )
    args = ap.parse_args()

    from longformer_pipeline.inference import predict_book  # noqa: PLC0415
    from longformer_pipeline.scorer import LongformerScorer  # noqa: PLC0415

    print(f"[batch] Loading scorer from: {args.model_dir}")
    t0 = time.time()
    scorer = LongformerScorer(args.model_dir, device=args.device, norm=args.norm)
    print(f"[batch] Scorer ready in {time.time() - t0:.1f}s")

    interactions_dir = Path(args.interactions_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.book_ids:
        ids = [x.strip() for x in args.book_ids.split(",") if x.strip()]
    else:
        ids = _ids_from_interactions(interactions_dir)
    print(f"[batch] Books to process: {len(ids)} -> {ids}")

    t_start = time.time()
    failures: List[str] = []
    for i, bid in enumerate(ids, start=1):
        out_csv = out_dir / f"{bid}_predicted.csv"
        gt_csv = _find_gt_csv(gt_dir, bid)
        gt_arg = str(gt_csv) if gt_csv else None
        print(f"\n[batch] [{i}/{len(ids)}] book={bid}  gt={'yes' if gt_csv else 'NO'}  -> {out_csv}")
        try:
            predict_book(
                book_id=bid,
                interactions_dir=str(interactions_dir),
                model_dir=args.model_dir,
                output_csv=str(out_csv),
                gt_csv=gt_arg,
                device=args.device,
                batch_size=args.batch_size,
                norm=args.norm,
                scorer=scorer,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[batch] FAIL on book {bid}: {exc!r}")
            failures.append(bid)

    elapsed = time.time() - t_start
    print(f"\n[batch] DONE in {elapsed/60:.1f} min over {len(ids)} books")
    if failures:
        print(f"[batch] Failures ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
