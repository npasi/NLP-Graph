"""Create a reproducible 85 / 15 / 16 train / val / test split over all 116 books.

Output: data/splits/train_val_test.json
  {
    "train": ["1", "7", ...],   # 85 book_ids
    "val":   ["3", ...],        # 15 book_ids
    "test":  ["2", ...]         # 16 book_ids
  }

Usage:
    python src/build_split.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CORPUS_DIR = _PROJECT_ROOT / "data" / "archive" / "corpus"
CSV_DIR    = _PROJECT_ROOT / "data" / "archive" / "csv"
SPLIT_OUT  = _PROJECT_ROOT / "data" / "splits" / "train_val_test.json"

N_TRAIN, N_VAL, N_TEST = 85, 15, 16
SEED = 42


def _book_id(filename: str):
    m = re.match(r"^(\d+)_", filename)
    return m.group(1) if m else None


def main() -> None:
    corpus_ids = {_book_id(p.name) for p in CORPUS_DIR.glob("*.txt") if _book_id(p.name)}
    csv_ids    = {_book_id(p.name) for p in CSV_DIR.glob("*.csv")    if _book_id(p.name)}
    all_ids    = sorted(corpus_ids & csv_ids, key=lambda x: int(x))

    if len(all_ids) != 116:
        print(f"WARNING: expected 116 books, found {len(all_ids)}")

    rng = random.Random(SEED)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    split = {
        "train": shuffled[:N_TRAIN],
        "val":   shuffled[N_TRAIN : N_TRAIN + N_VAL],
        "test":  shuffled[N_TRAIN + N_VAL :],
    }

    assert len(split["train"]) == N_TRAIN
    assert len(split["val"])   == N_VAL
    assert len(split["test"])  == N_TEST

    SPLIT_OUT.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Split saved to {SPLIT_OUT}")
    print(f"  train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")
    print(f"  train: {split['train']}")
    print(f"  val:   {split['val']}")
    print(f"  test:  {split['test']}")


if __name__ == "__main__":
    main()
