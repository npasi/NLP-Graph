"""Regenerate data/splits/train_val_test.json from book IDs in data/dataset/texts.

Uses filenames like ``22_A_Christmas_Carol_....txt`` → book_id ``22``.
Split ratio: ~70%% train, ~15%% val, ~15%% test (at least 1 book in val and test when n>=3).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def book_ids_from_texts_dir(texts_dir: Path) -> list[str]:
    ids: set[str] = set()
    for p in texts_dir.glob("*.txt"):
        m = re.match(r"^(\d+)_", p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=int)


def split_ids(ids: list[str]) -> dict[str, list[str]]:
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": ids[:], "val": [], "test": []}
    if n == 2:
        return {"train": [ids[0]], "val": [], "test": [ids[1]]}

    n_val = max(1, round(0.15 * n))
    n_test = max(1, round(0.15 * n))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val

    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    assert len(train) + len(val) + len(test) == n
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    return {"train": train, "val": val, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--texts-dir",
        type=Path,
        default=Path("data/dataset/texts"),
        help="Directory with <book_id>_*.txt files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/splits/train_val_test.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="If out exists, copy to train_val_test.json.bak before overwrite",
    )
    args = ap.parse_args()

    texts_dir = args.texts_dir
    if not texts_dir.is_dir():
        raise SystemExit(f"Not a directory: {texts_dir}")

    ids = book_ids_from_texts_dir(texts_dir)
    splits = split_ids(ids)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.backup and out.is_file():
        bak = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, bak)
        print(f"[init] Backed up to {bak}")

    out.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[init] texts_dir: {texts_dir.resolve()}")
    print(f"[init] books: {len(ids)}")
    print(f"[init] train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    print(f"[init] wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
