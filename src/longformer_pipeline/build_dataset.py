"""Step 3 of the Longformer pipeline: build training dataset (JSONL).

This script joins:
- Ground-truth CSVs (pairwise affinity per chapter)
- Chapter interaction files (per chapter, produced upstream)

It outputs JSONL examples suitable for fine-tuning Longformer.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    fuzz = None
    _HAS_RAPIDFUZZ = False


def _reconfigure_stdio_utf8() -> None:
    # input_builder prints non-ascii; avoid Windows cp1252 crashes
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_CHAPTER_COL_RE = re.compile(r"^chapter_(\d+)$", flags=re.IGNORECASE)


def _norm_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


_LEADING_ARTICLES_RE = re.compile(r"^(the|a|an)\s+", flags=re.IGNORECASE)
_HONORIFICS_RE = re.compile(
    r"^(mr|mrs|ms|miss|dr|sir|madam|lady|lord)\.?\s+",
    flags=re.IGNORECASE,
)


def _name_variants(name: str) -> List[str]:
    """Generate simple alias variants to improve GT↔interactions matching.

    Designed to map cases like:
    - "Ebenezer Scrooge" -> "Scrooge"
    - "Jacob Marley" -> "Marley"
    - "The Ghost of Christmas Past" -> "Ghost of Christmas Past" (helps a bit)
    """
    base = _norm_name(name)
    out = {base}

    # Strip leading articles / honorifics
    s = _LEADING_ARTICLES_RE.sub("", base)
    s = _HONORIFICS_RE.sub("", s)
    out.add(s)

    # Remove punctuation-ish quotes
    s2 = re.sub(r"[\"“”‘’]", "", s).strip()
    out.add(s2)

    parts = [p for p in re.split(r"\s+", s2) if p]
    if len(parts) >= 2:
        out.add(parts[-1])  # last token (surname / head noun)
        out.add(" ".join(parts[-2:]))  # last two tokens

    # Collapse multiple spaces again
    out2 = {re.sub(r"\s+", " ", x).strip() for x in out if x and x.strip()}
    return sorted(out2, key=len, reverse=True)


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    na, nb = _norm_name(a), _norm_name(b)
    return (na, nb) if na <= nb else (nb, na)


def load_gt_csv(csv_path: str) -> List[Dict]:
    """Load a GT CSV and return per-(pair,chapter) examples with valid scores in [0,1]."""
    print(f"[load_gt] File: {csv_path}")

    df = pd.read_csv(csv_path)
    chapter_cols = [c for c in df.columns if _CHAPTER_COL_RE.match(str(c).strip())]
    print(f"[load_gt] Rows: {len(df)}, Chapter columns: {len(chapter_cols)}")

    examples: List[Dict] = []
    count_minus1 = 0
    count_minus10 = 0

    title = str(df.loc[0, "title"]) if "title" in df.columns and len(df) else ""
    author = str(df.loc[0, "author"]) if "author" in df.columns and len(df) else ""

    for _, row in df.iterrows():
        a = str(row["character_1"]).strip()
        b = str(row["character_2"]).strip()
        for col in chapter_cols:
            m = _CHAPTER_COL_RE.match(str(col).strip())
            if not m:
                continue
            chapter_1based = int(m.group(1))
            chapter = chapter_1based - 1
            try:
                val = float(row[col])
            except Exception:
                continue

            if 0.0 <= val <= 1.0:
                prev_score: float | None = None
                if chapter_1based > 1:
                    prev_col = f"chapter_{chapter_1based - 1}"
                    if prev_col in df.columns:
                        try:
                            pv = float(row[prev_col])
                            if 0.0 <= pv <= 1.0:
                                prev_score = float(pv)
                        except Exception:
                            prev_score = None
                examples.append(
                    {
                        "char_a": a,
                        "char_b": b,
                        "chapter": chapter,
                        "score": float(val),
                        "title": title,
                        "author": author,
                        "prev_score": prev_score,
                    }
                )
            elif val == -1:
                count_minus1 += 1
            elif val == -10:
                count_minus10 += 1

    print(f"[load_gt] Valid scores (0-1): {len(examples)}")
    print(f"[load_gt] Skipped -1: {count_minus1}, Skipped -10: {count_minus10}")
    if examples:
        scores = [ex["score"] for ex in examples]
        min_score = min(scores)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        print(
            f"[load_gt] Score distribution: min={min_score:.2f}, "
            f"max={max_score:.2f}, mean={mean_score:.2f}"
        )

    assert len(examples) > 0, f"No valid examples found in {csv_path}"
    for ex in examples:
        assert 0.0 <= ex["score"] <= 1.0, f"Invalid score: {ex['score']}"

    return examples


def _find_interactions_file(
    *,
    interactions_dir: str,
    book_id: str,
    chapter: int,
) -> Optional[Path]:
    """Locate the chapter interactions file, supporting multiple layouts.

    Preferred (per spec):
      {interactions_dir}/{book_id}/chapter_{chapter}_proper_interactions.txt

    Fallback (single-book dumps, e.g. results_christmas_carol/):
      {interactions_dir}/chapter_{chapter}_proper_interactions.txt

    Dataset layout (data/dataset/interactions):
      {interactions_dir}/results_{book_id}/chapters/chapter_{chapter}_proper_interactions.txt
    """
    p1 = (
        Path(interactions_dir)
        / str(book_id)
        / f"chapter_{chapter}_proper_interactions.txt"
    )
    if p1.is_file():
        return p1

    p_chapters = (
        Path(interactions_dir)
        / f"results_{book_id}"
        / "chapters"
        / f"chapter_{chapter}_proper_interactions.txt"
    )
    if p_chapters.is_file():
        return p_chapters

    p_nested = (
        Path(interactions_dir)
        / str(book_id)
        / "chapters"
        / f"chapter_{chapter}_proper_interactions.txt"
    )
    if p_nested.is_file():
        return p_nested

    p2 = Path(interactions_dir) / f"chapter_{chapter}_proper_interactions.txt"
    if p2.is_file():
        return p2

    return None


def _fuzzy_ratio(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ and fuzz is not None:
        return float(fuzz.token_set_ratio(a, b))
    # Fallback: simple similarity (0..100)
    from difflib import SequenceMatcher

    return 100.0 * SequenceMatcher(a=a, b=b).ratio()


def match_gt_to_interactions(
    gt_examples: List[Dict],
    interactions_dir: str,
    book_id: str,
    *,
    fuzzy_threshold: float = 80.0,
) -> List[Dict]:
    """For each GT example, attach the corresponding Longformer text from interactions."""
    _reconfigure_stdio_utf8()
    # Import here so running as a script works even without package install.
    # Ensure src/ is on sys.path.
    this_file = Path(__file__).resolve()
    src_dir = this_file.parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from longformer_pipeline.input_builder import (  # noqa: PLC0415
        add_entity_markers,
        build_longformer_input,
        process_chapter_file,
    )

    print(f"[match] Book {book_id}: {len(gt_examples)} GT examples to match")
    if not _HAS_RAPIDFUZZ:
        print(
            "[match] WARNING: rapidfuzz not installed; using difflib fallback "
            "(matching may be worse). Install rapidfuzz for best results."
        )

    # Pre-load interactions per chapter referenced by GT.
    chapters = sorted({int(ex["chapter"]) for ex in gt_examples})
    chapter_pairs: Dict[int, List[Dict]] = {}
    chapter_index: Dict[int, Dict[Tuple[str, str], Dict]] = {}

    for ch in chapters:
        fpath = _find_interactions_file(
            interactions_dir=interactions_dir, book_id=book_id, chapter=ch
        )
        if fpath is None:
            print(
                f"[match] WARNING: missing interactions file for book={book_id} "
                f"chapter={ch} in {interactions_dir}"
            )
            chapter_pairs[ch] = []
            chapter_index[ch] = {}
            continue

        pairs = process_chapter_file(str(fpath))
        print(f"[match] Chapter {ch}: found {len(pairs)} pairs in interactions file")
        chapter_pairs[ch] = pairs

        idx: Dict[Tuple[str, str], Dict] = {}
        for p in pairs:
            idx[_pair_key(p["char_a"], p["char_b"])] = p
        chapter_index[ch] = idx

    total = len(gt_examples)
    matched = 0
    unmatched: List[Dict] = []
    out: List[Dict] = []

    for ex in gt_examples:
        ch = int(ex["chapter"])
        a = str(ex["char_a"])
        b = str(ex["char_b"])

        found: Optional[Dict] = None

        # 1) Exact normalized pair match
        found = chapter_index.get(ch, {}).get(_pair_key(a, b))

        # 2) Fuzzy match against same-chapter observed pairs
        if found is None and chapter_pairs.get(ch):
            best_score = -1.0
            best_pair: Optional[Dict] = None
            a_vars = _name_variants(a)
            b_vars = _name_variants(b)
            for p in chapter_pairs[ch]:
                pa = _name_variants(p["char_a"])
                pb = _name_variants(p["char_b"])

                # score is best average over variant pairs, considering swap
                best_here = -1.0
                for av in a_vars:
                    for bv in b_vars:
                        for pav in pa:
                            for pbv in pb:
                                s1 = (_fuzzy_ratio(av, pav) + _fuzzy_ratio(bv, pbv)) / 2.0
                                s2 = (_fuzzy_ratio(av, pbv) + _fuzzy_ratio(bv, pav)) / 2.0
                                best_here = max(best_here, s1, s2)
                s = best_here
                if s > best_score:
                    best_score = s
                    best_pair = p
            if best_pair is not None and best_score >= fuzzy_threshold:
                found = best_pair

        if found is None:
            unmatched.append(ex)
            continue

        matched += 1
        prev_score = ex.get("prev_score", None)
        try:
            prev_score_f = float(prev_score) if prev_score is not None else None
        except Exception:
            prev_score_f = None

        marked = add_entity_markers(
            found["raw_text"],
            found["char_a"],
            found["char_b"],
        )
        text_with_prev = build_longformer_input(
            found["char_a"],
            found["char_b"],
            marked,
            prev_score=prev_score_f,
        )
        out.append(
            {
                "book_id": str(book_id),
                "chapter": int(ch),
                "char_a": ex["char_a"],
                "char_b": ex["char_b"],
                "score": float(ex["score"]),
                "text": text_with_prev,
                "prev_score": prev_score_f,
            }
        )

    print(f"[match] Matched: {matched}/{total} ({(matched/total*100.0) if total else 0.0:.1f}%)")
    if unmatched:
        print(f"[match] WARNING: {len(unmatched)} unmatched pairs:")
        for u in unmatched[:50]:
            print(f"[match]   {u['char_a']} ◄──► {u['char_b']} (chapter {u['chapter']})")
        if len(unmatched) > 50:
            print(f"[match]   ... (+{len(unmatched) - 50} more)")

    return out


def _iter_gt_csvs(gt_dir: str) -> List[Path]:
    return sorted(Path(gt_dir).glob("*.csv"))


def book_ids_from_texts_dir(texts_dir: str) -> Set[str]:
    """Collect numeric book IDs from files like ``22_A_Christmas_Carol_....txt``."""
    root = Path(texts_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"texts dir not found: {texts_dir}")
    ids: Set[str] = set()
    for p in root.glob("*.txt"):
        m = re.match(r"^(\d+)_", p.name)
        if m:
            ids.add(m.group(1))
    return ids


def _book_id_from_filename(csv_path: Path) -> str:
    m = re.match(r"^(\d+)_", csv_path.name)
    if m:
        return m.group(1)
    # fallback: leading digits anywhere
    m2 = re.match(r"^(\d+)", csv_path.stem)
    if m2:
        return m2.group(1)
    raise ValueError(f"Cannot parse book_id from GT filename: {csv_path.name}")


def build_full_dataset(
    gt_dir: str,
    interactions_dir: str,
    output_path: str,
    *,
    allowed_book_ids: Optional[Set[str]] = None,
) -> None:
    print(f"\n{'='*60}")
    print("[build] Building full dataset")
    print(f"[build] GT dir: {gt_dir}")
    print(f"[build] Interactions dir: {interactions_dir}")
    if allowed_book_ids is not None:
        print(f"[build] Restricted to {len(allowed_book_ids)} book_id(s) from texts dir")
    print(f"{'='*60}\n")

    csvs_all = _iter_gt_csvs(gt_dir)
    if allowed_book_ids is not None:
        csvs = [
            p
            for p in csvs_all
            if _book_id_from_filename(p) in allowed_book_ids
        ]
        skipped = len(csvs_all) - len(csvs)
        if skipped:
            print(f"[build] Skipped {skipped} GT CSV(s) not in texts whitelist")
    else:
        csvs = csvs_all

    n_books = len(csvs)
    all_examples: List[Dict] = []

    for i, csv_path in enumerate(csvs):
        book_id = _book_id_from_filename(csv_path)
        gt_examples = load_gt_csv(str(csv_path))
        matched = match_gt_to_interactions(gt_examples, interactions_dir, book_id)
        print(f"[build] [{i+1}/{n_books}] Book {book_id}: {len(matched)} examples")
        all_examples.extend(matched)

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    scores = [ex["score"] for ex in all_examples] if all_examples else []
    if scores:
        import math

        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var)
        print(f"\n{'='*60}")
        print("[build] DATASET SUMMARY")
        print(f"[build] Total books processed (after filter): {n_books}")
        print(f"[build] Total examples: {len(all_examples)}")
        print(
            f"[build] Score distribution: min={min(scores):.2f}, "
            f"max={max(scores):.2f}, mean={mean:.2f}, std={std:.2f}"
        )
        print(f"[build] Written to: {output_path}")
        print(f"{'='*60}")
    else:
        print("[build] WARNING: no examples written (0 matched).")


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: str, examples: List[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def split_dataset(full_dataset_path: str, splits_json_path: str, output_dir: str) -> None:
    splits = json.loads(Path(splits_json_path).read_text(encoding="utf-8"))
    train_ids = {str(x) for x in splits.get("train", [])}
    val_ids = {str(x) for x in splits.get("val", [])}
    test_ids = {str(x) for x in splits.get("test", [])}

    print(
        f"[split] Train books: {len(train_ids)}, "
        f"Val books: {len(val_ids)}, Test books: {len(test_ids)}"
    )

    examples = _read_jsonl(full_dataset_path)
    train_ex: List[Dict] = []
    val_ex: List[Dict] = []
    test_ex: List[Dict] = []

    for ex in examples:
        bid = str(ex["book_id"])
        if bid in train_ids:
            train_ex.append(ex)
        elif bid in val_ids:
            val_ex.append(ex)
        elif bid in test_ids:
            test_ex.append(ex)
        else:
            # ignore examples not in any split
            continue

    print(
        f"[split] Train examples: {len(train_ex)}, "
        f"Val examples: {len(val_ex)}, Test examples: {len(test_ex)}"
    )

    out_dir = Path(output_dir)
    _write_jsonl(str(out_dir / "train.jsonl"), train_ex)
    _write_jsonl(str(out_dir / "val.jsonl"), val_ex)
    _write_jsonl(str(out_dir / "test.jsonl"), test_ex)

    # verify no leakage
    train_books = {ex["book_id"] for ex in train_ex}
    val_books = {ex["book_id"] for ex in val_ex}
    test_books = {ex["book_id"] for ex in test_ex}
    assert (
        len(train_books & val_books) == 0
    ), f"Data leakage! Books in both train and val: {train_books & val_books}"
    assert (
        len(train_books & test_books) == 0
    ), f"Data leakage! Books in both train and test: {train_books & test_books}"
    assert (
        len(val_books & test_books) == 0
    ), f"Data leakage! Books in both val and test: {val_books & test_books}"
    print("[split] ✓ No data leakage detected")


if __name__ == "__main__":
    import argparse

    _reconfigure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", default="data/dataset/ground_truth")
    parser.add_argument("--interactions-dir", default="data/dataset/interactions")
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument("--splits-json", default="data/splits/train_val_test.json")
    parser.add_argument(
        "--texts-dir",
        default=None,
        help=(
            "If set, only include books whose ID appears as prefix of a .txt here "
            "(e.g. data/dataset/texts/22_....txt → book 22). Ignores other GT CSVs."
        ),
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Write only full_dataset.jsonl; do not create train/val/test from splits JSON.",
    )
    parser.add_argument(
        "--only-book",
        default=None,
        help="Optional: process only one book_id (for quick testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    full_path = os.path.join(args.output_dir, "full_dataset.jsonl")

    allowed: Optional[Set[str]] = None
    if args.texts_dir:
        allowed = book_ids_from_texts_dir(args.texts_dir)
        print(f"[build] Whitelist from --texts-dir: {len(allowed)} book_id(s)")

    if args.only_book is not None:
        # build a tiny dataset for one book
        if allowed is not None and str(args.only_book) not in allowed:
            raise SystemExit(
                f"book_id={args.only_book} not in texts whitelist from {args.texts_dir}"
            )
        gt_dir = Path(args.gt_dir)
        candidates = sorted(gt_dir.glob(f"{args.only_book}_*.csv"))
        if not candidates:
            raise SystemExit(f"No GT CSV found for book_id={args.only_book} in {gt_dir}")
        csv_path = candidates[0]
        book_id = _book_id_from_filename(csv_path)
        gt_examples = load_gt_csv(str(csv_path))
        matched = match_gt_to_interactions(gt_examples, args.interactions_dir, book_id)
        outp = Path(args.output_dir) / f"{book_id}_dataset.jsonl"
        _write_jsonl(str(outp), matched)
        print(f"[build] Written: {outp} ({len(matched)} examples)")
    else:
        build_full_dataset(
            args.gt_dir,
            args.interactions_dir,
            full_path,
            allowed_book_ids=allowed,
        )
        if args.no_split:
            print("[split] Skipped (--no-split). Use full_dataset.jsonl for training.")
        else:
            split_dataset(full_path, args.splits_json, args.output_dir)

