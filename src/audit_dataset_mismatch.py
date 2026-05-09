from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _data_dir() -> Path:
    return _PROJECT_ROOT / "data"


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    # keep letters/numbers, turn everything else into spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_book_id_from_filename(name: str) -> Optional[str]:
    m = re.match(r"^(\d+)_", name)
    return m.group(1) if m else None


def _read_book_meta(book_path: Path) -> dict:
    return json.loads(book_path.read_text(encoding="utf-8", errors="replace"))


def _canonical_from_mentions(mentions: dict, coref_id: str) -> str:
    best_name = ""
    best_c = -1
    for key in ("proper", "common"):
        for m in (mentions.get(key, []) or []):
            name = str(m.get("n", "") or "").strip()
            cnt = int(m.get("c", 0) or 0)
            if name and cnt > best_c:
                best_name, best_c = name, cnt
    return best_name or f"coref_{coref_id}"


def _list_run_dirs(booknlp_root: Path) -> List[Path]:
    if not booknlp_root.is_dir():
        return []
    return sorted([p for p in booknlp_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def _chapter_character_names_from_run(run_dir: Path) -> List[Tuple[str, str]]:
    """Return [(canonical_name, coref_id), ...] for one chapter run dir."""
    book_files = sorted(run_dir.glob("*.book"))
    if not book_files:
        return []
    meta = _read_book_meta(book_files[0])
    out: List[Tuple[str, str]] = []
    for c in (meta.get("characters", []) or []):
        cid = str(c.get("id"))
        mentions = c.get("mentions", {}) or {}
        name = _canonical_from_mentions(mentions, cid)
        out.append((name, cid))
    return out


def _chapter_character_aliases_from_run(run_dir: Path) -> List[Tuple[str, Set[str]]]:
    """Return [(rep_name_norm, {alias_norm, ...}), ...] for one chapter.

    - rep_name_norm is the normalized canonical name chosen by `_canonical_from_mentions`.
    - alias set includes normalized surface forms from proper/common/pronoun mentions too.
    """
    book_files = sorted(run_dir.glob("*.book"))
    if not book_files:
        return []
    meta = _read_book_meta(book_files[0])
    out: List[Tuple[str, Set[str]]] = []
    for c in (meta.get("characters", []) or []):
        cid = str(c.get("id"))
        mentions = c.get("mentions", {}) or {}
        rep = _norm_name(_canonical_from_mentions(mentions, cid))
        aliases: Set[str] = set()
        for key in ("proper", "common", "pronoun"):
            for m in (mentions.get(key, []) or []):
                n = _norm_name(str(m.get("n", "") or ""))
                if n:
                    aliases.add(n)
        if rep:
            aliases.add(rep)
        out.append((rep, aliases))
    return out


def count_unnamed_clusters(booknlp_root: Path) -> Tuple[int, int]:
    """Return (unnamed, total) over all chapter-level character clusters."""
    unnamed = 0
    total = 0
    for run_dir in _list_run_dirs(booknlp_root):
        for name, _cid in _chapter_character_names_from_run(run_dir):
            total += 1
            if name.strip().startswith("coref_"):
                unnamed += 1
    return unnamed, total


def build_global_tagged_name_inventory(booknlp_root: Path) -> Set[str]:
    """Return set of normalized canonical names for all *tagged* (non-coref_) characters seen."""
    names: Set[str] = set()
    for run_dir in _list_run_dirs(booknlp_root):
        for name, _cid in _chapter_character_names_from_run(run_dir):
            if not name or name.strip().startswith("coref_"):
                continue
            names.add(_norm_name(name))
    return names


def build_possible_pairs_from_chapter_books(booknlp_root: Path) -> Set[Tuple[str, str]]:
    """Set of normalized unordered (name_a, name_b) pairs that would appear in step4's key universe.

    We intentionally ignore the exact step4 key string because it uses '.' inside names (e.g. 'Mrs.')
    which makes parsing ambiguous. For mismatch % we only need name-level existence.
    """
    pairs: Set[Tuple[str, str]] = set()
    for run_dir in _list_run_dirs(booknlp_root):
        chars = [
            _norm_name(name)
            for (name, _cid) in _chapter_character_names_from_run(run_dir)
            if name and not name.strip().startswith("coref_")
        ]
        chars = sorted(set(c for c in chars if c))
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                a, b = chars[i], chars[j]
                pairs.add((a, b))
    return pairs


def build_gt_pair_matcher(booknlp_root: Path) -> List[Tuple[Set[str], Set[str]]]:
    """Precompute per-chapter alias->rep inventories.

    Returns list over chapters of (rep_set, alias_to_rep_set) represented as:
      - rep_set: set of rep_name_norm for non-coref reps
      - alias_to_reps: dict-like collapsed into list of tuples handled on the fly

    We keep it simple: for each chapter we keep the list of (rep, aliases).
    """
    chapters: List[Tuple[Set[str], List[Tuple[str, Set[str]]]]] = []
    for run_dir in _list_run_dirs(booknlp_root):
        items = _chapter_character_aliases_from_run(run_dir)
        # only allow named reps for pairing universe (mirrors step4 dropping coref_* names)
        named = [(rep, aliases) for rep, aliases in items if rep and not rep.startswith("coref ")]
        named = [(rep, aliases) for rep, aliases in named if not rep.startswith("coref_")]
        rep_set = set(rep for rep, _a in named if rep)
        chapters.append((rep_set, named))
    return chapters  # type: ignore[return-value]


def gt_pair_is_matched(
    *,
    gt_a: str,
    gt_b: str,
    chapter_inventories: List[Tuple[Set[str], List[Tuple[str, Set[str]]]]],
) -> bool:
    """Return True if GT pair can be grounded to any chapter via alias matching."""
    for rep_set, items in chapter_inventories:
        reps_a = {rep for rep, aliases in items if gt_a in aliases}
        reps_b = {rep for rep, aliases in items if gt_b in aliases}
        if not reps_a or not reps_b:
            continue
        # if any (rep_a, rep_b) co-occur (and are distinct), pair exists in step4 universe
        for ra in reps_a:
            for rb in reps_b:
                if ra == rb:
                    continue
                a, b = (ra, rb) if ra <= rb else (rb, ra)
                if a in rep_set and b in rep_set:
                    return True
    return False


def iter_gt_pairs(csv_path: Path) -> Iterable[Tuple[str, str]]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            a = _norm_name(row.get("character_1", ""))
            b = _norm_name(row.get("character_2", ""))
            if not a or not b:
                continue
            if a <= b:
                yield (a, b)
            else:
                yield (b, a)


@dataclass(frozen=True)
class BookAudit:
    book_id: str
    unnamed_clusters: int
    total_clusters: int
    unnamed_pct: float
    gt_pairs: int
    gt_pairs_unmatched: int
    gt_pairs_unmatched_pct: float


def audit_book(*, book_id: str, booknlp_root: Path, gt_csv: Path) -> BookAudit:
    unnamed, total = count_unnamed_clusters(booknlp_root)
    unnamed_pct = (unnamed / total * 100.0) if total else 0.0

    chapter_inventories = build_gt_pair_matcher(booknlp_root)
    gt_pairs_list = list(iter_gt_pairs(gt_csv))
    gt_pairs = len(gt_pairs_list)
    gt_unmatched = sum(
        1
        for (a, b) in gt_pairs_list
        if not gt_pair_is_matched(gt_a=a, gt_b=b, chapter_inventories=chapter_inventories)
    )
    gt_unmatched_pct = (gt_unmatched / gt_pairs * 100.0) if gt_pairs else 0.0

    return BookAudit(
        book_id=book_id,
        unnamed_clusters=unnamed,
        total_clusters=total,
        unnamed_pct=unnamed_pct,
        gt_pairs=gt_pairs,
        gt_pairs_unmatched=gt_unmatched,
        gt_pairs_unmatched_pct=gt_unmatched_pct,
    )


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Audit BookNLP naming + GT-pair mismatch for dataset.")
    p.add_argument("--book-ids", default=None, help="Comma-separated list of book_ids to audit (default: detect).")
    p.add_argument("--data-root", default=None, help="Override data root (default: ./data).")
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    args = p.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else _data_dir()
    gt_dir = data_root / "dataset" / "ground_truth"
    booknlp_out_root = data_root / "output" / "booknlp"

    if args.book_ids:
        book_ids = [b.strip() for b in args.book_ids.split(",") if b.strip()]
    else:
        # Detect by existing BookNLP output folders under data/output/booknlp/<book_id>
        book_ids = sorted([p.name for p in booknlp_out_root.iterdir() if p.is_dir() and p.name.isdigit()], key=int)

    audits: List[dict] = []
    for bid in book_ids:
        gt = next(iter(gt_dir.glob(f"{bid}_*.csv")), None)
        if gt is None:
            audits.append({"book_id": bid, "status": "missing_gt"})
            continue
        booknlp_root = booknlp_out_root / bid
        if not booknlp_root.exists():
            audits.append({"book_id": bid, "status": "missing_booknlp"})
            continue
        a = audit_book(book_id=bid, booknlp_root=booknlp_root, gt_csv=gt)
        audits.append({**a.__dict__, "status": "ok", "gt_file": gt.name})

    # Aggregate over audited books
    ok = [a for a in audits if a.get("status") == "ok"]
    if ok:
        total_gt_pairs = sum(int(a["gt_pairs"]) for a in ok)
        total_unmatched = sum(int(a["gt_pairs_unmatched"]) for a in ok)
        overall_unmatched_pct = (total_unmatched / total_gt_pairs * 100.0) if total_gt_pairs else 0.0

        total_clusters = sum(int(a["total_clusters"]) for a in ok)
        total_unnamed = sum(int(a["unnamed_clusters"]) for a in ok)
        overall_unnamed_pct = (total_unnamed / total_clusters * 100.0) if total_clusters else 0.0
    else:
        overall_unmatched_pct = 0.0
        overall_unnamed_pct = 0.0

    report = {
        "books_total_detected": len(book_ids),
        "books_ok": len(ok),
        "overall_gt_pairs_unmatched_pct": overall_unmatched_pct,
        "overall_unnamed_clusters_pct": overall_unnamed_pct,
        "books": audits,
    }

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Minimal console summary
    print(json.dumps(
        {
            "books_ok": report["books_ok"],
            "overall_gt_pairs_unmatched_pct": round(report["overall_gt_pairs_unmatched_pct"], 3),
            "overall_unnamed_clusters_pct": round(report["overall_unnamed_clusters_pct"], 3),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

