from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_booknlp_root(book_id: str) -> Path:
    return Path("data") / "output" / "booknlp" / str(book_id)


def _parse_chapter_id(run_dir_name: str) -> Optional[int]:
    # Expected: booknlp_<bookid>_chapter_0000
    parts = run_dir_name.split("_")
    if len(parts) < 2:
        return None
    last = parts[-1]
    if not last.isdigit():
        return None
    return int(last)


def _read_book_meta(book_path: Path) -> Dict[str, Any]:
    return json.loads(book_path.read_text(encoding="utf-8", errors="replace"))


def _sum_mentions(mentions: Dict[str, Any]) -> Tuple[int, int, int]:
    proper = sum(int(m.get("c", 0) or 0) for m in (mentions.get("proper", []) or []))
    common = sum(int(m.get("c", 0) or 0) for m in (mentions.get("common", []) or []))
    pronoun = sum(int(m.get("c", 0) or 0) for m in (mentions.get("pronoun", []) or []))
    return proper, common, pronoun


def _aliases_from_mentions(mentions: Dict[str, Any]) -> List[str]:
    # Keep aliases ordered by frequency (descending) across mention types.
    freq: Dict[str, int] = {}
    for key in ("proper", "common", "pronoun"):
        for m in (mentions.get(key, []) or []):
            name = str(m.get("n", "") or "").strip()
            if not name:
                continue
            freq[name] = freq.get(name, 0) + int(m.get("c", 0) or 0)
    return [n for n, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0].lower()))]


def _canonical_name(mentions: Dict[str, Any], coref_id: Any) -> str:
    # Prefer the most frequent PROPER name; then COMMON; else a stable fallback.
    for key in ("proper", "common"):
        items = (mentions.get(key, []) or [])
        if items:
            best = max(items, key=lambda m: int(m.get("c", 0) or 0))
            name = str(best.get("n", "") or "").strip()
            if name:
                return name
    return f"coref_{coref_id}"


def build_character_dict_for_chapter(book_meta: Dict[str, Any]) -> Dict[str, Any]:
    chars = book_meta.get("characters", []) or []
    out: Dict[str, Any] = {}
    for c in chars:
        coref_id = c.get("id")
        mentions = c.get("mentions", {}) or {}
        proper, common, pronoun = _sum_mentions(mentions)
        aliases = _aliases_from_mentions(mentions)
        canonical = _canonical_name(mentions, coref_id)

        out[canonical] = {
            "coref_id": coref_id,
            "count": int(c.get("count", proper + common + pronoun) or 0),
            "mentions_total": int(proper + common + pronoun),
            "mentions_proper": int(proper),
            "mentions_common": int(common),
            "mentions_pronoun": int(pronoun),
            "aliases": aliases,
        }
    return out


def extract_characters_per_chapter(
    *,
    booknlp_root: Path,
) -> Dict[str, Dict[str, Any]]:
    chapters: Dict[str, Dict[str, Any]] = {}

    run_dirs = sorted([p for p in booknlp_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for run_dir in run_dirs:
        chapter_id = _parse_chapter_id(run_dir.name)
        if chapter_id is None:
            continue
        book_files = list(run_dir.glob("*.book"))
        if not book_files:
            logger.warning("No .book file found in %s (skipping)", run_dir)
            continue
        # There should be exactly one .book per run folder.
        book_path = sorted(book_files)[0]
        meta = _read_book_meta(book_path)
        chapters[str(chapter_id)] = build_character_dict_for_chapter(meta)

    return chapters


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract per-chapter character counts + aliases from BookNLP outputs.")
    p.add_argument("--book-id", required=True, help="Book id (e.g. 46).")
    p.add_argument("--booknlp-root", default=None, help="Root with per-chapter BookNLP folders.")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <booknlp-root>/characters_by_chapter.json).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    book_id = str(args.book_id)
    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else _default_booknlp_root(book_id)
    if not booknlp_root.exists():
        raise FileNotFoundError(str(booknlp_root))

    chapters = extract_characters_per_chapter(booknlp_root=booknlp_root)
    out_path = Path(args.output) if args.output else (booknlp_root / "characters_by_chapter.json")
    out_path.write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d chapter(s) to %s", len(chapters), out_path.resolve())


if __name__ == "__main__":
    main()

