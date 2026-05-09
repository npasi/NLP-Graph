from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_booknlp_root(book_id: str) -> Path:
    return Path("data") / "output" / "booknlp" / str(book_id)


def _default_chapters_root(book_id: str) -> Path:
    # Chapters are written by splitters under:
    #   data/output/chapters/<book_id>/chapters/chapter_000.txt ...
    return Path("data") / "output" / "chapters" / str(book_id) / "chapters"


def _parse_chapter_id(run_dir_name: str) -> Optional[int]:
    # Expected: booknlp_<bookid>_chapter_0000
    parts = run_dir_name.split("_")
    if len(parts) < 2:
        return None
    last = parts[-1]
    if not last.isdigit():
        return None
    return int(last)


def _read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    return header, rows


def _read_book_meta(book_path: Path) -> Dict[str, Any]:
    return json.loads(book_path.read_text(encoding="utf-8", errors="replace"))


_GENERIC_CHARACTER_NAMES = {
    "my readers",
    "each other",
    "no one",
    "someone",
    "somebody",
    "everyone",
    "everybody",
    "anyone",
    "anything",
    "nothing",
    "you",
    "your",
    "yours",
    "yourself",
    "myself",
    "himself",
    "herself",
    "themselves",
    "it",
    "its",
    "we",
    "us",
    "our",
    "ourselves",
}


def _is_tagged_character(name: str) -> bool:
    n = (name or "").strip()
    lower = n.lower()
    if not n or n.startswith("coref_"):
        return False
    if lower in _GENERIC_CHARACTER_NAMES:
        return False
    return True


def _canonical_from_mentions(mentions: Dict[str, Any], coref_id: str) -> str:
    best_name = ""
    best_c = -1
    for key in ("proper", "common"):
        for m in (mentions.get(key, []) or []):
            name = str(m.get("n", "") or "").strip()
            cnt = int(m.get("c", 0) or 0)
            if name and cnt > best_c:
                best_name, best_c = name, cnt
    return best_name or f"coref_{coref_id}"


def _looks_like_named_common(name: str) -> bool:
    if not name:
        return False
    tokens = [tok for tok in name.replace("'", "").split() if tok.isalpha()]
    return any(tok[0].isupper() for tok in tokens if tok)


def _normalize_mention_text(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("'s", "")
    normalized = normalized.replace("the ", "")
    normalized = normalized.replace("a ", "")
    normalized = normalized.replace("an ", "")
    normalized = normalized.replace("’s", "")
    normalized = normalized.replace("--", " ")
    normalized = normalized.replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized


def _build_canonical_name_map(characters: List[Dict[str, Any]]) -> Dict[str, str]:
    """Create a stable canonical name for each BookNLP character cluster.

    BookNLP clusters characters by coreference ID. We prefer proper names
    when available, but we also try to match common-name clusters against
    proper-name aliases from other clusters when the normalized string is
    effectively the same character.
    """
    canon: Dict[str, str] = {}
    # Build a map from normalized alias -> canonical proper-name cluster name.
    alias_to_proper: Dict[str, str] = {}
    cluster_aliases: Dict[str, List[str]] = {}

    for c in characters:
        cid = str(c.get("id"))
        mentions = c.get("mentions", {}) or {}
        name = _canonical_from_mentions(mentions, cid)
        if not _is_tagged_character(name):
            continue

        proper_count = sum(int(m.get("c", 0) or 0) for m in (mentions.get("proper", []) or []))
        common_count = sum(int(m.get("c", 0) or 0) for m in (mentions.get("common", []) or []))
        cluster_aliases[cid] = []

        for key in ("proper", "common"):
            for m in (mentions.get(key, []) or []):
                alias = str(m.get("n", "") or "").strip()
                if alias:
                    cluster_aliases[cid].append(alias)

        if proper_count > 0:
            canon[cid] = name
            for alias in cluster_aliases[cid]:
                normalized = _normalize_mention_text(alias)
                alias_to_proper.setdefault(normalized, name)

    for c in characters:
        cid = str(c.get("id"))
        mentions = c.get("mentions", {}) or {}
        if cid in canon:
            continue

        name = _canonical_from_mentions(mentions, cid)
        if not _is_tagged_character(name):
            continue

        common_count = sum(int(m.get("c", 0) or 0) for m in (mentions.get("common", []) or []))
        proper_count = sum(int(m.get("c", 0) or 0) for m in (mentions.get("proper", []) or []))
        aliases = cluster_aliases.get(cid, [])
        normalized_aliases = [_normalize_mention_text(alias) for alias in aliases]

        # If this cluster has no proper name but one of its normalized aliases
        # matches a proper-name alias from another cluster, adopt that proper
        # canonical name.
        mapped = None
        for normalized in normalized_aliases:
            if normalized in alias_to_proper:
                mapped = alias_to_proper[normalized]
                break

        if mapped:
            canon[cid] = mapped
        elif common_count >= 10 and _looks_like_named_common(name):
            canon[cid] = name
    return canon


def build_global_coref_canon(booknlp_root: Path) -> Dict[str, str]:
    best: Dict[str, Tuple[str, int]] = {}  # id -> (name, score)
    for run_dir in sorted([p for p in booknlp_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        book_files = sorted(run_dir.glob("*.book"))
        if not book_files:
            continue
        meta = _read_book_meta(book_files[0])
        for c in (meta.get("characters", []) or []):
            cid = str(c.get("id"))
            mentions = c.get("mentions", {}) or {}
            name = _canonical_from_mentions(mentions, cid)
            if not _is_tagged_character(name):
                continue
            score = 0
            for key in ("proper", "common"):
                score += sum(int(m.get("c", 0) or 0) for m in (mentions.get(key, []) or []))
            prev = best.get(cid)
            if prev is None or score > prev[1]:
                best[cid] = (name, score)
    return {cid: name for cid, (name, _score) in best.items()}


def _idx(header: List[str], col: str) -> int:
    try:
        return header.index(col)
    except ValueError:
        raise KeyError(f"Missing column '{col}'")


def _build_tok2sent_and_sent_spans(tokens_rows: List[List[str]], header: List[str]) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]:
    sent_i = _idx(header, "sentence_ID")
    tid_doc_i = _idx(header, "token_ID_within_document")
    onset_i = _idx(header, "byte_onset")
    offset_i = _idx(header, "byte_offset")

    tok2sent: Dict[int, int] = {}
    sent2span: Dict[int, Tuple[int, int]] = {}
    for r in tokens_rows:
        try:
            sid = int(r[sent_i])
            tid = int(r[tid_doc_i])
            onset = int(r[onset_i])
            off = int(r[offset_i])
        except Exception:
            continue

        tok2sent[tid] = sid
        prev = sent2span.get(sid)
        if prev is None:
            sent2span[sid] = (onset, off)
        else:
            sent2span[sid] = (min(prev[0], onset), max(prev[1], off))
    return tok2sent, sent2span


def _build_sentence_to_corefs(entities_rows: List[List[str]], tok2sent: Dict[int, int]) -> Dict[int, set[str]]:
    # .entities: COREF start_token end_token ...
    sent2corefs: Dict[int, set[str]] = {}
    for r in entities_rows:
        if len(r) < 3:
            continue
        coref = str(r[0])
        try:
            start = int(r[1])
            end = int(r[2])
        except Exception:
            continue
        for tid in range(start, end + 1):
            sid = tok2sent.get(tid)
            if sid is None:
                continue
            sent2corefs.setdefault(sid, set()).add(coref)
    return sent2corefs


def _pair_key(a_name: str, b_name: str, a_id: str, b_id: str) -> str:
    # user-requested key format: cha_a.char_b,char_a_id,char_b_id
    return f"{a_name}.{b_name},{a_id},{b_id}"


def extract_sentences_by_pair_for_chapter(
    *,
    run_dir: Path,
    chapter_txt: Path,
    global_canon: Dict[str, str],
) -> Dict[str, List[str]]:
    tokens_path = sorted(run_dir.glob("*.tokens"))[0]
    entities_path = sorted(run_dir.glob("*.entities"))[0]
    book_path = sorted(run_dir.glob("*.book"))[0]

    chapter_text = chapter_txt.read_text(encoding="utf-8", errors="replace")

    tok_header, tok_rows = _read_tsv(tokens_path)
    tok2sent, sent2span = _build_tok2sent_and_sent_spans(tok_rows, tok_header)

    _, ent_rows = _read_tsv(entities_path)
    sent2corefs = _build_sentence_to_corefs(ent_rows, tok2sent)

    out_lists: Dict[str, List[str]] = {}

    for sid, corefs in sent2corefs.items():
        # map to canonical names and drop globally-untagged
        tagged: List[Tuple[str, str]] = []
        for cid in corefs:
            name = global_canon.get(cid)
            if not name:
                continue
            if not _is_tagged_character(name):
                continue
            tagged.append((name, cid))

        # only keep sentences with at least two distinct characters
        uniq = sorted(set(tagged), key=lambda x: (x[0].lower(), x[1]))
        if len(uniq) < 2:
            continue

        span = sent2span.get(sid)
        if not span:
            continue
        onset, off = span
        sent_text = chapter_text[onset:off].strip()
        if not sent_text:
            continue

        # unordered character pairs within the same sentence
        for (n1, id1), (n2, id2) in combinations(uniq, 2):
            (a_name, a_id), (b_name, b_id) = sorted([(n1, id1), (n2, id2)], key=lambda x: (x[0].lower(), x[1]))
            k = _pair_key(a_name, b_name, a_id, b_id)
            out_lists.setdefault(k, []).append(sent_text)

    return out_lists


def _extract_sentences_by_pair_for_book(
    *,
    run_dir: Path,
    book_txt: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract sentences by character pairs for the entire book.
    This is similar to chapter extraction but processes the full text once,
    which enables better coreference resolution for repeated character mentions.
    """
    tokens_path = sorted(run_dir.glob("*.tokens"))[0]
    entities_path = sorted(run_dir.glob("*.entities"))[0]
    book_path = sorted(run_dir.glob("*.book"))[0]

    book_text = book_txt.read_text(encoding="utf-8", errors="replace")

    tok_header, tok_rows = _read_tsv(tokens_path)
    tok2sent, sent2span = _build_tok2sent_and_sent_spans(tok_rows, tok_header)

    _, ent_rows = _read_tsv(entities_path)
    sent2corefs = _build_sentence_to_corefs(ent_rows, tok2sent)

    # Build the canonical mapping for every character cluster in the book.
    meta = _read_book_meta(book_path)
    id_to_name = _build_canonical_name_map((meta.get("characters", []) or []))

    out_lists: Dict[str, List[Dict[str, Any]]] = {}

    for sid, corefs in sent2corefs.items():
        tagged: List[Tuple[str, str]] = []
        for cid in corefs:
            name = id_to_name.get(cid)
            if not name:
                continue
            tagged.append((name, cid))

        uniq = sorted(set(tagged), key=lambda x: (x[0].lower(), x[1]))
        if len(uniq) < 2:
            continue

        span = sent2span.get(sid)
        if not span:
            continue
        onset, off = span
        sent_text = book_text[onset:off].strip()
        if not sent_text:
            continue

        for (n1, id1), (n2, id2) in combinations(uniq, 2):
            (a_name, b_name), (a_id, b_id) = sorted([(n1, id1), (n2, id2)], key=lambda x: (x[0].lower(), x[1]))
            k = _pair_key(a_name, b_name, a_id, b_id)
            out_lists.setdefault(k, []).append({"text": sent_text, "sentence_id": sid})

    return out_lists


def extract_sentences_by_pair_by_chapter(
    *,
    book_id: str,
    booknlp_root: Path,
    chapters_root: Path,
) -> Dict[str, Dict[str, List[str]]]:
    global_canon = build_global_coref_canon(booknlp_root)

    chapters: Dict[str, Dict[str, List[str]]] = {}
    run_dirs = sorted([p for p in booknlp_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for run_dir in run_dirs:
        ch_id = _parse_chapter_id(run_dir.name)
        if ch_id is None:
            continue
        ch_txt = chapters_root / f"chapter_{ch_id:03d}.txt"
        if not ch_txt.exists():
            raise FileNotFoundError(str(ch_txt))
        chapters[str(ch_id)] = extract_sentences_by_pair_for_chapter(
            run_dir=run_dir,
            chapter_txt=ch_txt,
            global_canon=global_canon,
        )
        logger.info("chapter=%s pairs=%d", ch_id, len(chapters[str(ch_id)]))

    return chapters


def extract_sentences_by_pair_for_book(
    *,
    run_dir: Path,
    book_txt: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract full-book interactions for all co-occurring character pairs."""
    return _extract_sentences_by_pair_for_book(run_dir=run_dir, book_txt=book_txt)


def extract_sentences_by_pair_by_book(
    *,
    run_dir: Path,
    book_txt: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(str(run_dir))
    return _extract_sentences_by_pair_for_book(run_dir=run_dir, book_txt=book_txt)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build unordered character-pair interactions from BookNLP output."
    )
    p.add_argument("--book-id", required=True, help="Book identifier used for default paths.")
    p.add_argument("--booknlp-root", default=None, help="Root of BookNLP output directories.")
    p.add_argument("--chapters-root", default=None, help="Root of chapter text files for chapter mode.")
    p.add_argument("--mode", choices=["chapter", "book"], default="chapter", help="Extraction mode: per-chapter or full-book.")
    p.add_argument("--run-id", default=None, help="BookNLP run directory name for full-book mode (e.g. full_46). If omitted, the tool will infer a single subdirectory.")
    p.add_argument("--book-txt", default=None, help="Path to the full-book text file for full-book mode.")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <booknlp-root>/sentences_by_pair_by_chapter.json or <booknlp-root>/sentences_by_pair_by_book.json).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _infer_single_run_dir(booknlp_root: Path) -> Path:
    run_dirs = [p for p in booknlp_root.iterdir() if p.is_dir()]
    if len(run_dirs) != 1:
        raise ValueError(
            f"Expected exactly one BookNLP run directory in {booknlp_root}, found {len(run_dirs)}."
        )
    return run_dirs[0]


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

    if args.mode == "chapter":
        chapters_root = Path(args.chapters_root) if args.chapters_root else _default_chapters_root(book_id)
        if not chapters_root.exists():
            raise FileNotFoundError(str(chapters_root))
        out = extract_sentences_by_pair_by_chapter(
            book_id=book_id,
            booknlp_root=booknlp_root,
            chapters_root=chapters_root,
        )
        default_output = booknlp_root / "sentences_by_pair_by_chapter.json"
        logger.info("Extracted chapter-level character pairs for %s", book_id)
    else:
        if not args.book_txt:
            raise ValueError("--book-txt is required when --mode book is selected.")
        book_txt = Path(args.book_txt)
        if not book_txt.exists():
            raise FileNotFoundError(str(book_txt))

        run_dir = Path(args.run_id) if args.run_id else _infer_single_run_dir(booknlp_root)
        if not run_dir.is_absolute():
            run_dir = booknlp_root / run_dir
        if not run_dir.exists():
            raise FileNotFoundError(str(run_dir))

        out = extract_sentences_by_pair_by_book(run_dir=run_dir, book_txt=book_txt)
        default_output = booknlp_root / "sentences_by_pair_by_book.json"
        logger.info("Extracted full-book character pairs from %s", run_dir)

    out_path = Path(args.output) if args.output else default_output
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d keys to %s", len(out), out_path.resolve())


if __name__ == "__main__":
    main()

