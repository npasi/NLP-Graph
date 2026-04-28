from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_booknlp_root(book_id: str) -> Path:
    return Path("NLP-Graph") / "data" / "booknlp_chapter_output" / str(book_id)


def _default_chapters_root(book_id: str) -> Path:
    return Path("NLP-Graph") / "data" / "books" / str(book_id) / "chapters"


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


def _is_tagged_character(name: str) -> bool:
    n = (name or "").strip()
    return bool(n) and not n.startswith("coref_")


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
) -> Dict[str, str]:
    tokens_path = sorted(run_dir.glob("*.tokens"))[0]
    entities_path = sorted(run_dir.glob("*.entities"))[0]
    book_path = sorted(run_dir.glob("*.book"))[0]

    chapter_text = chapter_txt.read_text(encoding="utf-8", errors="replace")

    tok_header, tok_rows = _read_tsv(tokens_path)
    tok2sent, sent2span = _build_tok2sent_and_sent_spans(tok_rows, tok_header)

    _, ent_rows = _read_tsv(entities_path)
    sent2corefs = _build_sentence_to_corefs(ent_rows, tok2sent)

    # Build the full set of character ids present in this chapter (as extensive as possible),
    # but name them using the global canonical mapping.
    meta = _read_book_meta(book_path)
    chapter_ids: List[Tuple[str, str]] = []
    for c in (meta.get("characters", []) or []):
        cid = str(c.get("id"))
        name = global_canon.get(cid)
        if not name or not _is_tagged_character(name):
            continue
        chapter_ids.append((name, cid))
    chapter_ids = sorted(set(chapter_ids), key=lambda x: (x[0].lower(), x[1]))

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

        # must co-occur
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

        # unordered pairs among all characters in sentence
        for (n1, id1), (n2, id2) in combinations(uniq, 2):
            # sort pair order by name (then id) for stability and to ignore direction
            (a_name, a_id), (b_name, b_id) = sorted([(n1, id1), (n2, id2)], key=lambda x: (x[0].lower(), x[1]))
            k = _pair_key(a_name, b_name, a_id, b_id)
            out_lists.setdefault(k, []).append(sent_text)

    # Ensure EVERY possible unordered character pair is present as a key.
    # If they never co-occur, set empty string.
    out: Dict[str, str] = {}
    for (n1, id1), (n2, id2) in combinations(chapter_ids, 2):
        (a_name, a_id), (b_name, b_id) = sorted([(n1, id1), (n2, id2)], key=lambda x: (x[0].lower(), x[1]))
        k = _pair_key(a_name, b_name, a_id, b_id)
        vals = out_lists.get(k, [])
        out[k] = "\n".join(vals) if vals else ""
    return out


def extract_sentences_by_pair_by_chapter(
    *,
    book_id: str,
    booknlp_root: Path,
    chapters_root: Path,
) -> Dict[str, Dict[str, str]]:
    global_canon = build_global_coref_canon(booknlp_root)

    chapters: Dict[str, Dict[str, str]] = {}
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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build per-chapter unordered character-pair -> concatenated sentences.")
    p.add_argument("--book-id", required=True)
    p.add_argument("--booknlp-root", default=None)
    p.add_argument("--chapters-root", default=None)
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <booknlp-root>/sentences_by_pair_by_chapter.json).",
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
    chapters_root = Path(args.chapters_root) if args.chapters_root else _default_chapters_root(book_id)
    if not booknlp_root.exists():
        raise FileNotFoundError(str(booknlp_root))
    if not chapters_root.exists():
        raise FileNotFoundError(str(chapters_root))

    out = extract_sentences_by_pair_by_chapter(book_id=book_id, booknlp_root=booknlp_root, chapters_root=chapters_root)
    out_path = Path(args.output) if args.output else (booknlp_root / "sentences_by_pair_by_chapter.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d chapter(s) to %s", len(out), out_path.resolve())


if __name__ == "__main__":
    main()

