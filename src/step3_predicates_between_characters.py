from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

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


def _read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    return header, rows


def _read_book_meta(book_path: Path) -> Dict[str, Any]:
    return json.loads(book_path.read_text(encoding="utf-8", errors="replace"))


def _canonical_by_coref_id(book_meta: Dict[str, Any]) -> Dict[str, str]:
    """Map coref_id -> canonical name, using BookNLP's mentions lists."""
    out: Dict[str, str] = {}
    for c in (book_meta.get("characters", []) or []):
        coref_id = str(c.get("id"))
        mentions = c.get("mentions", {}) or {}
        best_name = ""
        best_c = -1
        for key in ("proper", "common"):
            for m in (mentions.get(key, []) or []):
                name = str(m.get("n", "") or "").strip()
                cnt = int(m.get("c", 0) or 0)
                if name and cnt > best_c:
                    best_name, best_c = name, cnt
        out[coref_id] = best_name or f"coref_{coref_id}"
    return out


def _is_tagged_character(name: str) -> bool:
    # "coref_123" indicates no proper/common mentions, so drop it.
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
    """Best-effort coref_id -> canonical name across all chapters.

    This lets us keep clusters that are pronoun-only in one chapter but named in another.
    """
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


def _build_token_to_coref_map(entities_rows: List[List[str]]) -> Dict[int, str]:
    """Map token_id_within_document -> COREF id using BookNLP .entities spans."""
    # .entities columns: COREF start_token end_token prop cat text
    tok2coref: Dict[int, str] = {}
    for r in entities_rows:
        if len(r) < 3:
            continue
        coref = str(r[0])
        try:
            start = int(r[1])
            end = int(r[2])
        except ValueError:
            continue
        for t in range(start, end + 1):
            tok2coref[t] = coref
    return tok2coref


def _idx(header: List[str], col: str) -> int:
    try:
        return header.index(col)
    except ValueError:
        raise KeyError(f"Missing column '{col}'")


_SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS_DIRECT = {"dobj", "obj", "iobj"}
_OBJ_BRIDGE_DEPS = {"prep", "agent", "obl", "dative"}
_OBJ_LEAF_DEPS = {"pobj", "obj"}


def _extract_pairs_from_run_dir(run_dir: Path, *, global_canon: Dict[str, str]) -> Dict[str, Any]:
    tokens_path = sorted(run_dir.glob("*.tokens"))[0]
    entities_path = sorted(run_dir.glob("*.entities"))[0]
    book_path = sorted(run_dir.glob("*.book"))[0]

    book_meta = _read_book_meta(book_path)
    coref2name_local = _canonical_by_coref_id(book_meta)
    # Prefer global naming if available; fallback to local.
    coref2name: Dict[str, str] = {}
    for cid, local_name in coref2name_local.items():
        coref2name[cid] = global_canon.get(cid, local_name)
    tagged_ids = {cid for cid, name in coref2name.items() if _is_tagged_character(name)}

    _, ent_rows = _read_tsv(entities_path)
    tok2coref = _build_token_to_coref_map(ent_rows)

    tok_header, tok_rows = _read_tsv(tokens_path)
    sent_i = _idx(tok_header, "sentence_ID")
    tid_doc_i = _idx(tok_header, "token_ID_within_document")
    lemma_i = _idx(tok_header, "lemma")
    dep_i = _idx(tok_header, "dependency_relation")
    head_i = _idx(tok_header, "syntactic_head_ID")
    event_i = _idx(tok_header, "event")

    by_sent: DefaultDict[int, List[List[str]]] = defaultdict(list)
    for r in tok_rows:
        try:
            by_sent[int(r[sent_i])].append(r)
        except Exception:
            continue

    pair2preds: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

    for _sid, rows in by_sent.items():
        dependents: DefaultDict[int, List[List[str]]] = defaultdict(list)
        rows_by_doc: Dict[int, List[str]] = {}
        for r in rows:
            try:
                t_doc = int(r[tid_doc_i])
                h_doc = int(r[head_i])
            except Exception:
                continue
            rows_by_doc[t_doc] = r
            if h_doc != t_doc:
                dependents[h_doc].append(r)

        for t_doc, r in rows_by_doc.items():
            if r[event_i] != "EVENT":
                continue
            pred = str(r[lemma_i]).strip()
            if not pred:
                continue

            subj: List[str] = []
            obj: List[str] = []

            # direct children
            for d in dependents.get(t_doc, []):
                dep = d[dep_i]
                try:
                    d_doc = int(d[tid_doc_i])
                except Exception:
                    continue
                coref = tok2coref.get(d_doc)
                if dep in _SUBJ_DEPS and coref:
                    subj.append(coref)
                if dep in _OBJ_DEPS_DIRECT and coref:
                    obj.append(coref)

                # bridge via prepositions/obliques: verb -> prep/obl -> pobj
                if dep in _OBJ_BRIDGE_DEPS:
                    for g in dependents.get(d_doc, []):
                        if g[dep_i] not in _OBJ_LEAF_DEPS:
                            continue
                        try:
                            g_doc = int(g[tid_doc_i])
                        except Exception:
                            continue
                        g_coref = tok2coref.get(g_doc)
                        if g_coref:
                            obj.append(g_coref)

            if not subj or not obj:
                continue
            for s_c in sorted(set(subj)):
                for o_c in sorted(set(obj)):
                    if s_c == o_c:
                        continue
                    if (str(s_c) not in tagged_ids) or (str(o_c) not in tagged_ids):
                        continue
                    # unordered pair key (ignore direction)
                    s_name = coref2name[str(s_c)]
                    o_name = coref2name[str(o_c)]
                    a, b = sorted([(str(s_c), s_name), (str(o_c), o_name)], key=lambda x: x[1].lower())
                    pair2preds[(a[0], b[0])].append(pred)

    # Convert to output dict keyed by "A||B"
    out: Dict[str, Any] = {}
    for (a_id, b_id), preds in pair2preds.items():
        a_name = coref2name.get(str(a_id), f"coref_{a_id}")
        b_name = coref2name.get(str(b_id), f"coref_{b_id}")
        key = f"{a_name}||{b_name}"
        out[key] = {
            "char_a": a_name,
            "char_b": b_name,
            "char_a_id": int(a_id) if str(a_id).isdigit() else a_id,
            "char_b_id": int(b_id) if str(b_id).isdigit() else b_id,
            "predicates": preds,  # concatenation (keeps repetition)
        }
    return out


def extract_predicates_by_chapter(*, booknlp_root: Path) -> Dict[str, Dict[str, Any]]:
    chapters: Dict[str, Dict[str, Any]] = {}
    global_canon = build_global_coref_canon(booknlp_root)
    run_dirs = sorted([p for p in booknlp_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for run_dir in run_dirs:
        chapter_id = _parse_chapter_id(run_dir.name)
        if chapter_id is None:
            continue
        try:
            chapters[str(chapter_id)] = _extract_pairs_from_run_dir(run_dir, global_canon=global_canon)
        except Exception as e:
            logger.exception("Failed extracting predicates for %s: %s", run_dir, e)
            raise
    return chapters


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract per-chapter character-pair predicates from BookNLP outputs.")
    p.add_argument("--book-id", required=True, help="Book id (e.g. 46).")
    p.add_argument("--booknlp-root", default=None, help="Root with per-chapter BookNLP folders.")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <booknlp-root>/predicates_by_chapter.json).",
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

    chapters = extract_predicates_by_chapter(booknlp_root=booknlp_root)
    out_path = Path(args.output) if args.output else (booknlp_root / "predicates_by_chapter.json")
    out_path.write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d chapter(s) to %s", len(chapters), out_path.resolve())


if __name__ == "__main__":
    main()

