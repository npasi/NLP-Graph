"""Extract predicates between canonical characters (normalized).

Reads BookNLP .tokens + .entities plus the Character Identity Layer outputs
and produces DIRECT_EVENT evidence keyed by canonical character pairs.

Key rule: map (chapter_id, local_coref_id) -> canonical_id BEFORE forming pairs.
If either endpoint has no canonical mapping, skip.
If both endpoints resolve to the same canonical id, skip.

Usage:
    python -m src.step3b_normalized_predicates --book-id the_trial
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from src.character_identity_layer import load_coref_mapping
from src.utils.io import data_dir

logger = logging.getLogger(__name__)

_SUBJ_DEPS      = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DIRECT     = {"dobj", "obj", "iobj"}
_OBJ_BRIDGE     = {"prep", "agent", "obl", "dative"}
_OBJ_LEAF       = {"pobj", "obj"}


def _default_booknlp_root(book_id: str) -> Path:
    return data_dir() / "booknlp_chapter_output" / book_id


def _parse_chapter_id(name: str) -> Optional[int]:
    parts = name.split("_")
    return int(parts[-1]) if parts and parts[-1].isdigit() else None


def _read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return [], []
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:] if ln.strip()]


def _col(header: List[str], col: str) -> int:
    try:
        return header.index(col)
    except ValueError:
        raise KeyError(f"Missing column '{col}'")


def _tok2coref(ent_rows: List[List[str]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for r in ent_rows:
        if len(r) < 3:
            continue
        try:
            s, e = int(r[1]), int(r[2])
        except ValueError:
            continue
        for t in range(s, e + 1):
            out[t] = str(r[0])
    return out


def _extract_chapter_events(
    run_dir: Path,
    coref_map: Dict[str, str],  # local coref_id -> canonical_id
) -> List[Dict[str, Any]]:
    tok_files = sorted(run_dir.glob("*.tokens"))
    ent_files = sorted(run_dir.glob("*.entities"))
    if not tok_files or not ent_files:
        logger.warning("Missing .tokens/.entities in %s — skipping", run_dir)
        return []

    tok_hdr, tok_rows = _read_tsv(tok_files[0])
    _, ent_rows = _read_tsv(ent_files[0])

    t2coref = _tok2coref(ent_rows)

    try:
        i_sent  = _col(tok_hdr, "sentence_ID")
        i_tid   = _col(tok_hdr, "token_ID_within_document")
        i_lemma = _col(tok_hdr, "lemma")
        i_dep   = _col(tok_hdr, "dependency_relation")
        i_head  = _col(tok_hdr, "syntactic_head_ID")
    except KeyError as e:
        logger.warning("Column error in %s: %s — skipping", run_dir, e)
        return []

    try:
        i_event: Optional[int] = _col(tok_hdr, "event")
    except KeyError:
        i_event = None

    # Group tokens by sentence
    by_sent: DefaultDict[int, List[List[str]]] = defaultdict(list)
    for r in tok_rows:
        try:
            by_sent[int(r[i_sent])].append(r)
        except (ValueError, IndexError):
            continue

    events: List[Dict[str, Any]] = []

    for sid, rows in by_sent.items():
        dep_children: DefaultDict[int, List[List[str]]] = defaultdict(list)
        by_doc: Dict[int, List[str]] = {}
        for r in rows:
            try:
                td, hd = int(r[i_tid]), int(r[i_head])
            except (ValueError, IndexError):
                continue
            by_doc[td] = r
            if hd != td:
                dep_children[hd].append(r)

        for td, r in by_doc.items():
            is_ev = True
            if i_event is not None:
                try:
                    is_ev = str(r[i_event]).strip().upper() == "EVENT"
                except IndexError:
                    is_ev = False
            if not is_ev:
                continue
            try:
                pred = str(r[i_lemma]).strip()
            except IndexError:
                continue
            if not pred:
                continue

            subj_c: List[str] = []
            obj_c:  List[str] = []

            for d in dep_children.get(td, []):
                try:
                    dep = d[i_dep]
                    dd  = int(d[i_tid])
                except (IndexError, ValueError):
                    continue
                c = t2coref.get(dd)
                if dep in _SUBJ_DEPS and c:
                    subj_c.append(c)
                if dep in _OBJ_DIRECT and c:
                    obj_c.append(c)
                if dep in _OBJ_BRIDGE:
                    for g in dep_children.get(dd, []):
                        try:
                            if g[i_dep] not in _OBJ_LEAF:
                                continue
                            gc = t2coref.get(int(g[i_tid]))
                            if gc:
                                obj_c.append(gc)
                        except (IndexError, ValueError):
                            continue

            for sc in sorted(set(subj_c)):
                for oc in sorted(set(obj_c)):
                    if sc == oc:
                        continue
                    s_can = coref_map.get(sc)
                    o_can = coref_map.get(oc)
                    if not s_can or not o_can or s_can == o_can:
                        continue
                    events.append({
                        "evidence_type": "DIRECT_EVENT",
                        "predicate":     pred,
                        "subject":       s_can,
                        "object":        o_can,
                        "sentence_id":   sid,
                        "confidence":    1.0,
                    })
    return events


def extract_normalized_events(
    booknlp_root: Path,
    full_mapping: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for run_dir in sorted(
        (p for p in booknlp_root.iterdir() if p.is_dir()), key=lambda p: p.name
    ):
        ch_id = _parse_chapter_id(run_dir.name)
        if ch_id is None:
            continue
        coref_map = full_mapping.get(str(ch_id), {})
        events = _extract_chapter_events(run_dir, coref_map)

        by_pair: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ev in events:
            a, b = sorted([ev["subject"], ev["object"]])
            by_pair[f"{a}||{b}"].append(ev)

        result[str(ch_id)] = dict(sorted(by_pair.items()))
        logger.info("Chapter %d: %d events → %d pairs", ch_id, len(events), len(by_pair))
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract normalized predicates between canonical characters."
    )
    p.add_argument("--book-id", required=True)
    p.add_argument("--booknlp-root", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    book_id = str(args.book_id)
    root = Path(args.booknlp_root) if args.booknlp_root else _default_booknlp_root(book_id)
    if not root.exists():
        raise FileNotFoundError(str(root))

    result = extract_normalized_events(root, load_coref_mapping(root))
    out = Path(args.output) if args.output else root / "normalized_event_evidence_by_chapter.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d chapters to %s", len(result), out)


if __name__ == "__main__":
    main()
