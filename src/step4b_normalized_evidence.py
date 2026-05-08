"""Build normalized pair evidence from BookNLP outputs + canonical character mapping.

Evidence types:
  DIRECT_EVENT      – predicate with subject/object both canonical chars (from step3b)
  CO_PRESENCE       – two canonical chars in the same sentence (confidence 0.25)
                    OR both observed anywhere in the same chapter (confidence 0.15)

  TODO: DIALOGUE / QUOTE_ABOUT from .quotes — needs char_id → canonical mapping;
        left as TODO to avoid blocking the pipeline on edge cases.

Default: only emit pairs that have at least one observed evidence item.
--include-empty also emits NOT_OBSERVED pairs (for debugging / negative sampling).

Usage:
    python -m src.step4b_normalized_evidence --book-id the_trial
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from src.character_identity_layer import load_canonical_characters, load_coref_mapping
from src.utils.io import data_dir

logger = logging.getLogger(__name__)


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


def _col(header: List[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError:
        raise KeyError(f"Missing column '{name}'")


def _load_event_evidence(root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    p = root / "normalized_event_evidence_by_chapter.json"
    if not p.exists():
        logger.warning("%s not found — no DIRECT_EVENT evidence (run step3b first)", p.name)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _sent_to_canonicals(
    run_dir: Path,
    coref_map: Dict[str, str],
) -> Dict[int, Set[str]]:
    """sentence_id → set of canonical_ids present (sentence-level co-presence)."""
    ent_files = sorted(run_dir.glob("*.entities"))
    tok_files = sorted(run_dir.glob("*.tokens"))
    if not ent_files or not tok_files:
        return {}

    tok_hdr, tok_rows = _read_tsv(tok_files[0])
    _, ent_rows = _read_tsv(ent_files[0])

    try:
        i_sent = _col(tok_hdr, "sentence_ID")
        i_tid  = _col(tok_hdr, "token_ID_within_document")
    except KeyError:
        return {}

    tok2sent: Dict[int, int] = {}
    for r in tok_rows:
        try:
            tok2sent[int(r[i_tid])] = int(r[i_sent])
        except (ValueError, IndexError):
            continue

    sent2canons: Dict[int, Set[str]] = {}
    for r in ent_rows:
        if len(r) < 3:
            continue
        canon = coref_map.get(str(r[0]))
        if not canon:
            continue
        try:
            s, e = int(r[1]), int(r[2])
        except ValueError:
            continue
        for t in range(s, e + 1):
            sid = tok2sent.get(t)
            if sid is not None:
                sent2canons.setdefault(sid, set()).add(canon)
    return sent2canons


def _chapter_canonical_chars(
    run_dir: Path,
    coref_map: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Return info for every canonical character observed anywhere in this chapter.

    Returns:
        {canonical_id: {"mentions": int, "samples": [str]}}
    """
    ent_files = sorted(run_dir.glob("*.entities"))
    if not ent_files:
        return {}

    _, ent_rows = _read_tsv(ent_files[0])

    info: Dict[str, Dict[str, Any]] = {}
    for r in ent_rows:
        if len(r) < 3:
            continue
        canon = coref_map.get(str(r[0]))
        if not canon:
            continue
        if canon not in info:
            info[canon] = {"mentions": 0, "samples": []}
        info[canon]["mentions"] += 1
        # Collect up to 3 text samples from the "text" column (index 5).
        if len(r) > 5 and len(info[canon]["samples"]) < 3:
            sample = r[5].strip()
            if sample and sample not in info[canon]["samples"]:
                info[canon]["samples"].append(sample)
    return info


def _sent_text_index(
    run_dir: Path,
    chapter_txt: Optional[Path],
) -> Dict[int, str]:
    if chapter_txt is None or not chapter_txt.exists():
        return {}
    tok_files = sorted(run_dir.glob("*.tokens"))
    if not tok_files:
        return {}
    tok_hdr, tok_rows = _read_tsv(tok_files[0])
    try:
        i_sent   = _col(tok_hdr, "sentence_ID")
        i_onset  = _col(tok_hdr, "byte_onset")
        i_offset = _col(tok_hdr, "byte_offset")
    except KeyError:
        return {}

    text = chapter_txt.read_text(encoding="utf-8", errors="replace")
    spans: Dict[int, Tuple[int, int]] = {}
    for r in tok_rows:
        try:
            sid = int(r[i_sent])
            on  = int(r[i_onset])
            off = int(r[i_offset])
        except (ValueError, IndexError):
            continue
        prev = spans.get(sid)
        spans[sid] = (min(prev[0], on), max(prev[1], off)) if prev else (on, off)
    return {sid: text[sp[0]: sp[1]].strip() for sid, sp in spans.items()}


def _co_presence_evidence(
    run_dir: Path,
    coref_map: Dict[str, str],
    chapter_txt: Optional[Path],
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate sentence-level CO_PRESENCE evidence (two chars in same sentence)."""
    sent2canons  = _sent_to_canonicals(run_dir, coref_map)
    sent2text    = _sent_text_index(run_dir, chapter_txt)
    by_pair: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for sid, canons in sorted(sent2canons.items()):
        canon_list = sorted(canons)
        if len(canon_list) < 2:
            continue
        snippet = sent2text.get(sid, "")
        for i in range(len(canon_list)):
            for j in range(i + 1, len(canon_list)):
                a, b = canon_list[i], canon_list[j]
                by_pair[f"{a}||{b}"].append({
                    "evidence_type":       "CO_PRESENCE",
                    "sentence_id":         sid,
                    "text":                snippet,
                    "involved_characters": [a, b],
                    "predicate":           None,
                    "confidence":          0.25,
                })
    return dict(by_pair)


def _chapter_co_presence_evidence(
    char_info: Dict[str, Dict[str, Any]],
    existing_pair_keys: Set[str],
    chapter_id: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate one chapter-level CO_PRESENCE item per pair not already in existing_pair_keys.

    This covers the case where two canonical characters appear in the same chapter
    but never in the same sentence (e.g. Heart of Darkness narrative style).
    """
    canon_ids = sorted(char_info.keys())
    by_pair: Dict[str, List[Dict[str, Any]]] = {}

    for a, b in combinations(canon_ids, 2):
        pk = f"{a}||{b}"
        if pk in existing_pair_keys:
            continue  # already covered by sentence-level or direct evidence
        by_pair[pk] = [{
            "evidence_type":       "CO_PRESENCE",
            "sentence_id":         None,
            "text":                "",
            "involved_characters": [a, b],
            "predicate":           None,
            "confidence":          0.15,
            "scope":               "chapter",
        }]

    return by_pair


def build_normalized_pair_evidence(
    booknlp_root:    Path,
    full_mapping:    Dict[str, Dict[str, str]],
    chapters_root:   Optional[Path]  = None,
    include_empty:   bool            = False,
    all_canon_ids:   Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, Any]]]:
    """
    Returns (result, diagnostics).

    result: {chapter_id_str: {pair_key: [evidence_dicts]}}
    diagnostics: {chapter_id_str: per-chapter diagnostic dict}

    pair_key = "canon_id_a||canon_id_b"  (alphabetically sorted).
    Empty pairs are only emitted when include_empty=True (NOT_OBSERVED label).
    """
    ev_evidence = _load_event_evidence(booknlp_root)
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for run_dir in sorted(
        (p for p in booknlp_root.iterdir() if p.is_dir()), key=lambda p: p.name
    ):
        ch_id = _parse_chapter_id(run_dir.name)
        if ch_id is None:
            continue
        ch_str    = str(ch_id)
        coref_map = full_mapping.get(ch_str, {})

        chapter_txt: Optional[Path] = None
        if chapters_root is not None:
            cand = chapters_root / f"chapter_{ch_id:03d}.txt"
            if cand.exists():
                chapter_txt = cand

        direct_ev = ev_evidence.get(ch_str, {})

        # Sentence-level CO_PRESENCE (existing behaviour).
        sent_co_ev = _co_presence_evidence(run_dir, coref_map, chapter_txt)

        # Chapter-level: collect all canonical chars observed in this chapter.
        char_info = _chapter_canonical_chars(run_dir, coref_map)

        # Chapter-level CO_PRESENCE for pairs not covered by sentence-level evidence.
        existing_covered = set(direct_ev) | set(sent_co_ev)
        chap_co_ev = _chapter_co_presence_evidence(char_info, existing_covered, ch_id)

        co_ev = {**sent_co_ev, **chap_co_ev}

        all_pairs: Set[str] = set(direct_ev) | set(co_ev)
        if include_empty and all_canon_ids:
            ids = sorted(all_canon_ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    all_pairs.add(f"{ids[i]}||{ids[j]}")

        chapter_out: Dict[str, List[Dict[str, Any]]] = {}
        for pk in sorted(all_pairs):
            items: List[Dict[str, Any]] = direct_ev.get(pk, []) + co_ev.get(pk, [])
            if not items and not include_empty:
                continue
            if not items and include_empty:
                parts = pk.split("||")
                items = [{
                    "evidence_type":       "NOT_OBSERVED",
                    "sentence_id":         None,
                    "text":                "",
                    "involved_characters": parts,
                    "predicate":           None,
                    "confidence":          0.0,
                }]
            chapter_out[pk] = items

        result[ch_str] = chapter_out

        # Diagnostics for this chapter.
        n_chars = len(char_info)
        candidate_pairs = n_chars * (n_chars - 1) // 2
        reason_if_zero = None
        if n_chars < 2:
            reason_if_zero = "fewer_than_two_canonical_characters"

        diagnostics[ch_str] = {
            "canonical_characters_observed": n_chars,
            "canonical_mentions": sum(v["mentions"] for v in char_info.values()),
            "candidate_pairs": candidate_pairs,
            "pairs_written": len(chapter_out),
            "direct_event_pairs": len(direct_ev),
            "sentence_co_presence_pairs": len(sent_co_ev),
            "chapter_co_presence_pairs": len(chap_co_ev),
            "reason_if_zero": reason_if_zero,
            "characters": [
                {
                    "canonical_id": cid,
                    "mentions": info["mentions"],
                    "samples": info["samples"],
                }
                for cid, info in sorted(
                    char_info.items(), key=lambda x: -x[1]["mentions"]
                )
            ],
        }

        logger.info(
            "Chapter %d: %d pairs  (direct=%d sent_co_presence=%d chap_co_presence=%d canonical_chars=%d)",
            ch_id, len(chapter_out), len(direct_ev), len(sent_co_ev), len(chap_co_ev), n_chars,
        )

    return result, diagnostics


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build normalized pair evidence (DIRECT_EVENT + CO_PRESENCE)."
    )
    p.add_argument("--book-id", required=True)
    p.add_argument("--booknlp-root", default=None)
    p.add_argument("--chapters-root", default=None)
    p.add_argument("--include-empty", action="store_true")
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

    full_mapping = load_coref_mapping(root)
    canon_ids = [c["canonical_id"] for c in load_canonical_characters(root)]

    chapters_root: Optional[Path] = None
    if args.chapters_root:
        chapters_root = Path(args.chapters_root)
    else:
        cr = data_dir() / "books" / book_id / "chapters"
        if cr.exists():
            chapters_root = cr

    result, diagnostics = build_normalized_pair_evidence(
        booknlp_root=root,
        full_mapping=full_mapping,
        chapters_root=chapters_root,
        include_empty=args.include_empty,
        all_canon_ids=canon_ids if args.include_empty else None,
    )

    out = Path(args.output) if args.output else root / "normalized_pair_evidence_by_chapter.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d chapters to %s", len(result), out)

    diag_out = root / "co_presence_diagnostics_by_chapter.json"
    diag_out.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote diagnostics to %s", diag_out)


if __name__ == "__main__":
    main()
