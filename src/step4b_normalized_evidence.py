"""Build normalized pair evidence from BookNLP outputs + canonical character mapping.

Evidence types:
  DIRECT_EVENT   – predicate with subject/object both canonical chars (from step3b)
  DIALOGUE_TURN  – adjacent quotes by two different mapped canonical speakers (confidence 0.85)
  QUOTE_ABOUT    – speaker mentions another canonical character inside a quote (confidence 0.75)
  CO_PRESENCE    – two canonical chars in the same sentence (confidence 0.25)
                   OR both observed anywhere in the same chapter (confidence 0.15)

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

_MAX_DIALOGUE_GAP = 80  # token gap between adjacent quotes for DIALOGUE_TURN


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
    """Return info for every canonical character observed anywhere in this chapter."""
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
    """Generate one chapter-level CO_PRESENCE item per pair not already covered."""
    canon_ids = sorted(char_info.keys())
    by_pair: Dict[str, List[Dict[str, Any]]] = {}

    for a, b in combinations(canon_ids, 2):
        pk = f"{a}||{b}"
        if pk in existing_pair_keys:
            continue
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


# ---------------------------------------------------------------------------
# Quote evidence helpers
# ---------------------------------------------------------------------------

def _read_entities_rows(run_dir: Path) -> List[List[str]]:
    ent_files = sorted(run_dir.glob("*.entities"))
    if not ent_files:
        return []
    _, rows = _read_tsv(ent_files[0])
    return rows


def _read_quotes(run_dir: Path) -> Tuple[List[str], List[List[str]]]:
    """Read .quotes TSV; return (header, rows)."""
    qt_files = sorted(run_dir.glob("*.quotes"))
    if not qt_files:
        return [], []
    return _read_tsv(qt_files[0])


def _dialogue_turn_evidence(
    qt_hdr: List[str],
    qt_rows: List[List[str]],
    coref_map: Dict[str, str],
    max_gap: int = _MAX_DIALOGUE_GAP,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate DIALOGUE_TURN evidence for adjacent mapped-speaker quote transitions."""
    if not qt_hdr:
        return {}
    try:
        i_qs  = _col(qt_hdr, "quote_start")
        i_qe  = _col(qt_hdr, "quote_end")
        i_cid = _col(qt_hdr, "char_id")
        i_qt  = _col(qt_hdr, "quote")
    except KeyError:
        return {}

    # Build list of (quote_start, quote_end, canonical_speaker, quote_text)
    mapped: List[Dict[str, Any]] = []
    min_cols = max(i_qs, i_qe, i_cid, i_qt) + 1
    for r in qt_rows:
        if len(r) < min_cols:
            continue
        canon = coref_map.get(str(r[i_cid]))
        if not canon:
            continue
        try:
            qs = int(r[i_qs])
            qe = int(r[i_qe])
        except ValueError:
            continue
        mapped.append({
            "speaker": canon,
            "quote_start": qs,
            "quote_end": qe,
            "quote": r[i_qt].strip(),
        })

    mapped.sort(key=lambda x: x["quote_start"])

    by_pair: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(len(mapped) - 1):
        a = mapped[i]
        b = mapped[i + 1]
        if a["speaker"] == b["speaker"]:
            continue
        gap = b["quote_start"] - a["quote_end"]
        if gap < 0 or gap > max_gap:
            continue
        spk_a, spk_b = sorted([a["speaker"], b["speaker"]])
        pk = f"{spk_a}||{spk_b}"
        by_pair[pk].append({
            "evidence_type":       "DIALOGUE_TURN",
            "speaker_a":           a["speaker"],
            "speaker_b":           b["speaker"],
            "quote_a":             a["quote"],
            "quote_b":             b["quote"],
            "quote_start_a":       a["quote_start"],
            "quote_end_a":         a["quote_end"],
            "quote_start_b":       b["quote_start"],
            "quote_end_b":         b["quote_end"],
            "token_gap":           gap,
            "text":                a["quote"],   # for evidence_text() in downstream scripts
            "involved_characters": [spk_a, spk_b],
            "predicate":           None,
            "confidence":          0.85,
        })

    return dict(by_pair)


def _quote_about_evidence(
    qt_hdr: List[str],
    qt_rows: List[List[str]],
    ent_rows: List[List[str]],
    coref_map: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate QUOTE_ABOUT evidence: speaker mentions another canonical char inside a quote."""
    if not qt_hdr:
        return {}
    try:
        i_qs  = _col(qt_hdr, "quote_start")
        i_qe  = _col(qt_hdr, "quote_end")
        i_cid = _col(qt_hdr, "char_id")
        i_qt  = _col(qt_hdr, "quote")
    except KeyError:
        return {}

    # Build token-indexed entity lookup: start_token -> list of (coref_id_str, text)
    ent_by_start: Dict[int, List[Tuple[str, str]]] = {}
    for r in ent_rows:
        if len(r) < 3:
            continue
        try:
            start = int(r[1])
        except ValueError:
            continue
        ent_text = r[5].strip() if len(r) > 5 else ""
        ent_by_start.setdefault(start, []).append((str(r[0]), ent_text))

    min_cols = max(i_qs, i_qe, i_cid, i_qt) + 1
    by_pair: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in qt_rows:
        if len(r) < min_cols:
            continue
        speaker_canon = coref_map.get(str(r[i_cid]))
        if not speaker_canon:
            continue
        try:
            qs = int(r[i_qs])
            qe = int(r[i_qe])
        except ValueError:
            continue
        quote_text = r[i_qt].strip()

        # Collect distinct mentioned canonical chars inside [qs, qe].
        # One QUOTE_ABOUT item per (speaker, mentioned, quote_start).
        mentioned_this_quote: Dict[str, str] = {}   # canon_id -> first mention text
        for tok in range(qs, qe + 1):
            for ent_coref, ent_text in ent_by_start.get(tok, []):
                mentioned_canon = coref_map.get(ent_coref)
                if not mentioned_canon or mentioned_canon == speaker_canon:
                    continue
                if mentioned_canon not in mentioned_this_quote:
                    mentioned_this_quote[mentioned_canon] = ent_text

        for mentioned_canon, mention_text in mentioned_this_quote.items():
            spk_a, spk_b = sorted([speaker_canon, mentioned_canon])
            pk = f"{spk_a}||{spk_b}"
            by_pair[pk].append({
                "evidence_type":       "QUOTE_ABOUT",
                "speaker":             speaker_canon,
                "mentioned_character": mentioned_canon,
                "quote":               quote_text,
                "quote_start":         qs,
                "quote_end":           qe,
                "mention_text":        mention_text,
                "text":                quote_text,   # for evidence_text() in downstream scripts
                "involved_characters": [spk_a, spk_b],
                "predicate":           None,
                "confidence":          0.75,
            })

    return dict(by_pair)


def _build_quote_diagnostics(
    qt_hdr: List[str],
    qt_rows: List[List[str]],
    coref_map: Dict[str, str],
    dialogue_ev: Dict[str, List[Dict[str, Any]]],
    quote_about_ev: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build per-chapter quote diagnostics dict."""
    if not qt_hdr or not qt_rows:
        return {
            "quotes_total": 0,
            "mapped_speaker_quotes": 0,
            "unmapped_speaker_quotes": 0,
            "mapped_speaker_ratio": 0.0,
            "distinct_mapped_speakers": 0,
            "dialogue_turn_pairs": 0,
            "dialogue_turn_evidence": 0,
            "quote_about_pairs": 0,
            "quote_about_evidence": 0,
            "top_mapped_speakers": [],
            "top_unmapped_speakers": [],
        }

    try:
        i_cid   = _col(qt_hdr, "char_id")
        i_mphr  = _col(qt_hdr, "mention_phrase")
    except KeyError:
        i_mphr = -1
        try:
            i_cid = _col(qt_hdr, "char_id")
        except KeyError:
            return {"quotes_total": len(qt_rows)}

    mapped_counts: Dict[str, int] = defaultdict(int)
    unmapped_counts: Dict[str, List[str]] = defaultdict(list)

    min_cols = i_cid + 1
    for r in qt_rows:
        if len(r) < min_cols:
            continue
        cid_str = str(r[i_cid])
        canon = coref_map.get(cid_str)
        if canon:
            mapped_counts[canon] += 1
        else:
            phrase = r[i_mphr].strip() if i_mphr >= 0 and len(r) > i_mphr else ""
            unmapped_counts[cid_str].append(phrase)

    total = len(qt_rows)
    n_mapped = sum(mapped_counts.values())
    n_unmapped = total - n_mapped

    top_mapped = sorted(
        [{"canonical_id": cid, "quotes": cnt} for cid, cnt in mapped_counts.items()],
        key=lambda x: -x["quotes"],
    )[:5]

    top_unmapped = [
        {
            "coref_id": cid,
            "mention_phrase": phrases[0] if phrases else "",
            "quotes": len(phrases),
        }
        for cid, phrases in sorted(unmapped_counts.items(), key=lambda x: -len(x[1]))
    ][:5]

    return {
        "quotes_total": total,
        "mapped_speaker_quotes": n_mapped,
        "unmapped_speaker_quotes": n_unmapped,
        "mapped_speaker_ratio": round(n_mapped / total, 4) if total else 0.0,
        "distinct_mapped_speakers": len(mapped_counts),
        "dialogue_turn_pairs": len(dialogue_ev),
        "dialogue_turn_evidence": sum(len(v) for v in dialogue_ev.values()),
        "quote_about_pairs": len(quote_about_ev),
        "quote_about_evidence": sum(len(v) for v in quote_about_ev.values()),
        "top_mapped_speakers": top_mapped,
        "top_unmapped_speakers": top_unmapped,
    }


# ---------------------------------------------------------------------------
# Main evidence builder
# ---------------------------------------------------------------------------

def build_normalized_pair_evidence(
    booknlp_root:    Path,
    full_mapping:    Dict[str, Dict[str, str]],
    chapters_root:   Optional[Path]  = None,
    include_empty:   bool            = False,
    all_canon_ids:   Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns (result, co_presence_diagnostics, quote_diagnostics).

    result: {chapter_id_str: {pair_key: [evidence_dicts]}}

    pair_key = "canon_id_a||canon_id_b"  (alphabetically sorted).
    Empty pairs are only emitted when include_empty=True (NOT_OBSERVED label).
    """
    ev_evidence = _load_event_evidence(booknlp_root)
    result:           Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    co_diag:          Dict[str, Dict[str, Any]] = {}
    quote_diag:       Dict[str, Dict[str, Any]] = {}

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

        # --- Sentence-level CO_PRESENCE ---
        sent_co_ev = _co_presence_evidence(run_dir, coref_map, chapter_txt)

        # --- Chapter-level CO_PRESENCE (fallback for pairs not sentence-covered) ---
        char_info = _chapter_canonical_chars(run_dir, coref_map)
        existing_covered = set(direct_ev) | set(sent_co_ev)
        chap_co_ev = _chapter_co_presence_evidence(char_info, existing_covered, ch_id)

        co_ev = {**sent_co_ev, **chap_co_ev}

        # --- Quote evidence: DIALOGUE_TURN + QUOTE_ABOUT ---
        qt_hdr, qt_rows = _read_quotes(run_dir)
        ent_rows = _read_entities_rows(run_dir)

        diag_ev_dt:  Dict[str, List[Dict[str, Any]]] = {}
        diag_ev_qa:  Dict[str, List[Dict[str, Any]]] = {}

        if qt_hdr:
            diag_ev_dt = _dialogue_turn_evidence(qt_hdr, qt_rows, coref_map)
            diag_ev_qa = _quote_about_evidence(qt_hdr, qt_rows, ent_rows, coref_map)

        # --- Merge all evidence ---
        all_pairs: Set[str] = set(direct_ev) | set(co_ev) | set(diag_ev_dt) | set(diag_ev_qa)
        if include_empty and all_canon_ids:
            ids = sorted(all_canon_ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    all_pairs.add(f"{ids[i]}||{ids[j]}")

        chapter_out: Dict[str, List[Dict[str, Any]]] = {}
        for pk in sorted(all_pairs):
            items: List[Dict[str, Any]] = (
                direct_ev.get(pk, [])
                + diag_ev_dt.get(pk, [])
                + diag_ev_qa.get(pk, [])
                + co_ev.get(pk, [])
            )
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

        # --- Co-presence diagnostics ---
        n_chars = len(char_info)
        candidate_pairs = n_chars * (n_chars - 1) // 2
        reason_if_zero = None
        if n_chars < 2:
            reason_if_zero = "fewer_than_two_canonical_characters"

        co_diag[ch_str] = {
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

        # --- Quote diagnostics ---
        quote_diag[ch_str] = _build_quote_diagnostics(
            qt_hdr, qt_rows, coref_map, diag_ev_dt, diag_ev_qa
        )

        logger.info(
            "Chapter %d: %d pairs  "
            "(direct=%d dialogue=%d quote_about=%d sent_co=%d chap_co=%d canonical_chars=%d)",
            ch_id,
            len(chapter_out),
            len(direct_ev),
            len(diag_ev_dt),
            len(diag_ev_qa),
            len(sent_co_ev),
            len(chap_co_ev),
            n_chars,
        )

    return result, co_diag, quote_diag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build normalized pair evidence (DIRECT_EVENT + DIALOGUE_TURN + QUOTE_ABOUT + CO_PRESENCE)."
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

    result, co_diag, quote_diag = build_normalized_pair_evidence(
        booknlp_root=root,
        full_mapping=full_mapping,
        chapters_root=chapters_root,
        include_empty=args.include_empty,
        all_canon_ids=canon_ids if args.include_empty else None,
    )

    out = Path(args.output) if args.output else root / "normalized_pair_evidence_by_chapter.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d chapters to %s", len(result), out)

    co_diag_out = root / "co_presence_diagnostics_by_chapter.json"
    co_diag_out.write_text(json.dumps(co_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote co-presence diagnostics to %s", co_diag_out)

    qt_diag_out = root / "quote_evidence_diagnostics_by_chapter.json"
    qt_diag_out.write_text(json.dumps(quote_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote quote diagnostics to %s", qt_diag_out)


if __name__ == "__main__":
    main()
