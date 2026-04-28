"""Extract (agent, predicate, patient) triplets from BookNLP outputs.

Reads ONLY native BookNLP artefacts:
  - <run_id>.tokens (dependency parse + lemma + sentence_ID + token IDs)
  - <run_id>.entities (PER mentions with COREF id + token span)
  - <run_id>.book (character clusters + mention lists; used for display names)

Writes derived files next to BookNLP outputs (does not modify them):
  - <run_id>.event_triplets.tsv
  - <run_id>.event_triplets.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=3, keep_default_na=False)


def _load_canonical_by_coref(book_path: Path) -> Dict[int, str]:
    """Map coref_id -> canonical display name (prefer top proper mention)."""
    data = json.loads(book_path.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for ch in data.get("characters", []) or []:
        try:
            cid = int(ch.get("id"))
        except Exception:
            continue
        mentions = ch.get("mentions") or {}
        canonical = ""
        for bucket in ("proper", "common", "pronoun"):
            arr = mentions.get(bucket) or []
            if arr:
                canonical = str(arr[0].get("n", "")).strip()
                break
        out[cid] = canonical or f"character_{cid}"
    return out


def _build_token_to_coref(entities: pd.DataFrame) -> Dict[int, int]:
    """Assign each token_ID_within_document that belongs to a PER mention to its COREF id."""
    if entities.empty:
        return {}
    required = {"COREF", "start_token", "end_token", "cat"}
    if not required.issubset(set(entities.columns)):
        return {}

    tok2c: Dict[int, int] = {}
    per = entities[entities["cat"].astype(str) == "PER"].copy()
    for _, row in per.iterrows():
        try:
            coref = int(row["COREF"])
            s = int(row["start_token"])
            e = int(row["end_token"])
        except Exception:
            continue
        for tid in range(s, e + 1):
            # If overlapping mentions disagree, keep the first assignment.
            tok2c.setdefault(tid, coref)
    return tok2c


def _children_index(tokens: pd.DataFrame) -> Dict[int, List[int]]:
    """head_token_id -> [child_token_id]."""
    out: Dict[int, List[int]] = defaultdict(list)
    if tokens.empty:
        return out
    for tid, head in zip(tokens["token_ID_within_document"].astype(int), tokens["syntactic_head_ID"].astype(int)):
        out[head].append(tid)
    return out


def _token_row(tokens: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    idx: Dict[int, Dict[str, Any]] = {}
    for _, r in tokens.iterrows():
        tid = int(r["token_ID_within_document"])
        idx[tid] = {
            "tid": tid,
            "word": str(r.get("word", "")),
            "lemma": str(r.get("lemma", "")),
            "pos": str(r.get("POS_tag", "")),
            "fine_pos": str(r.get("fine_POS_tag", "")),
            "dep": str(r.get("dependency_relation", "")),
            "head": int(r.get("syntactic_head_ID", tid)),
            "sentence_id": int(r.get("sentence_ID", -1)),
            "paragraph_id": int(r.get("paragraph_ID", -1)),
        }
    return idx


def _is_verb(row: Dict[str, Any]) -> bool:
    pos = (row.get("pos") or "").upper()
    fine = (row.get("fine_pos") or "").upper()
    return pos == "VERB" or fine.startswith("VB")


def _first_coref_in_subtree(
    *,
    root_tid: int,
    children: Dict[int, List[int]],
    tok2coref: Dict[int, int],
    max_nodes: int = 60,
) -> Optional[int]:
    """Pick a coref id from the subtree under root_tid (BFS), if any."""
    seen = set()
    q = [root_tid]
    n = 0
    while q and n < max_nodes:
        tid = q.pop(0)
        if tid in seen:
            continue
        seen.add(tid)
        n += 1
        if tid in tok2coref:
            return tok2coref[tid]
        for ch in children.get(tid, []):
            q.append(ch)
    return None


def extract_event_triplets(
    *,
    run_dir: Path,
    run_id: str,
    only_person_person: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    run_dir = Path(run_dir)
    tokens_path = run_dir / f"{run_id}.tokens"
    entities_path = run_dir / f"{run_id}.entities"
    book_path = run_dir / f"{run_id}.book"

    tokens = _read_tsv(tokens_path)
    entities = _read_tsv(entities_path)
    canon = _load_canonical_by_coref(book_path) if book_path.exists() else {}

    # Build indices.
    tok_rows = _token_row(tokens)
    children = _children_index(tokens)
    tok2coref = _build_token_to_coref(entities)

    trip_rows: List[Dict[str, Any]] = []
    jsonl: List[Dict[str, Any]] = []

    # Iterate tokens; treat each VERB as an event candidate.
    for tid, row in tok_rows.items():
        if not _is_verb(row):
            continue
        dep = row.get("dep", "")
        # Skip auxiliaries / non-event tokens if they show up as VERB sometimes.
        if dep in {"aux", "auxpass"}:
            continue

        sent = int(row.get("sentence_id", -1))
        lemma = row.get("lemma") or row.get("word") or ""

        # Identify subject/object via dependency children.
        agent_coref: Optional[int] = None
        patient_coref: Optional[int] = None
        voice = "active"

        for ch_tid in children.get(tid, []):
            ch = tok_rows.get(ch_tid)
            if not ch:
                continue
            d = ch.get("dep", "")
            if d == "nsubj":
                agent_coref = _first_coref_in_subtree(root_tid=ch_tid, children=children, tok2coref=tok2coref)
            elif d == "nsubjpass":
                voice = "passive"
                patient_coref = _first_coref_in_subtree(root_tid=ch_tid, children=children, tok2coref=tok2coref)
            elif d in {"dobj", "obj"}:
                patient_coref = _first_coref_in_subtree(root_tid=ch_tid, children=children, tok2coref=tok2coref)
            elif d == "iobj":
                # keep as fallback patient if no direct object
                if patient_coref is None:
                    patient_coref = _first_coref_in_subtree(root_tid=ch_tid, children=children, tok2coref=tok2coref)
            elif d == "agent":
                # passive "by X": agent -> pobj
                for gc_tid in children.get(ch_tid, []):
                    gc = tok_rows.get(gc_tid)
                    if not gc:
                        continue
                    if gc.get("dep") == "pobj":
                        agent_coref = _first_coref_in_subtree(root_tid=gc_tid, children=children, tok2coref=tok2coref)

        if agent_coref is None or patient_coref is None:
            continue
        if only_person_person:
            if agent_coref not in canon or patient_coref not in canon:
                # If .book missing, allow numeric anyway.
                if canon:
                    continue

        rec = {
            "sentence_id": sent,
            "verb_token_id": tid,
            "predicate_lemma": lemma,
            "predicate_word": row.get("word", ""),
            "voice": voice,
            "agent_coref": int(agent_coref),
            "patient_coref": int(patient_coref),
            "agent_name": canon.get(int(agent_coref), f"character_{agent_coref}"),
            "patient_name": canon.get(int(patient_coref), f"character_{patient_coref}"),
        }
        trip_rows.append(rec)
        jsonl.append(rec)

    df = pd.DataFrame(trip_rows)
    return df, jsonl


def write_outputs(*, run_dir: Path, run_id: str, df: pd.DataFrame, jsonl: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    run_dir = Path(run_dir)
    tsv_path = run_dir / f"{run_id}.event_triplets.tsv"
    jsonl_path = run_dir / f"{run_id}.event_triplets.jsonl"

    df.to_csv(tsv_path, sep="\t", index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for obj in jsonl:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return tsv_path, jsonl_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract agent-predicate-patient triplets from BookNLP outputs.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--include-nonperson", action="store_true", help="Include events where agent/patient are not PER clusters.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    df, jsonl = extract_event_triplets(
        run_dir=Path(args.run_dir),
        run_id=args.run_id,
        only_person_person=not bool(args.include_nonperson),
    )
    tsv_path, jsonl_path = write_outputs(run_dir=Path(args.run_dir), run_id=args.run_id, df=df, jsonl=jsonl)
    logger.info("Triplets: %d rows", int(df.shape[0]))
    logger.info("Wrote %s", tsv_path)
    logger.info("Wrote %s", jsonl_path)


if __name__ == "__main__":
    main()

