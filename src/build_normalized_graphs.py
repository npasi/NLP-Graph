"""Build NetworkX character graphs from the normalized identity + evidence layer.

Input (all under <booknlp_root>/):
  canonical_characters.json
  normalized_pair_evidence_by_chapter.json

Output:
  data/graphs/chapters/<book_id>_normalized.pkl         list[nx.Graph]
  data/graphs/chapters/<book_id>_normalized_metadata.json

Edge weight formula:
  weight = direct_event_count * 2.0 + co_presence_count * 0.25

Usage:
    python -m src.build_normalized_graphs --book-id the_trial
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.character_identity_layer import load_canonical_characters
from src.utils.io import data_dir

logger = logging.getLogger(__name__)


def _default_booknlp_root(book_id: str) -> Path:
    return data_dir() / "booknlp_chapter_output" / book_id


def _default_graphs_dir() -> Path:
    return data_dir() / "graphs" / "chapters"


def _load_pair_evidence(root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    p = root / "normalized_pair_evidence_by_chapter.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} not found in {root}. Run step4b_normalized_evidence first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _edge_attrs(
    items: List[Dict[str, Any]],
    char_conf: Dict[str, float],
) -> Dict[str, Any]:
    direct  = sum(1 for e in items if e.get("evidence_type") == "DIRECT_EVENT")
    co_pres = sum(1 for e in items if e.get("evidence_type") == "CO_PRESENCE")
    preds   = sorted({e["predicate"] for e in items
                      if e.get("evidence_type") == "DIRECT_EVENT" and e.get("predicate")})

    weight = direct * 2.0 + co_pres * 0.25

    if direct > 0:
        rel_conf = min(1.0, 0.5 + 0.05 * direct)
    elif co_pres > 0:
        rel_conf = min(0.4, 0.1 + 0.01 * co_pres)
    else:
        rel_conf = 0.0

    involved = {ch for e in items for ch in e.get("involved_characters", [])}
    id_conf  = sum(char_conf.get(ch, 0.5) for ch in involved) / max(len(involved), 1)

    return {
        "direct_event_count":  direct,
        "co_presence_count":   co_pres,
        "predicate_count":     len(preds),
        "predicates":          preds,
        "evidence":            items,
        "weight":              round(weight, 4),
        "relation_confidence": round(rel_conf, 4),
        "identity_confidence": round(id_conf, 4),
    }


def build_normalized_graphs(
    booknlp_root: Path,
    book_id:      str,
) -> Tuple[List[nx.Graph], Dict[str, Any]]:
    canon_chars  = load_canonical_characters(booknlp_root)
    pair_ev      = _load_pair_evidence(booknlp_root)

    char_info    = {c["canonical_id"]: c              for c in canon_chars}
    char_conf    = {c["canonical_id"]: float(c.get("confidence", 0.5)) for c in canon_chars}

    chapter_ids  = sorted(pair_ev, key=int)
    max_ch       = int(chapter_ids[-1]) if chapter_ids else -1

    graphs: List[nx.Graph] = []
    stats:  List[Dict[str, Any]] = []

    def _add_node(G: nx.Graph, cid: str) -> None:
        if G.has_node(cid):
            return
        info = char_info.get(cid, {})
        G.add_node(cid,
                   canonical_id=cid,
                   name=info.get("name", cid),
                   type=info.get("type", "unknown"),
                   aliases=sorted(info.get("aliases", [])),
                   confidence=float(info.get("confidence", 0.5)))

    for ch_idx in range(max_ch + 1):
        ch_str = str(ch_idx)
        G = nx.Graph()
        G.graph.update({"chapter_id": ch_idx, "book_id": book_id,
                         "source": "normalized_identity_layer"})

        for pk, items in sorted((pair_ev.get(ch_str) or {}).items()):
            # Skip NOT_OBSERVED-only pairs
            real = [e for e in items if e.get("evidence_type") != "NOT_OBSERVED"]
            if not real:
                continue
            parts = pk.split("||")
            if len(parts) != 2:
                continue
            ca, cb = parts
            if ca not in char_info or cb not in char_info:
                continue
            attrs = _edge_attrs(real, char_conf)
            if attrs["weight"] == 0.0:
                continue
            _add_node(G, ca)
            _add_node(G, cb)
            attrs["char_a"] = ca
            attrs["char_b"] = cb
            G.add_edge(ca, cb, **attrs)

        graphs.append(G)
        stats.append({"chapter_id": ch_idx,
                       "nodes": G.number_of_nodes(),
                       "edges": G.number_of_edges()})
        logger.info("Chapter %d: %d nodes, %d edges",
                    ch_idx, G.number_of_nodes(), G.number_of_edges())

    metadata = {
        "book_id":                   book_id,
        "chapters":                  stats,
        "total_canonical_characters": len(canon_chars),
        "source":                    "normalized_identity_layer",
    }
    return graphs, metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build normalized character graphs from identity layer outputs."
    )
    p.add_argument("--book-id",     required=True)
    p.add_argument("--booknlp-root", default=None)
    p.add_argument("--graphs-dir",  default=None)
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    book_id  = str(args.book_id)
    root     = Path(args.booknlp_root) if args.booknlp_root else _default_booknlp_root(book_id)
    if not root.exists():
        raise FileNotFoundError(str(root))

    graphs_dir = Path(args.graphs_dir) if args.graphs_dir else _default_graphs_dir()
    graphs_dir.mkdir(parents=True, exist_ok=True)

    graphs, meta = build_normalized_graphs(root, book_id)

    pkl  = graphs_dir / f"{book_id}_normalized.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(graphs, fh)
    logger.info("Wrote %d graphs to %s", len(graphs), pkl)

    meta_path = graphs_dir / f"{book_id}_normalized_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote metadata to %s", meta_path)


if __name__ == "__main__":
    main()
