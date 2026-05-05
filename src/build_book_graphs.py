"""Build per-chapter character graphs for an already-split book.

This module is the missing "book-level" glue:

- Input: `data/books/<book_id>/chapters/chapters.json` produced by
  `python -m src.step1_split_only ...`
- For each chapter, run/reuse BookNLP caches and build a NetworkX graph
  via `src.graph_builder.build_chapter_graph`.
- Output:
  - `data/graphs/chapters/<book_id>.pkl` (list[nx.Graph])
  - `data/graphs/chapters/<book_id>_metadata.json` (per-chapter stats)

If BookNLP outputs already exist under `data/booknlp_output/<run_id>/`,
they are reused automatically (no rerun).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.graph_builder import FILTER_MODES, build_chapter_graph
from src.utils.io import data_dir, ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)


def _load_chapters(chapters_json: Path) -> List[dict]:
    """Return chapter records with `chapter_id,title,text`."""
    items = read_json(chapters_json)
    if not isinstance(items, list):
        raise ValueError(f"Expected list in {chapters_json}, got {type(items).__name__}")
    out: List[dict] = []
    for rec in items:
        if not isinstance(rec, dict):
            continue
        ch_id = int(rec.get("chapter_id", 0))
        title = str(rec.get("title", "") or "")
        # step1_split_only writes {path: ".../chapter_000.txt"}.
        p = rec.get("path")
        if p:
            text = Path(str(p)).read_text(encoding="utf-8", errors="replace")
        else:
            # Fallback to embedded text, if user provided a different schema.
            text = str(rec.get("text", "") or "")
        out.append({"chapter_id": ch_id, "title": title, "text": text, "short": bool(rec.get("short", False))})
    out.sort(key=lambda r: int(r["chapter_id"]))
    return out


def _chapter_stats(g: nx.Graph) -> Dict[str, Any]:
    nodes = list(g.nodes(data=True))
    mention_counts = [int(d.get("mention_count", 0) or 0) for _, d in nodes]
    top = sorted(
        [
            {
                "coref_id": int(n),
                "name": str(d.get("name", "")),
                "mention_count": int(d.get("mention_count", 0) or 0),
            }
            for n, d in nodes
        ],
        key=lambda r: -int(r["mention_count"]),
    )[:10]
    return {
        "chapter_id": g.graph.get("chapter_id"),
        "chapter_title": g.graph.get("chapter_title", ""),
        "num_tokens": int(g.graph.get("num_tokens", 0) or 0),
        "short_chapter": bool(g.graph.get("short_chapter", False)),
        "num_characters": int(g.number_of_nodes()),
        "num_edges": int(g.number_of_edges()),
        "mentions_total": int(sum(mention_counts)),
        "mentions_max": int(max(mention_counts) if mention_counts else 0),
        "top_characters": top,
    }


def build_book_graphs(
    *,
    book_id: str,
    chapters_json: Optional[Path] = None,
    output_root: Optional[Path] = None,
    tmp_root: Optional[Path] = None,
    model_size: str = "big",
    filter_mode: str = "curated",
    graphs_out: Optional[Path] = None,
    metadata_out: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[List[nx.Graph], List[Dict[str, Any]]]:
    """Build a graph per chapter and return `(graphs, metadata)`."""
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"filter_mode must be one of {FILTER_MODES}")

    book_id = str(book_id)
    chapters_json = chapters_json or (data_dir() / "books" / book_id / "chapters" / "chapters.json")
    chapters = _load_chapters(Path(chapters_json))
    if limit is not None:
        chapters = chapters[: int(limit)]

    graphs_dir = ensure_dir(data_dir() / "graphs" / "chapters")
    graphs_out = graphs_out or (graphs_dir / f"{book_id}.pkl")
    metadata_out = metadata_out or (graphs_dir / f"{book_id}_metadata.json")

    chapter_graphs: List[nx.Graph] = []
    meta: List[Dict[str, Any]] = []

    for ch in chapters:
        g = build_chapter_graph(
            ch,
            book_id=book_id,
            output_root=output_root,
            tmp_root=tmp_root,
            model_size=model_size,
            filter_mode=filter_mode,
        )
        chapter_graphs.append(g)
        meta.append(_chapter_stats(g))

    ensure_dir(Path(graphs_out).parent)
    with Path(graphs_out).open("wb") as f:
        pickle.dump(chapter_graphs, f)

    write_json(Path(metadata_out), meta, indent=2)

    logger.info("Built %d chapter graphs for book_id=%s", len(chapter_graphs), book_id)
    logger.info("Wrote graphs: %s", graphs_out)
    logger.info("Wrote metadata: %s", metadata_out)
    return chapter_graphs, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build one NetworkX character graph per chapter for a split book.")
    p.add_argument("--book-id", required=True, help="Folder under data/books/<book-id>/chapters/chapters.json")
    p.add_argument("--chapters-json", default=None, help="Override chapters.json path")
    p.add_argument("--model-size", choices=["small", "big"], default="big")
    p.add_argument("--filter-mode", choices=list(FILTER_MODES), default="curated")
    p.add_argument("--limit", type=int, default=None, help="Only build first N chapters")
    p.add_argument("--output-root", default=None, help="BookNLP cache root (default: data/booknlp_output)")
    p.add_argument("--tmp-root", default=None, help="BookNLP temp input root (default: data/tmp)")
    p.add_argument("--graphs-out", default=None, help="Output .pkl path for list[nx.Graph]")
    p.add_argument("--metadata-out", default=None, help="Output JSON path for per-chapter stats")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    build_book_graphs(
        book_id=args.book_id,
        chapters_json=Path(args.chapters_json) if args.chapters_json else None,
        output_root=Path(args.output_root) if args.output_root else None,
        tmp_root=Path(args.tmp_root) if args.tmp_root else None,
        model_size=args.model_size,
        filter_mode=args.filter_mode,
        graphs_out=Path(args.graphs_out) if args.graphs_out else None,
        metadata_out=Path(args.metadata_out) if args.metadata_out else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_book_graphs"]

