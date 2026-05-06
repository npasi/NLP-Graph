from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pair_to_chars(pair_key: str) -> tuple[str, str]:
    parts = pair_key.split("||")
    if len(parts) == 2:
        return parts[0], parts[1]
    return pair_key, ""


def character_lookup(characters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["canonical_id"]: c for c in characters if c.get("canonical_id")}


def should_keep_edge(
    score: dict[str, Any],
    *,
    min_score: float,
    require_direct: bool,
) -> bool:
    direct_event_count = int(score.get("direct_event_count", 0) or 0)
    interaction_score = float(score.get("interaction_score", 0.0) or 0.0)

    if require_direct:
        return direct_event_count > 0

    return direct_event_count > 0 or interaction_score >= min_score


def build_graph_for_chapter(
    *,
    chapter_id: str,
    scored_pairs: dict[str, Any],
    evidence_pairs: dict[str, Any],
    characters_by_id: dict[str, dict[str, Any]],
    min_score: float,
    require_direct: bool,
) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["chapter_id"] = int(chapter_id)
    graph.graph["graph_type"] = "filtered"

    for pair_key, score in scored_pairs.items():
        if not should_keep_edge(score, min_score=min_score, require_direct=require_direct):
            continue

        char_a, char_b = pair_to_chars(pair_key)
        if not char_a or not char_b or char_a == char_b:
            continue

        if char_a not in characters_by_id or char_b not in characters_by_id:
            continue

        for cid in (char_a, char_b):
            c = characters_by_id[cid]
            if cid not in graph:
                graph.add_node(
                    cid,
                    canonical_id=cid,
                    name=c.get("name", cid),
                    type=c.get("type", ""),
                    aliases=c.get("aliases", []),
                    confidence=c.get("confidence", None),
                )

        evidence = evidence_pairs.get(pair_key, [])

        graph.add_edge(
            char_a,
            char_b,
            pair_key=pair_key,
            direct_event_count=int(score.get("direct_event_count", 0) or 0),
            co_presence_count=int(score.get("co_presence_count", 0) or 0),
            total_evidence_count=int(score.get("total_evidence_count", 0) or 0),
            interaction_score=float(score.get("interaction_score", 0.0) or 0.0),
            relation_confidence=score.get("confidence", ""),
            relation_type=score.get("relation_type", ""),
            top_predicates=score.get("top_predicates", []),
            evidence=evidence,
            weight=float(score.get("interaction_score", 0.0) or 0.0),
        )

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered semantic character graphs.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--graphs-root", default="data/graphs/chapters")
    parser.add_argument("--min-score", type=float, default=2.0)
    parser.add_argument(
        "--require-direct",
        action="store_true",
        help="Keep only edges with at least one DIRECT_EVENT.",
    )
    args = parser.parse_args()

    book_id = args.book_id

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / book_id
    )
    graphs_root = Path(args.graphs_root)
    graphs_root.mkdir(parents=True, exist_ok=True)

    characters = load_json(booknlp_root / "canonical_characters.json", default=[])
    scored = load_json(booknlp_root / "scored_pair_evidence_by_chapter.json", default={})
    evidence = load_json(booknlp_root / "normalized_pair_evidence_by_chapter.json", default={})

    characters_by_id = character_lookup(characters)

    graphs: list[nx.Graph] = []
    metadata_chapters: list[dict[str, Any]] = []

    for chapter_id, scored_pairs in sorted(scored.items(), key=lambda x: int(x[0])):
        evidence_pairs = evidence.get(chapter_id, {})

        g = build_graph_for_chapter(
            chapter_id=chapter_id,
            scored_pairs=scored_pairs,
            evidence_pairs=evidence_pairs,
            characters_by_id=characters_by_id,
            min_score=args.min_score,
            require_direct=args.require_direct,
        )
        graphs.append(g)

        direct_edges = sum(
            1 for _, _, data in g.edges(data=True)
            if int(data.get("direct_event_count", 0)) > 0
        )
        co_only_edges = sum(
            1 for _, _, data in g.edges(data=True)
            if int(data.get("direct_event_count", 0)) == 0
        )

        metadata_chapters.append({
            "chapter_id": int(chapter_id),
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "direct_event_edges": direct_edges,
            "co_presence_only_edges": co_only_edges,
            "total_interaction_score": round(
                sum(float(d.get("interaction_score", 0.0)) for _, _, d in g.edges(data=True)),
                4,
            ),
        })

        print(
            f"Chapter {chapter_id}: {g.number_of_nodes()} nodes, "
            f"{g.number_of_edges()} edges "
            f"(direct={direct_edges}, co_only={co_only_edges})"
        )

    suffix = "direct_only" if args.require_direct else f"minscore_{str(args.min_score).replace('.', '_')}"
    graph_path = graphs_root / f"{book_id}_filtered_{suffix}.pkl"
    meta_path = graphs_root / f"{book_id}_filtered_{suffix}_metadata.json"

    with graph_path.open("wb") as f:
        pickle.dump(graphs, f)

    metadata = {
        "book_id": book_id,
        "source": "filtered_graphs",
        "filter": {
            "min_score": args.min_score,
            "require_direct": args.require_direct,
            "rule": "keep if direct_event_count > 0 OR interaction_score >= min_score; if require_direct, keep only direct_event_count > 0",
        },
        "chapters": metadata_chapters,
        "total_nodes_across_chapters": sum(c["nodes"] for c in metadata_chapters),
        "total_edges_across_chapters": sum(c["edges"] for c in metadata_chapters),
    }

    write_json(meta_path, metadata)

    print(f"Wrote {graph_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
