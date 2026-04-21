"""
main_baseline_injected.py — CLI orchestrator for the Markov-conditioned pipeline.

Differs from baseline/main_baseline.py in the following critical ways:

  1. SEQUENTIAL LOOP: chapters must be processed strictly in order
     k = 1, 2, ..., K.  Parallel processing is forbidden because G_k
     depends on G_{k-1}.

  2. MARKOV STATE: G_prev is threaded through the loop:
       G_0 = empty graph
       for k in 1..K:
           G_k = f(chapter_k, G_{k-1})
           G_prev = G_k

  3. NEW CLI ARGS: --alpha (decay factor), --strategy / --topN
     (graph serialization for the G_{k-1} snapshot log).

  4. EXTENDED SUMMARY TABLE: includes carried_edges and alpha columns.

Usage
-----
    python main_baseline_injected.py \\
        --book_path data/raw/pride_prejudice.txt \\
        --book_id   pride_prejudice \\
        --window    100 \\
        --min_freq  3 \\
        --alpha     0.7 \\
        --strategy  topN \\
        --topN      20

    # with evaluation
    python main_baseline_injected.py ... --evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — must be configured before project imports
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_baseline_injected")


# ---------------------------------------------------------------------------
# Path constants (resolved relative to this file's location)
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_GOLD   = ROOT / "data" / "gold_standard"
OUTPUTS_DIR = ROOT / "outputs" / "baseline_injected_graphs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return a formatted timestamp string for console output."""
    return datetime.now().strftime("%H:%M:%S")


def _print_step(step: str, msg: str) -> None:
    print(f"[{_ts()}]  [{step}]  {msg}", flush=True)


def _load_gold_graph(book_id: str, chapter_num: int):
    """Load a gold-standard graph from data/gold_standard/ if it exists.

    Expects a GraphML file named ``{book_id}_ch{N:02d}.graphml``.

    Args:
        book_id:     Short book identifier.
        chapter_num: 1-based chapter index.

    Returns:
        :class:`networkx.Graph` if the file exists, otherwise ``None``.
    """
    import networkx as nx

    gold_path = DATA_GOLD / f"{book_id}_ch{chapter_num:02d}.graphml"
    if not gold_path.exists():
        return None
    return nx.read_graphml(str(gold_path))


def _print_summary_table(rows: list[dict]) -> None:
    """Print the per-chapter summary table to stdout.

    Args:
        rows: List of dicts with keys: chapter, nodes, edges,
              carried_edges, density, avg_sentiment, alpha.
    """
    header = (
        f"{'Chapter':>8}  {'Nodes':>6}  {'Edges':>6}  "
        f"{'Carried':>8}  {'Density':>9}  {'AvgSentiment':>13}  {'Alpha':>6}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['chapter']:>8}  {r['nodes']:>6}  {r['edges']:>6}  "
            f"{r['carried_edges']:>8}  {r['density']:>9.6f}  "
            f"{r['avg_sentiment']:>13.4f}  {r['alpha']:>6.2f}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full Markov-conditioned pipeline for a single book.

    Chapters are processed strictly in ascending order so that each
    G_k is conditioned on the immediately preceding G_{k-1}.

    Args:
        args: Parsed :class:`argparse.Namespace` from the CLI.
    """
    import networkx as nx

    from src.utils.chapter_splitter import split_book_into_chapters
    from src.utils.text_cleaner import clean_chapter_text, tokenize
    from src.utils.graph_serializer import serialize
    from src.baseline_injected.run_booknlp import run_booknlp
    from src.baseline_injected.extract_edges import extract_edges
    from src.baseline_injected.build_graph import build_graph

    book_path = Path(args.book_path)
    book_id   = args.book_id

    if not book_path.exists():
        logger.error("Book file not found: %s", book_path)
        sys.exit(1)

    if not (0.0 <= args.alpha <= 1.0):
        logger.error("--alpha must be in [0.0, 1.0]; got %s", args.alpha)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 1 — Split book into chapters
    # -----------------------------------------------------------------------
    _print_step("SPLIT", f"Splitting '{book_path.name}' into chapters …")
    t0 = time.perf_counter()
    chapter_files = split_book_into_chapters(book_path, book_id, DATA_RAW)
    _print_step("SPLIT", f"Done — {len(chapter_files)} chapters  ({time.perf_counter()-t0:.1f}s)")

    if not chapter_files:
        logger.error("No chapters extracted — aborting.")
        sys.exit(1)

    if len(chapter_files) < 2:
        logger.warning(
            "Only %d chapter(s) found.  Markov conditioning has no effect "
            "until chapter 2 — proceeding anyway.", len(chapter_files)
        )

    # Sort chapter files by chapter number — MANDATORY for correct Markov ordering
    def _chapter_num(p: Path) -> int:
        try:
            return int(p.stem.split("_ch")[-1])
        except (ValueError, IndexError):
            return 0

    chapter_files = sorted(chapter_files, key=_chapter_num)

    summary_rows: list[dict] = []
    eval_results: list[dict] = []

    # G_0 is an empty graph — no prior knowledge at chapter 1
    G_prev: nx.Graph = nx.Graph()

    for chapter_path in chapter_files:
        stem = chapter_path.stem
        try:
            chapter_num = int(stem.split("_ch")[-1])
        except (ValueError, IndexError):
            logger.warning("Cannot parse chapter number from '%s'; skipping.", stem)
            continue

        _print_step(f"CH{chapter_num:02d}", f"Processing '{chapter_path.name}' …")

        # -------------------------------------------------------------------
        # Stage 2 — BookNLP (NER + coreference)
        # -------------------------------------------------------------------
        _print_step(f"CH{chapter_num:02d}", "  Running BookNLP …")
        try:
            character_positions = run_booknlp(
                chapter_path=chapter_path,
                book_id=book_id,
                chapter_num=chapter_num,
                processed_dir=DATA_PROC,
            )
        except FileNotFoundError as exc:
            logger.error("  %s — skipping chapter.", exc)
            continue
        except RuntimeError as exc:
            logger.error("  BookNLP error: %s — skipping chapter.", exc)
            continue

        if not character_positions:
            logger.warning("  No characters found in chapter %d — skipping.", chapter_num)
            continue

        # -------------------------------------------------------------------
        # Stage 3 — Tokenise + edge extraction (new co-occurrences only)
        # -------------------------------------------------------------------
        _print_step(
            f"CH{chapter_num:02d}",
            f"  Extracting edges (window={args.window}, min_freq={args.min_freq}) …",
        )
        raw_text = chapter_path.read_text(encoding="utf-8", errors="replace")
        cleaned  = clean_chapter_text(raw_text)
        tokens   = tokenize(cleaned)

        edges = extract_edges(
            tokens=tokens,
            character_positions=character_positions,
            window=args.window,
            min_freq=args.min_freq,
            G_prev=G_prev,
        )
        _print_step(f"CH{chapter_num:02d}", f"  → {len(edges)} new edges found.")

        # Serialize G_{k-1} snapshot for logging BEFORE the Markov update
        chapter_chars = set(character_positions.keys())
        serialized = serialize(
            G_prev,
            strategy=args.strategy,
            top_n=args.topN,
            chapter_chars=chapter_chars,
        )

        # -------------------------------------------------------------------
        # Stage 4 — Build and export graph (Markov update applied here)
        # -------------------------------------------------------------------
        _print_step(f"CH{chapter_num:02d}", "  Building and exporting graph …")
        G_k = build_graph(
            edges=edges,
            character_positions=character_positions,
            book_id=book_id,
            chapter_num=chapter_num,
            output_dir=OUTPUTS_DIR,
            G_prev=G_prev if G_prev.number_of_nodes() > 0 else None,
            alpha=args.alpha,
        )

        # Log G_{k-1} snapshot after build so it's associated with this chapter
        if serialized:
            logger.info(
                "Chapter %d: built G_%d conditioned on G_%d\nG_{%d} snapshot:\n%s",
                chapter_num, chapter_num, chapter_num - 1,
                chapter_num - 1, serialized,
            )
        else:
            logger.info(
                "Chapter %d: built G_%d (G_%d was empty — no prior state).",
                chapter_num, chapter_num, chapter_num - 1,
            )

        # Collect summary row
        n = G_k.number_of_nodes()
        e = G_k.number_of_edges()
        density = nx.density(G_k) if n > 1 else 0.0
        sentiments = [d["sentiment"] for _, _, d in G_k.edges(data=True) if "sentiment" in d]
        avg_sent   = sum(sentiments) / len(sentiments) if sentiments else 0.0
        carried    = sum(
            1 for _, _, d in G_k.edges(data=True) if d.get("is_carry", False)
        )
        summary_rows.append({
            "chapter":       chapter_num,
            "nodes":         n,
            "edges":         e,
            "carried_edges": carried,
            "density":       density,
            "avg_sentiment": avg_sent,
            "alpha":         args.alpha,
        })

        # -------------------------------------------------------------------
        # Stage 5 (optional) — Evaluation
        # -------------------------------------------------------------------
        if args.evaluate:
            from src.evaluation.graph_metrics import (
                edges_to_frozensets,
                precision_recall_f1,
                jaccard_index,
                graph_edit_distance,
                polarity_agreement,
            )
            gold_graph = _load_gold_graph(book_id, chapter_num)
            if gold_graph is None:
                _print_step(f"CH{chapter_num:02d}", "  No gold standard found — skipping eval.")
            else:
                pred_fe = edges_to_frozensets(G_k)
                gold_fe = edges_to_frozensets(gold_graph)
                metrics = precision_recall_f1(pred_fe, gold_fe)
                metrics["jaccard"]  = jaccard_index(pred_fe, gold_fe)
                metrics["ged"]      = graph_edit_distance(G_k, gold_graph)
                metrics["polarity_agreement"] = polarity_agreement(G_k, gold_graph)
                metrics["chapter"]  = chapter_num
                metrics["alpha"]    = args.alpha
                eval_results.append(metrics)
                _print_step(
                    f"CH{chapter_num:02d}",
                    f"  Eval → P={metrics['precision']:.3f}  "
                    f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
                    f"J={metrics['jaccard']:.3f}",
                )

        # Advance Markov state — G_k becomes G_{k-1} for the next chapter
        G_prev = G_k

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  BASELINE-INJECTED PIPELINE COMPLETE — {book_id}  (alpha={args.alpha})")
    print("=" * 70)
    _print_summary_table(summary_rows)

    if args.evaluate and eval_results:
        eval_out = OUTPUTS_DIR / f"{book_id}_eval_results.json"
        eval_out.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
        print(f"Evaluation results saved → {eval_out}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_baseline_injected",
        description=(
            "Markov-conditioned character-graph extraction pipeline. "
            "Processes a full book .txt and produces one GraphML file per chapter "
            "where G_k = f(chapter_k, G_{k-1}).  "
            "Outputs are directly comparable with baseline/ using the same GraphML format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--book_path",
        required=True,
        metavar="PATH",
        help="Path to the full book .txt file (Project Gutenberg format recommended).",
    )
    parser.add_argument(
        "--book_id",
        required=True,
        metavar="ID",
        help="Short alphanumeric identifier for the book (e.g. 'pride_prejudice').",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        metavar="W",
        help="Co-occurrence window size in tokens.",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=3,
        metavar="N",
        help="Minimum co-occurrence count to keep an edge.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        metavar="A",
        help=(
            "Markov decay factor in [0.0, 1.0].  "
            "Higher values favour the prior graph; lower values favour the current chapter.  "
            "alpha=0.0 → pure current chapter (equivalent to baseline)."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["flat", "edges", "topN"],
        default="topN",
        help="Serialization strategy for the G_{k-1} snapshot log.",
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=20,
        metavar="N",
        help="Number of top edges for the 'topN' serialization strategy.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Run evaluation against gold-standard graphs in "
            "data/gold_standard/ (if available)."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    # Ensure imports resolve from project root
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    run_pipeline(args)
