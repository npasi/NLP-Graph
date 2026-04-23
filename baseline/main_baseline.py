"""
main_baseline.py — CLI orchestrator for the baseline graph-extraction pipeline.

Processes a full book through five sequential stages:
  1. Chapter splitting        (chapter_splitter)
  2. BookNLP NER + coref      (run_booknlp)
  3. Co-occurrence + sentiment (extract_edges)
  4. Graph assembly + export  (build_graph)
  5. Optional evaluation      (graph_metrics)

Usage
-----
    python main_baseline.py \\
        --book_path data/raw/pride_prejudice.txt \\
        --book_id   pride_prejudice \\
        --window    100 \\
        --min_freq  3

    # with evaluation against gold standard
    python main_baseline.py ... --evaluate
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
logger = logging.getLogger("main_baseline")


# ---------------------------------------------------------------------------
# Path constants (resolved relative to this file's location)
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROC     = ROOT / "data" / "processed"
DATA_GOLD     = ROOT / "data" / "gold_standard"
OUTPUTS_DIR   = ROOT / "outputs" / "baseline_graphs"


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
    """Print a formatted summary table to stdout.

    Args:
        rows: List of dicts with keys: chapter, nodes, edges,
              density, avg_sentiment.
    """
    header = f"{'Chapter':>8}  {'Nodes':>6}  {'Edges':>6}  {'Density':>9}  {'AvgSentiment':>13}"
    sep    = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['chapter']:>8}  {r['nodes']:>6}  {r['edges']:>6}"
            f"  {r['density']:>9.6f}  {r['avg_sentiment']:>13.4f}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full baseline pipeline for a single book.

    Args:
        args: Parsed :class:`argparse.Namespace` from the CLI.
    """
    # --- Import project modules here (after sys.path is set) ---
    from src.utils.chapter_splitter import split_book_into_chapters
    from src.utils.text_cleaner import clean_chapter_text, tokenize
    from src.baseline.run_booknlp import run_booknlp
    from src.baseline.extract_edges import extract_edges
    from src.baseline.build_graph import build_graph

    book_path = Path(args.book_path)
    book_id   = args.book_id

    if not book_path.exists():
        logger.error("Book file not found: %s", book_path)
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

    summary_rows: list[dict] = []
    eval_results: list[dict] = []

    for chapter_path in chapter_files:
        # Derive chapter number from filename (e.g. pride_prejudice_ch03.txt → 3)
        stem = chapter_path.stem          # e.g. "pride_prejudice_ch03"
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
        # Stage 3 — Tokenise + edge extraction
        # -------------------------------------------------------------------
        _print_step(f"CH{chapter_num:02d}", f"  Extracting edges (window={args.window}, min_freq={args.min_freq}) …")
        raw_text = chapter_path.read_text(encoding="utf-8", errors="replace")
        cleaned  = clean_chapter_text(raw_text)
        tokens   = tokenize(cleaned)

        edges = extract_edges(
            tokens=tokens,
            character_positions=character_positions,
            window=args.window,
            min_freq=args.min_freq,
        )
        _print_step(f"CH{chapter_num:02d}", f"  → {len(edges)} edges found.")

        # -------------------------------------------------------------------
        # Stage 4 — Build and export graph
        # -------------------------------------------------------------------
        _print_step(f"CH{chapter_num:02d}", "  Building and exporting graph …")
        G = build_graph(
            edges=edges,
            character_positions=character_positions,
            book_id=book_id,
            chapter_num=chapter_num,
            output_dir=OUTPUTS_DIR,
        )

        # Collect summary row
        import networkx as nx
        n = G.number_of_nodes()
        e = G.number_of_edges()
        density = nx.density(G) if n > 1 else 0.0
        sentiments = [d["sentiment"] for _, _, d in G.edges(data=True) if "sentiment" in d]
        avg_sent   = sum(sentiments) / len(sentiments) if sentiments else 0.0
        summary_rows.append({
            "chapter": chapter_num,
            "nodes":   n,
            "edges":   e,
            "density": density,
            "avg_sentiment": avg_sent,
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
                pred_fe = edges_to_frozensets(G)
                gold_fe = edges_to_frozensets(gold_graph)
                metrics = precision_recall_f1(pred_fe, gold_fe)
                metrics["jaccard"]  = jaccard_index(pred_fe, gold_fe)
                metrics["ged"]      = graph_edit_distance(G, gold_graph)
                metrics["polarity_agreement"] = polarity_agreement(G, gold_graph)
                metrics["chapter"]  = chapter_num
                eval_results.append(metrics)
                _print_step(
                    f"CH{chapter_num:02d}",
                    f"  Eval → P={metrics['precision']:.3f}  "
                    f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
                    f"J={metrics['jaccard']:.3f}",
                )

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  BASELINE PIPELINE COMPLETE — {book_id}")
    print("=" * 60)
    _print_summary_table(summary_rows)

    if args.evaluate and eval_results:
        eval_out = OUTPUTS_DIR / f"{book_id}_eval_results.json"
        eval_out.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
        print(f"Evaluation results saved → {eval_out}\n")

    try:
        from src.utils.build_book_visualizer import build_visualizer
        build_visualizer(book_id, OUTPUTS_DIR)
        print(f"  Full book visualizer: outputs/baseline_graphs/{book_id}_full_book.html")
    except Exception as exc:
        logger.warning("Could not build visualizer: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_baseline",
        description=(
            "Baseline character-graph extraction pipeline. "
            "Processes a full book .txt and produces one GraphML file per chapter."
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
