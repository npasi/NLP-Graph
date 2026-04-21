"""
grid_search_alpha.py — Grid search over α for the Markov-conditioned pipeline.

New utility with no equivalent in baseline/.  Finds the optimal alpha by
running the full baseline_injected pipeline for each candidate value and
measuring F1 against the gold standard.

Usage
-----
    python src/utils/grid_search_alpha.py --book_id pride_prejudice

The script must be run from the baseline_injected/ root directory so that
module imports resolve correctly.

Algorithm
---------
For each α in {0.3, 0.5, 0.7, 0.9}:
  1. Run the complete pipeline: split → booknlp → extract_edges → build_graph.
  2. For each chapter with a gold-standard graph, compute P/R/F1.
  3. Average F1 across all evaluated chapters.

Outputs
-------
  Prints a table: alpha | precision | recall | F1
  Saves best alpha to: outputs/best_alpha_{book_id}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO noise during grid search
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grid_search_alpha")

ALPHA_GRID: list[float] = [0.3, 0.5, 0.7, 0.9]

ROOT        = Path(__file__).parent.parent.parent   # baseline_injected/
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_GOLD   = ROOT / "data" / "gold_standard"
OUTPUTS_DIR = ROOT / "outputs" / "baseline_injected_graphs"


def _run_alpha(
    alpha: float,
    book_id: str,
    chapter_files: list[Path],
    window: int,
    min_freq: int,
) -> dict[str, float]:
    """Run the full pipeline for a single alpha and return aggregated metrics.

    Args:
        alpha:         Markov decay factor.
        book_id:       Short book identifier.
        chapter_files: Sorted list of chapter file paths.
        window:        Co-occurrence window size.
        min_freq:      Minimum co-occurrence frequency threshold.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1`` averaged over all
        chapters that have gold-standard graphs.  Returns zeros if no gold
        graphs are found.
    """
    import networkx as nx

    from src.utils.text_cleaner import clean_chapter_text, tokenize
    from src.baseline_injected.run_booknlp import run_booknlp
    from src.baseline_injected.extract_edges import extract_edges
    from src.baseline_injected.build_graph import build_graph
    from src.evaluation.graph_metrics import (
        edges_to_frozensets,
        precision_recall_f1,
    )

    G_prev: nx.Graph = nx.Graph()
    all_precision: list[float] = []
    all_recall:    list[float] = []
    all_f1:        list[float] = []

    for chapter_path in chapter_files:
        stem = chapter_path.stem
        try:
            chapter_num = int(stem.split("_ch")[-1])
        except (ValueError, IndexError):
            continue

        # BookNLP (cached from previous full run)
        try:
            character_positions = run_booknlp(
                chapter_path=chapter_path,
                book_id=book_id,
                chapter_num=chapter_num,
                processed_dir=DATA_PROC,
            )
        except Exception:
            G_prev = nx.Graph()
            continue

        if not character_positions:
            continue

        raw_text = chapter_path.read_text(encoding="utf-8", errors="replace")
        tokens   = tokenize(clean_chapter_text(raw_text))

        edges = extract_edges(
            tokens=tokens,
            character_positions=character_positions,
            window=window,
            min_freq=min_freq,
            G_prev=G_prev,
        )

        G_k = build_graph(
            edges=edges,
            character_positions=character_positions,
            book_id=book_id,
            chapter_num=chapter_num,
            output_dir=OUTPUTS_DIR / f"alpha_{alpha:.1f}",
            G_prev=G_prev if G_prev.number_of_nodes() > 0 else None,
            alpha=alpha,
        )

        # Evaluate if gold graph exists
        gold_path = DATA_GOLD / f"{book_id}_ch{chapter_num:02d}.graphml"
        if gold_path.exists():
            gold_graph = nx.read_graphml(str(gold_path))
            pred_fe = edges_to_frozensets(G_k)
            gold_fe = edges_to_frozensets(gold_graph)
            metrics = precision_recall_f1(pred_fe, gold_fe)
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])
            all_f1.append(metrics["f1"])

        G_prev = G_k

    if not all_f1:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": round(sum(all_precision) / len(all_precision), 4),
        "recall":    round(sum(all_recall)    / len(all_recall),    4),
        "f1":        round(sum(all_f1)        / len(all_f1),        4),
    }


def run_grid_search(args: argparse.Namespace) -> None:
    """Execute the grid search and print results.

    Args:
        args: Parsed :class:`argparse.Namespace` from the CLI.
    """
    from src.utils.chapter_splitter import split_book_into_chapters

    book_id = args.book_id

    # Locate the chapter files — they may already exist from a prior run
    existing = sorted(
        DATA_RAW.glob(f"{book_id}_ch*.txt"),
        key=lambda p: int(p.stem.split("_ch")[-1]) if "_ch" in p.stem else 0,
    )

    if not existing:
        # No cached chapters — find the source book
        book_path = args.book_path
        if book_path is None:
            logger.error(
                "No chapter files found in data/raw/ for '%s' and --book_path "
                "was not provided.  Run the full pipeline first or supply "
                "--book_path.", book_id
            )
            sys.exit(1)
        existing = split_book_into_chapters(Path(book_path), book_id, DATA_RAW)
        existing = sorted(
            existing,
            key=lambda p: int(p.stem.split("_ch")[-1]) if "_ch" in p.stem else 0,
        )

    gold_files = list(DATA_GOLD.glob(f"{book_id}_ch*.graphml"))
    if not gold_files:
        logger.error(
            "No gold-standard graphs found in data/gold_standard/ for '%s'.\n"
            "Expected files named: %s_ch{N:02d}.graphml", book_id, book_id
        )
        sys.exit(1)

    print(f"\nGrid search over alpha for book: {book_id}")
    print(f"Gold standard chapters available: {len(gold_files)}")
    print(f"Alpha candidates: {ALPHA_GRID}\n")

    results: list[dict] = []

    # Table header
    header = f"{'Alpha':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for alpha in ALPHA_GRID:
        print(f"  Running alpha={alpha:.1f} …", end="", flush=True)
        metrics = _run_alpha(
            alpha=alpha,
            book_id=book_id,
            chapter_files=existing,
            window=args.window,
            min_freq=args.min_freq,
        )
        metrics["alpha"] = alpha
        results.append(metrics)
        print(
            f"\r  {alpha:>6.1f}  {metrics['precision']:>10.4f}  "
            f"{metrics['recall']:>8.4f}  {metrics['f1']:>8.4f}"
        )

    print(sep)

    # Best alpha
    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest alpha: {best['alpha']:.1f}  (F1={best['f1']:.4f})\n")

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"best_alpha_{book_id}.json"
    out_path.write_text(
        json.dumps({"book_id": book_id, "best_alpha": best["alpha"], "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid_search_alpha",
        description=(
            "Grid search over the Markov decay factor alpha for a given book.  "
            "Requires gold-standard graphs in data/gold_standard/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--book_id",
        required=True,
        metavar="ID",
        help="Short book identifier (must match filenames in data/raw/ and data/gold_standard/).",
    )
    parser.add_argument(
        "--book_path",
        default=None,
        metavar="PATH",
        help=(
            "Path to the full book .txt if chapters have not been split yet.  "
            "Optional when data/raw/{book_id}_ch*.txt files already exist."
        ),
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
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    project_root = ROOT
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    run_grid_search(args)
