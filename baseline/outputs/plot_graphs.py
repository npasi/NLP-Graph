"""
plot_graphs.py — Generate PNG visualizations for all chapter graphs.

Produces:
  plots/ch{N:02d}_graph.png      — one network plot per chapter
  plots/overview_metrics.png     — 4-panel summary across all chapters
  plots/cumulative_characters.png — character appearance heatmap

Run from baseline/ :
    python outputs/plot_graphs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
GRAPHS_DIR = ROOT / "baseline_graphs"
PLOTS_DIR  = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

GRAPHML_FILES = sorted(GRAPHS_DIR.glob("*_ch*.graphml"))
SUMMARY_FILES = sorted(GRAPHS_DIR.glob("*_summary.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_color(polarity: str) -> str:
    return {"positive": "#2ecc71", "negative": "#e74c3c"}.get(polarity, "#95a5a6")


def _node_sizes(G: nx.Graph, scale: int = 80) -> list[int]:
    mc = nx.get_node_attributes(G, "mention_count")
    vals = [mc.get(n, 1) for n in G.nodes()]
    if not vals:
        return []
    max_v = max(vals) or 1
    return [max(scale, int(v / max_v * scale * 8)) for v in vals]


def _top_chars(G: nx.Graph, n: int = 12) -> set[str]:
    mc = nx.get_node_attributes(G, "mention_count")
    return {k for k, _ in sorted(mc.items(), key=lambda x: -x[1])[:n]}


# ---------------------------------------------------------------------------
# Per-chapter plots
# ---------------------------------------------------------------------------

def plot_chapter(graphml_path: Path) -> None:
    G = nx.read_graphml(str(graphml_path))
    ch = graphml_path.stem.split("_ch")[-1]
    title = f"Pride & Prejudice — Chapter {int(ch)}"

    if G.number_of_nodes() == 0:
        print(f"  ch{ch}: empty graph, skipping.")
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.axis("off")

    # Layout: spring with heavy nodes anchored
    top = _top_chars(G)
    fixed_pos = None
    try:
        pos = nx.spring_layout(G, seed=42, k=2.2 / max(1, G.number_of_nodes() ** 0.5))
    except Exception:
        pos = nx.circular_layout(G)

    # Edge colours and widths by weight
    edges      = list(G.edges(data=True))
    edge_cols  = [_edge_color(d.get("polarity", "neutral")) for _, _, d in edges]
    weights    = [d.get("weight", 1) for _, _, d in edges]
    max_w      = max(weights) if weights else 1
    edge_widths = [0.4 + 3.0 * w / max_w for w in weights]

    # Node colours: top characters in gold, others in steel blue
    node_colors = ["#f39c12" if n in top else "#5dade2" for n in G.nodes()]
    node_sizes  = _node_sizes(G)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_cols, width=edge_widths, alpha=0.55)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.92)

    # Labels only for top characters
    labels = {n: n for n in G.nodes() if n in top}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                             font_size=7.5, font_weight="bold")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elems = [
        Patch(facecolor="#f39c12", label="Top characters (by mentions)"),
        Patch(facecolor="#5dade2", label="Other characters"),
        Line2D([0], [0], color="#2ecc71", lw=2, label="Positive sentiment"),
        Line2D([0], [0], color="#e74c3c", lw=2, label="Negative sentiment"),
        Line2D([0], [0], color="#95a5a6", lw=2, label="Neutral sentiment"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8, framealpha=0.85)

    # Stats box
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    sents   = [d.get("sentiment", 0) for _, _, d in G.edges(data=True)]
    avg_s   = sum(sents) / len(sents) if sents else 0
    stats   = f"Nodes: {n_nodes}  |  Edges: {n_edges}  |  Density: {density:.3f}  |  AvgSentiment: {avg_s:+.3f}"
    fig.text(0.5, 0.01, stats, ha="center", fontsize=9, color="#555555")

    out = PLOTS_DIR / f"ch{ch}_graph.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Overview metrics plot
# ---------------------------------------------------------------------------

def plot_overview(summaries: list[dict]) -> None:
    summaries = sorted(summaries, key=lambda x: x["chapter"])
    chapters  = [s["chapter"]       for s in summaries]
    nodes     = [s["node_count"]     for s in summaries]
    edges     = [s["edge_count"]     for s in summaries]
    density   = [s["density"]        for s in summaries]
    sentiment = [s["avg_sentiment"]  for s in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Pride & Prejudice — Character Graph Metrics Across Chapters",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(chapters, nodes, color="#5dade2", alpha=0.85)
    ax.set_title("Characters per chapter (nodes)")
    ax.set_xlabel("Chapter"); ax.set_ylabel("Count")
    ax.axhline(np.mean(nodes), color="red", linestyle="--", linewidth=1, label=f"Mean={np.mean(nodes):.0f}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(chapters, edges, color="#a29bfe", alpha=0.85)
    ax.set_title("Co-occurrence edges per chapter")
    ax.set_xlabel("Chapter"); ax.set_ylabel("Count")
    ax.axhline(np.mean(edges), color="red", linestyle="--", linewidth=1, label=f"Mean={np.mean(edges):.0f}")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(chapters, density, "o-", color="#00b894", linewidth=1.8, markersize=4)
    ax.set_title("Graph density per chapter")
    ax.set_xlabel("Chapter"); ax.set_ylabel("Density")
    ax.axhline(np.mean(density), color="red", linestyle="--", linewidth=1, label=f"Mean={np.mean(density):.3f}")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    colors = ["#2ecc71" if s > 0.05 else "#e74c3c" if s < -0.05 else "#95a5a6" for s in sentiment]
    ax.bar(chapters, sentiment, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(np.mean(sentiment), color="red", linestyle="--", linewidth=1,
               label=f"Mean={np.mean(sentiment):.3f}")
    ax.set_title("Average VADER sentiment per chapter")
    ax.set_xlabel("Chapter"); ax.set_ylabel("Compound score")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = PLOTS_DIR / "overview_metrics.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Character presence heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(graphml_files: list[Path]) -> None:
    # Collect all characters and their per-chapter mention counts
    char_chapter: dict[str, dict[int, int]] = {}

    for gf in graphml_files:
        ch_str = gf.stem.split("_ch")[-1]
        try:
            ch = int(ch_str)
        except ValueError:
            continue
        G = nx.read_graphml(str(gf))
        mc = nx.get_node_attributes(G, "mention_count")
        for char, count in mc.items():
            char_chapter.setdefault(char, {})[ch] = count

    # Keep only characters with >= 5 total mentions across all chapters
    totals = {c: sum(v.values()) for c, v in char_chapter.items()}
    top_chars = sorted([c for c, t in totals.items() if t >= 5],
                       key=lambda c: -totals[c])[:40]

    if not top_chars:
        return

    chapters = sorted({ch for d in char_chapter.values() for ch in d})
    matrix   = np.array([
        [char_chapter.get(char, {}).get(ch, 0) for ch in chapters]
        for char in top_chars
    ], dtype=float)

    # Normalise each row to [0,1] for readability
    row_max = matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    matrix_norm = matrix / row_max

    fig, ax = plt.subplots(figsize=(22, max(8, len(top_chars) * 0.35)))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(chapters)))
    ax.set_xticklabels([str(c) for c in chapters], fontsize=7, rotation=0)
    ax.set_yticks(range(len(top_chars)))
    ax.set_yticklabels(top_chars, fontsize=7.5)
    ax.set_xlabel("Chapter", fontsize=11)
    ax.set_title("Character presence across chapters\n(colour intensity = mention count, normalised per character)",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Normalised mentions", fontsize=9)

    fig.tight_layout()
    out = PLOTS_DIR / "character_presence_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Plotting {len(GRAPHML_FILES)} chapter graphs …")
    for gf in GRAPHML_FILES:
        plot_chapter(gf)

    print("\nGenerating overview metrics …")
    summaries = []
    for sf in SUMMARY_FILES:
        try:
            summaries.append(json.loads(sf.read_text(encoding="utf-8")))
        except Exception:
            pass
    if summaries:
        plot_overview(summaries)

    print("\nGenerating character presence heatmap …")
    plot_heatmap(GRAPHML_FILES)

    print(f"\nDone. All plots saved to:\n  {PLOTS_DIR}")
