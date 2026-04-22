"""
plot_interactive.py — Generate interactive HTML network visualizations.

Each chapter gets a self-contained HTML file with:
  - Zoom / pan (mouse wheel + drag)
  - Hover tooltips (character name, mentions, edges)
  - Physics simulation: high-degree nodes (protagonists) gravitate to centre
  - Node size proportional to mention count
  - Edge thickness proportional to co-occurrence weight
  - Edge colour: green=positive, red=negative, grey=neutral

Plus an index.html linking all chapters.

Run from baseline/:
    python outputs/plot_interactive.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
GRAPHS_DIR = ROOT / "baseline_graphs"
PLOTS_DIR  = ROOT / "plots" / "interactive"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GRAPHML_FILES = sorted(GRAPHS_DIR.glob("*_ch*.graphml"))
SUMMARY_FILES = sorted(GRAPHS_DIR.glob("*_summary.json"))

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
POLARITY_COLOR = {
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

# Top-N protagonists get a distinct gold colour
TOP_N_GOLD = 10

# Physics options — Barnes-Hut with strong gravity pulls hubs to centre
PHYSICS_OPTIONS = """
{
  "physics": {
    "enabled": true,
    "solver": "barnesHut",
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.4,
      "springLength": 180,
      "springConstant": 0.04,
      "damping": 0.12,
      "avoidOverlap": 0.3
    },
    "stabilization": {
      "enabled": true,
      "iterations": 300,
      "updateInterval": 25
    }
  },
  "interaction": {
    "hover": true,
    "zoomView": true,
    "dragView": true,
    "navigationButtons": true,
    "keyboard": { "enabled": true }
  },
  "edges": {
    "smooth": { "type": "continuous" }
  }
}
"""


# ---------------------------------------------------------------------------
# Per-chapter plot
# ---------------------------------------------------------------------------

def _chapter_num(path: Path) -> int:
    try:
        return int(path.stem.split("_ch")[-1])
    except ValueError:
        return 0


def plot_chapter(graphml_path: Path) -> Path | None:
    G = nx.read_graphml(str(graphml_path))
    ch = _chapter_num(graphml_path)

    if G.number_of_nodes() == 0:
        print(f"  ch{ch:02d}: empty — skipping.")
        return None

    # ---- rank nodes by degree (number of edges) then mention_count --------
    degree  = dict(G.degree())
    mc_attr = nx.get_node_attributes(G, "mention_count")

    rank_score = {
        n: degree[n] * 1000 + mc_attr.get(n, 0)
        for n in G.nodes()
    }
    sorted_nodes = sorted(rank_score, key=rank_score.get, reverse=True)
    top_nodes    = set(sorted_nodes[:TOP_N_GOLD])

    # ---- scaling -----------------------------------------------------------
    mc_vals   = [mc_attr.get(n, 1) for n in G.nodes()]
    max_mc    = max(mc_vals) if mc_vals else 1

    w_vals    = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_w     = max(w_vals) if w_vals else 1

    # ---- build pyvis network -----------------------------------------------
    title_str = f"Pride &amp; Prejudice — Chapter {ch}"
    net = Network(
        height="95vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#ecf0f1",
        heading=title_str,
        directed=False,
        notebook=False,
    )
    net.set_options(PHYSICS_OPTIONS)

    for node in G.nodes():
        mc    = mc_attr.get(node, 1)
        deg   = degree[node]
        size  = 12 + int(mc / max_mc * 50)     # 12–62 px
        mass  = 1.0 + deg * 0.4                 # heavier → pulled to centre

        if node in top_nodes:
            color  = "#f39c12"
            border = "#e67e22"
            font_size = 18
        else:
            color  = "#5dade2"
            border = "#2e86c1"
            font_size = 12

        tooltip = (
            f"<b>{node}</b><br>"
            f"Mentions: {mc}<br>"
            f"Edges: {deg}<br>"
            f"Rank: #{sorted_nodes.index(node)+1}"
        )

        net.add_node(
            node,
            label=node if node in top_nodes else (node if deg >= 3 else ""),
            title=tooltip,
            size=size,
            color={"background": color, "border": border,
                   "highlight": {"background": "#f1c40f", "border": "#f39c12"}},
            font={"size": font_size, "color": "#ffffff",
                  "strokeWidth": 3, "strokeColor": "#000000"},
            mass=mass,
            borderWidth=2,
        )

    for u, v, data in G.edges(data=True):
        w        = data.get("weight", 1)
        pol      = data.get("polarity", "neutral")
        sent     = data.get("sentiment", 0.0)
        width    = 0.5 + 6.0 * w / max_w
        color    = POLARITY_COLOR.get(pol, "#95a5a6")
        tooltip  = (
            f"<b>{u} ↔ {v}</b><br>"
            f"Co-occurrences: {w:.0f}<br>"
            f"Sentiment: {sent:+.3f} ({pol})"
        )
        net.add_edge(u, v,
                     title=tooltip,
                     width=width,
                     color={"color": color, "highlight": "#f1c40f", "opacity": 0.75})

    # Stats bar injected into HTML
    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    density  = nx.density(G)
    sents    = [d.get("sentiment", 0) for _, _, d in G.edges(data=True)]
    avg_sent = sum(sents) / len(sents) if sents else 0.0
    stats_html = (
        f'<div style="font-family:monospace;font-size:13px;color:#bdc3c7;'
        f'background:#16213e;padding:6px 16px;border-radius:6px;display:inline-block">'
        f'Nodes: <b>{n_nodes}</b> &nbsp;|&nbsp; '
        f'Edges: <b>{n_edges}</b> &nbsp;|&nbsp; '
        f'Density: <b>{density:.4f}</b> &nbsp;|&nbsp; '
        f'Avg sentiment: <b>{avg_sent:+.3f}</b>'
        f'</div>'
    )

    out_path = PLOTS_DIR / f"ch{ch:02d}_graph.html"
    net.save_graph(str(out_path))

    # Inject stats bar and navigation arrows into the saved HTML
    html = out_path.read_text(encoding="utf-8")
    prev_ch = ch - 1
    next_ch = ch + 1

    prev_link = (
        f'<a href="ch{prev_ch:02d}_graph.html" '
        f'style="color:#f39c12;font-size:22px;text-decoration:none">&#8678;</a>'
        if prev_ch >= 1 else ""
    )
    next_link = (
        f'<a href="ch{next_ch:02d}_graph.html" '
        f'style="color:#f39c12;font-size:22px;text-decoration:none">&#8680;</a>'
        if next_ch <= 61 else ""
    )
    nav = (
        '<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        'z-index:9999;display:flex;gap:12px;align-items:center">'
        + prev_link
        + f'<span style="color:#ecf0f1;font-size:15px;font-weight:bold">Ch {ch} / 61</span>'
        + next_link
        + '<a href="index.html" style="color:#a29bfe;font-size:13px;margin-left:8px">&#8962; Index</a>'
        '</div>'
        '<div style="position:fixed;bottom:10px;left:50%;transform:translateX(-50%);z-index:9999">'
        + stats_html
        + '</div>'
    )

    html = html.replace("<body>", "<body>\n" + nav, 1)
    out_path.write_text(html, encoding="utf-8")

    print(f"  ch{ch:02d}_graph.html  ({n_nodes} nodes, {n_edges} edges)")
    return out_path


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def build_index(summaries: list[dict]) -> None:
    by_ch = {s["chapter"]: s for s in summaries}
    rows  = ""
    for ch in sorted(by_ch):
        s = by_ch[ch]
        rows += (
            f'<tr>'
            f'<td><a href="ch{ch:02d}_graph.html">Chapter {ch}</a></td>'
            f'<td>{s["node_count"]}</td>'
            f'<td>{s["edge_count"]}</td>'
            f'<td>{s["density"]:.4f}</td>'
            f'<td>{s["avg_sentiment"]:+.4f}</td>'
            f'<td>{"&nbsp;".join(c["name"] for c in s.get("top_characters", [])[:5])}</td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pride &amp; Prejudice — Character Graphs</title>
<style>
  body  {{ background:#1a1a2e; color:#ecf0f1; font-family:sans-serif; padding:30px }}
  h1   {{ color:#f39c12; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px }}
  th   {{ background:#16213e; color:#a29bfe; padding:8px 12px; text-align:left }}
  td   {{ padding:7px 12px; border-bottom:1px solid #2c3e50 }}
  tr:hover td {{ background:#16213e }}
  a    {{ color:#5dade2; text-decoration:none }}
  a:hover {{ color:#f39c12 }}
</style>
</head>
<body>
<h1>Pride &amp; Prejudice — Character Graph Index</h1>
<p style="color:#95a5a6">Click a chapter to open the interactive graph.
Scroll to zoom · Drag to pan · Hover nodes/edges for details.</p>
<table>
<thead><tr>
  <th>Chapter</th><th>Nodes</th><th>Edges</th>
  <th>Density</th><th>Avg Sentiment</th><th>Top 5 characters</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>"""

    out = PLOTS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"  index.html  ({len(by_ch)} chapters)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating interactive HTML plots for {len(GRAPHML_FILES)} chapters …\n")
    for gf in GRAPHML_FILES:
        plot_chapter(gf)

    print("\nBuilding index …")
    summaries = []
    for sf in SUMMARY_FILES:
        try:
            summaries.append(json.loads(sf.read_text(encoding="utf-8")))
        except Exception:
            pass
    if summaries:
        build_index(summaries)

    print(f"\nDone. Open in browser:\n  {PLOTS_DIR / 'index.html'}")
