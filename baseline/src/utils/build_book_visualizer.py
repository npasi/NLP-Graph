#!/usr/bin/env python3
"""
build_book_visualizer.py — Build a self-contained HTML book graph visualizer.

Reads all .graphml files for a given book_id and produces a single HTML file
with a chapter slider and stable (frozen) node positions.

CLI:
    python src/utils/build_book_visualizer.py \\
      --book_id jekyll_hyde \\
      --output_dir outputs/baseline_graphs

Programmatic:
    from src.utils.build_book_visualizer import build_visualizer
    build_visualizer(book_id, output_dir)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def _parse_graphml(path: Path) -> tuple[list[dict], list[dict], int]:
    """Parse a GraphML file; return (nodes, edges, chapter_num)."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"gml": _GRAPHML_NS}

    key_map: dict[str, str] = {}
    for key_el in root.findall("gml:key", ns):
        key_map[key_el.get("id", "")] = key_el.get("attr.name", "")

    graph = root.find("gml:graph", ns)
    if graph is None:
        return [], [], 0

    chapter_num = 0
    for data in graph.findall("gml:data", ns):
        if key_map.get(data.get("key", "")) == "chapter" and data.text:
            chapter_num = int(data.text)
            break

    nodes: list[dict] = []
    for node_el in graph.findall("gml:node", ns):
        node_id = node_el.get("id", "")
        node: dict = {"id": node_id, "label": node_id, "mentions": 0}
        for data in node_el.findall("gml:data", ns):
            attr = key_map.get(data.get("key", ""))
            if attr == "name" and data.text:
                node["label"] = data.text
            elif attr == "mention_count" and data.text:
                node["mentions"] = int(data.text)
        nodes.append(node)

    edges: list[dict] = []
    for edge_el in graph.findall("gml:edge", ns):
        edge: dict = {
            "from": edge_el.get("source", ""),
            "to": edge_el.get("target", ""),
            "weight": 1,
            "sentiment": 0.0,
            "polarity": "neutral",
        }
        for data in edge_el.findall("gml:data", ns):
            attr = key_map.get(data.get("key", ""))
            if attr == "weight" and data.text:
                edge["weight"] = int(data.text)
            elif attr == "sentiment" and data.text:
                edge["sentiment"] = round(float(data.text), 4)
            elif attr == "polarity" and data.text:
                edge["polarity"] = data.text
        edges.append(edge)

    return nodes, edges, chapter_num


def _chapter_title(data_raw_dir: Path, book_id: str, num: int) -> str:
    """Return the chapter title from the first non-empty line of the chapter file."""
    p = data_raw_dir / f"{book_id}_ch{num:02d}.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return f"Chapter {num}"


def _build_book_data(book_id: str, output_dir: Path) -> dict:
    """Aggregate all chapter graphs into a single BOOK_DATA structure."""
    pattern = re.compile(r"_ch(\d+)\.graphml$")
    graphml_files = sorted(
        (p for p in output_dir.glob(f"{book_id}_ch*.graphml") if pattern.search(p.name)),
        key=lambda p: int(pattern.search(p.name).group(1)),
    )
    if not graphml_files:
        raise FileNotFoundError(
            f"No .graphml files for book_id='{book_id}' in {output_dir}"
        )

    # raw chapter text files live two levels above output_dir → .../data/raw/
    data_raw_dir = output_dir.parent.parent / "data" / "raw"

    chapters: list[dict] = []
    node_registry: dict[str, dict] = {}

    for gml_path in graphml_files:
        nodes, edges, chapter_num = _parse_graphml(gml_path)
        title = _chapter_title(data_raw_dir, book_id, chapter_num)

        for n in nodes:
            if n["id"] not in node_registry:
                node_registry[n["id"]] = {
                    "id": n["id"],
                    "label": n["label"],
                    "total_mentions": n["mentions"],
                    "first_chapter": chapter_num,
                }
            else:
                node_registry[n["id"]]["total_mentions"] += n["mentions"]

        chapters.append({"num": chapter_num, "title": title, "nodes": nodes, "edges": edges})

    book_title = book_id.replace("_", " ").title()
    return {
        "title": book_title,
        "chapters": chapters,
        "all_nodes": list(node_registry.values()),
    }


# ---------------------------------------------------------------------------
# HTML template — placeholders replaced at generation time:
#   |||BOOK_TITLE|||   → book title string
#   |||BOOK_DATA|||    → JSON-encoded BOOK_DATA object
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>|||BOOK_TITLE||| — Character Graph</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d1a;color:#e0e0e0;font-family:'SF Mono','Courier New',monospace;height:100vh;display:flex;flex-direction:column;overflow:hidden}
#header{display:flex;align-items:center;padding:0 20px;background:#12122a;border-bottom:1px solid #2a2a4a;height:50px;flex-shrink:0}
#book-title{font-size:14px;font-weight:bold;color:#7b8cde;white-space:nowrap;flex:1}
#chapter-label{flex:1;font-size:13px;color:#a0a0c0;text-align:center}
#header-right{flex:1}
#graph-container{flex:1;position:relative;background:#1a1a2e;overflow:hidden}
#network{width:100%;height:100%}
#loading{position:absolute;inset:0;background:#1a1a2eee;display:flex;align-items:center;justify-content:center;color:#7b8cde;font-size:14px;gap:10px;z-index:10}
.spinner{width:16px;height:16px;border:2px solid #7b8cde;border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#footer{background:#12122a;border-top:1px solid #2a2a4a;padding:10px 20px 8px;flex-shrink:0}
#controls{display:flex;align-items:center;gap:12px;margin-bottom:6px}
.btn{background:#2a2a4a;color:#a0a0c0;border:1px solid #3a3a5a;padding:4px 14px;cursor:pointer;font-size:13px;font-family:inherit;border-radius:3px;transition:background .15s;user-select:none}
.btn:hover:not(:disabled){background:#3a3a5a;color:#e0e0e0}
.btn:disabled{opacity:.3;cursor:not-allowed}
#slider{flex:1;accent-color:#7b8cde;cursor:pointer}
#stats{font-size:11px;color:#606080;text-align:center;letter-spacing:.03em}
</style>
</head>
<body>
<div id="header">
  <span id="book-title">|||BOOK_TITLE|||</span>
  <span id="chapter-label"></span>
  <span id="header-right"></span>
</div>
<div id="graph-container">
  <div id="loading"><div class="spinner"></div>Computing layout…</div>
  <div id="network"></div>
</div>
<div id="footer">
  <div id="controls">
    <button class="btn" id="btn-prev">‹ Prev</button>
    <input type="range" id="slider" min="0" step="1" value="0">
    <button class="btn" id="btn-next">Next ›</button>
  </div>
  <div id="stats">Active nodes: — | Active edges: — | Density: — | Avg sentiment: —</div>
</div>
<script>
const BOOK_DATA = |||BOOK_DATA|||;

const K = BOOK_DATA.chapters.length;
const allNodeMap = Object.fromEntries(BOOK_DATA.all_nodes.map(n => [n.id, n]));
const chMaxMentions = BOOK_DATA.chapters.map(ch => ch.nodes.reduce((m, n) => Math.max(m, n.mentions), 1));
const chMaxWeight   = BOOK_DATA.chapters.map(ch => ch.edges.reduce((m, e) => Math.max(m, e.weight), 1));

// Union edges for layout only — always hidden so they never display
const seenKeys = new Set();
const unionEdges = [];
BOOK_DATA.chapters.forEach(ch => ch.edges.forEach(e => {
  const k = [e.from, e.to].sort().join('\x01');
  if (!seenKeys.has(k)) {
    seenKeys.add(k);
    unionEdges.push({ id: 'u_' + k, from: e.from, to: e.to, hidden: true });
  }
}));

// Initialise DataSets with all nodes as ghost placeholders + union edges
const nodesDS = new vis.DataSet(BOOK_DATA.all_nodes.map(n => ({
  id: n.id, label: n.label, size: 8,
  color: { background: '#2d2d4e', border: '#3d3d6e' },
  font: { color: '#505078', size: 11 },
})));
const edgesDS = new vis.DataSet(unionEdges);

// Build network — physics runs once for layout, then freezes
const network = new vis.Network(
  document.getElementById('network'),
  { nodes: nodesDS, edges: edgesDS },
  {
    nodes: { shape: 'dot', borderWidth: 1.5 },
    edges: { smooth: { type: 'continuous', roundness: 0.15 } },
    physics: {
      enabled: true,
      barnesHut: {
        gravitationalConstant: -4000,
        centralGravity: 0.4,
        springLength: 150,
        springConstant: 0.04,
      },
      stabilization: { iterations: 400, updateInterval: 25 },
    },
    interaction: { hover: true, tooltipDelay: 120 },
  }
);

let curIdx = 0, ready = false;

function onReady() {
  if (ready) return;
  ready = true;
  // Freeze positions permanently — node positions never change after this
  network.setOptions({ physics: { enabled: false } });
  document.getElementById('loading').style.display = 'none';
  switchChapter(0);
}

network.on('stabilizationIterationsDone', onReady);
setTimeout(onReady, 8000); // fallback if event never fires

function edgeCol(polarity) {
  return polarity === 'positive' ? '#27ae60' : polarity === 'negative' ? '#e74c3c' : '#95a5a6';
}

function switchChapter(idx) {
  curIdx = idx;
  const ch = BOOK_DATA.chapters[idx];
  const active = new Set(ch.nodes.map(n => n.id));
  const chNodeMap = Object.fromEntries(ch.nodes.map(n => [n.id, n]));
  const maxM = chMaxMentions[idx];
  const maxW = chMaxWeight[idx];

  // Update node appearance — positions are never touched
  nodesDS.update(BOOK_DATA.all_nodes.map(n => {
    if (active.has(n.id)) {
      const cn = chNodeMap[n.id];
      const ec = ch.edges.filter(e => e.from === n.id || e.to === n.id).length;
      const a  = allNodeMap[n.id];
      return {
        id: n.id,
        size: 10 + Math.round((cn.mentions / maxM) * 40),
        color: { background: '#4ecdc4', border: '#38b2ac' },
        font: { color: '#fff', size: 13, strokeWidth: 2, strokeColor: '#1a1a2e' },
        title: '<b>' + n.label + '</b>'
          + '<br>Mentions this chapter: ' + cn.mentions
          + '<br>Total mentions: ' + (a ? a.total_mentions : '?')
          + '<br>First appeared: ch ' + (a ? a.first_chapter : '?')
          + '<br>Edges in this chapter: ' + ec,
      };
    }
    return {
      id: n.id, size: 8,
      color: { background: '#2d2d4e', border: '#3d3d6e' },
      font: { color: '#505078', size: 11, strokeWidth: 0 },
      title: n.label + ' (not in this chapter)',
    };
  }));

  // Swap chapter edges — union edges remain permanently hidden
  edgesDS.remove(edgesDS.getIds().filter(id => String(id).startsWith('ch_')));
  edgesDS.add(ch.edges.map((e, i) => {
    const col = edgeCol(e.polarity);
    const s   = (e.sentiment >= 0 ? '+' : '') + e.sentiment.toFixed(3);
    return {
      id: 'ch_' + idx + '_' + i,
      from: e.from, to: e.to,
      width: 1 + Math.round((e.weight / maxW) * 5),
      color: { color: col, highlight: col, hover: col },
      title: e.from + ' ↔ ' + e.to
        + '<br>co-occurrences: ' + e.weight
        + '<br>sentiment: ' + s,
    };
  }));

  // Header / slider / button state
  document.getElementById('chapter-label').textContent =
    'Chapter ' + ch.num + ' / ' + K + '  —  ' + ch.title;
  document.getElementById('slider').value = idx;
  document.getElementById('btn-prev').disabled = (idx === 0);
  document.getElementById('btn-next').disabled = (idx === K - 1);

  // Stats bar
  const n = ch.nodes.length, e = ch.edges.length;
  const d   = n > 1 ? (2 * e / (n * (n - 1))).toFixed(4) : '0.0000';
  const avg = e ? (ch.edges.reduce((s, x) => s + x.sentiment, 0) / e).toFixed(4) : '0.0000';
  document.getElementById('stats').textContent =
    'Active nodes: ' + n + '  |  Active edges: ' + e
    + '  |  Density: ' + d + '  |  Avg sentiment: ' + avg;
}

// Controls
document.getElementById('slider').max = K - 1;
document.getElementById('slider').addEventListener('input', ev => switchChapter(+ev.target.value));
document.getElementById('btn-prev').addEventListener('click', () => {
  if (curIdx > 0) switchChapter(curIdx - 1);
});
document.getElementById('btn-next').addEventListener('click', () => {
  if (curIdx < K - 1) switchChapter(curIdx + 1);
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'ArrowLeft'  && curIdx > 0)     switchChapter(curIdx - 1);
  if (ev.key === 'ArrowRight' && curIdx < K - 1) switchChapter(curIdx + 1);
});
</script>
</body>
</html>
"""


def _generate_html(book_data: dict) -> str:
    title = book_data["title"]
    data_json = json.dumps(book_data, ensure_ascii=False, separators=(",", ":"))
    # Prevent </script> in data from closing the script tag prematurely
    data_json = data_json.replace("</", "<\\/")
    return (
        _HTML
        .replace("|||BOOK_TITLE|||", title)
        .replace("|||BOOK_DATA|||", data_json)
    )


def build_visualizer(book_id: str, output_dir: Path) -> Path:
    """Build HTML visualizer for *book_id*; return path to the saved file."""
    output_dir = Path(output_dir)
    book_data = _build_book_data(book_id, output_dir)
    html = _generate_html(book_data)
    out_path = output_dir / f"{book_id}_full_book.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="build_book_visualizer",
        description="Build a self-contained HTML visualizer for a book's character graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--book_id", required=True, metavar="ID",
                        help="Book identifier matching {book_id}_ch*.graphml files.")
    parser.add_argument("--output_dir", required=True, metavar="DIR",
                        help="Directory containing the .graphml files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output_dir does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        out = build_visualizer(args.book_id, output_dir)
        print(f"Visualizer saved → {out}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
