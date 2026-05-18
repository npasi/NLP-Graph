"""Plot character affinity graphs from prediction-style CSVs (chapter_* columns).

Shared helpers for batch scripts that compare models; does not plot ground-truth-only graphs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _chapter_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        if s.startswith("chapter_"):
            try:
                int(s.split("_", 1)[1])
                cols.append(c)
            except Exception:
                pass
    cols.sort(key=lambda x: int(str(x).strip().split("_", 1)[1]))
    return cols


def _cell_in_unit_interval(x) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    if math.isnan(v):
        return False
    return 0.0 <= v <= 1.0


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    aa = str(a).strip()
    bb = str(b).strip()
    return (aa, bb) if aa <= bb else (bb, aa)


def _pretty_name(name: str, *, max_len: int = 22) -> str:
    s = str(name).strip()
    s = s.split("(", 1)[0].strip()
    s = " ".join(s.split())
    return s if len(s) <= max_len else (s[: max_len - 1].rstrip() + "…")


def _book_title_from_csv_path(csv_path: Path) -> str:
    stem = csv_path.stem
    parts = stem.split("_", 1)
    rest = parts[1] if len(parts) == 2 else stem
    rest = rest.replace("_predicted", "").replace("_affinity", "")
    rest = rest.replace("_pairwise_affinity", "").replace("_sparknotes", "")
    rest = rest.replace("_", " ").strip()
    return rest.title() if rest else stem


# Back-compat alias
_book_title_from_gt_path = _book_title_from_csv_path


def _chapter_weights(df: pd.DataFrame, chapter_col: str) -> Dict[Tuple[str, str], float]:
    weights: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        if not _cell_in_unit_interval(row[chapter_col]):
            continue
        a = str(row["character_1"]).strip()
        b = str(row["character_2"]).strip()
        weights[_pair_key(a, b)] = float(row[chapter_col])
    return weights


def _best_transition(df: pd.DataFrame) -> Optional[int]:
    ch = _chapter_cols(df)
    if len(ch) < 2:
        return None

    mat: List[np.ndarray] = []
    for _, row in df.iterrows():
        vals = [float(row[c]) if _cell_in_unit_interval(row[c]) else np.nan for c in ch]
        arr = np.array(vals, dtype=float)
        if np.all(~np.isfinite(arr)):
            continue
        mat.append(arr)
    if not mat:
        return None

    M = np.vstack(mat)
    diffs = np.abs(M[:, 1:] - M[:, :-1])
    valid = np.isfinite(M[:, 1:]) & np.isfinite(M[:, :-1])
    diffs = np.where(valid, diffs, np.nan)
    mean_d = np.nanmean(diffs, axis=0)
    if not np.any(np.isfinite(mean_d)):
        return None
    return int(np.nanargmax(mean_d))


def _top_edges_union(
    w_a: Dict[Tuple[str, str], float],
    w_b: Dict[Tuple[str, str], float],
    top_k_edges: int,
) -> List[Tuple[Tuple[str, str], float, float]]:
    keys = set(w_a.keys()) | set(w_b.keys())
    ranked = []
    for k in keys:
        a = float(w_a.get(k, float("nan")))
        b = float(w_b.get(k, float("nan")))
        m = np.nanmax([a, b])
        if not np.isfinite(m):
            continue
        ranked.append((k, a, b, float(m)))
    ranked.sort(key=lambda t: t[3], reverse=True)
    trimmed = ranked[: int(top_k_edges)]
    return [(k, a, b) for (k, a, b, _m) in trimmed]


def _layout_from_union(edges: Sequence[Tuple[Tuple[str, str], float, float]], seed: int = 42):
    G = nx.Graph()
    for (u, v), a, b in edges:
        w = float(np.nanmax([a, b]))
        if np.isfinite(w):
            G.add_edge(u, v, weight=w)
    if G.number_of_edges() == 0:
        return {}, G
    return nx.spring_layout(G, seed=seed), G


def _plot_two_chapter_panel(
    *,
    pos,
    union_graph: nx.Graph,
    weights: Dict[Tuple[str, str], float],
    title: str,
    out_path: Path,
) -> None:
    G = nx.Graph()
    for u, v in union_graph.edges():
        key = _pair_key(u, v)
        if key in weights and np.isfinite(weights[key]):
            G.add_edge(u, v, weight=float(weights[key]))
        else:
            G.add_edge(u, v, weight=0.0)

    ws = np.array([float(d["weight"]) for _, _, d in G.edges(data=True)], dtype=float)
    vmin, vmax = 0.0, 1.0
    widths = 0.8 + 5.0 * np.clip(ws, 0.0, 1.0)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="black")
    ax.set_facecolor("black")
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=widths,
        edge_color=ws,
        edge_cmap=plt.cm.RdYlGn,
        edge_vmin=vmin,
        edge_vmax=vmax,
        alpha=0.9,
    )
    deg = dict(union_graph.degree())
    node_sizes = [340 + 160 * deg.get(n, 0) for n in union_graph.nodes()]
    nx.draw_networkx_nodes(
        union_graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color="#ECEFF1",
        edgecolors="#263238",
        linewidths=1.0,
    )
    labels = {n: _pretty_name(n) for n in union_graph.nodes()}
    nx.draw_networkx_labels(union_graph, pos, ax=ax, labels=labels, font_size=11, font_color="white", font_weight="bold")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Affinity score (0→1)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(title, color="white", pad=16)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, facecolor="black")
    plt.close()


def plot_two_chapters(
    csv_path: Path,
    out_dir: Path,
    *,
    col_a: str,
    col_b: str,
    file_prefix: str = "graph_pred",
    title_label: str = "pred",
    top_k_edges: int = 10,
    seed: int = 42,
    layout_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Plot two consecutive chapter graphs from a prediction CSV."""
    df = pd.read_csv(csv_path)
    layout = layout_df if layout_df is not None else df
    w_a = _chapter_weights(df, col_a)
    w_b = _chapter_weights(df, col_b)
    lw_a = _chapter_weights(layout, col_a)
    lw_b = _chapter_weights(layout, col_b)
    edges = _top_edges_union(lw_a, lw_b, top_k_edges=int(top_k_edges))
    pos, union_graph = _layout_from_union(edges, seed=int(seed))

    title = _book_title_from_csv_path(csv_path)
    out_dir = Path(out_dir)
    out_a = out_dir / f"{file_prefix}_{col_a}.png"
    out_b = out_dir / f"{file_prefix}_{col_b}.png"

    _plot_two_chapter_panel(
        pos=pos, union_graph=union_graph, weights=w_a,
        title=f"{title} — {title_label} {col_a}", out_path=out_a,
    )
    _plot_two_chapter_panel(
        pos=pos, union_graph=union_graph, weights=w_b,
        title=f"{title} — {title_label} {col_b}", out_path=out_b,
    )

    return {
        "book": title,
        "csv": str(csv_path),
        "chapter_a": col_a,
        "chapter_b": col_b,
        "title_label": title_label,
        "top_k_edges_union": int(top_k_edges),
        "files": {"a": str(out_a), "b": str(out_b)},
    }


def plot_chapter_graph(
    csv_path: Path,
    chapter_col: str,
    out_path: Path,
    *,
    title_label: str = "pred",
    file_prefix: str = "graph_pred",
    top_k_edges: int = 10,
    max_nodes: int = 12,
    seed: int = 42,
) -> bool:
    """Plot one chapter column from a prediction-style CSV."""
    df = pd.read_csv(csv_path)
    if chapter_col not in df.columns:
        return False
    weights = _chapter_weights(df, chapter_col)
    if len(weights) < 2:
        return False
    title = _book_title_from_csv_path(csv_path)
    out_path = Path(out_path)
    if not str(out_path.name).startswith(file_prefix):
        out_path = out_path.parent / f"{file_prefix}_{chapter_col}.png"
    try:
        _plot_single_chapter_graph(
            weights=weights,
            title=f"{title} — {title_label} ({chapter_col})",
            out_path=out_path,
            top_k_edges=top_k_edges,
            max_nodes=max_nodes,
            seed=seed,
        )
    except ValueError:
        return False
    return True


def _plot_single_chapter_graph(
    *,
    weights: Dict[Tuple[str, str], float],
    title: str,
    out_path: Path,
    top_k_edges: int = 10,
    max_nodes: int = 12,
    seed: int = 42,
) -> None:
    if not weights:
        raise ValueError("No weights to plot")

    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k_edges)]
    G = nx.Graph()
    for (a, b), w in items:
        G.add_edge(a, b, weight=float(w))
    if G.number_of_edges() == 0:
        raise ValueError("No edges after filtering")

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if comps:
        G = G.subgraph(comps[0]).copy()

    if G.number_of_nodes() > max_nodes:
        strength = {n: 0.0 for n in G.nodes()}
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 0.0))
            strength[u] += w
            strength[v] += w
        keep = set(sorted(strength.keys(), key=lambda n: strength[n], reverse=True)[: int(max_nodes)])
        G = G.subgraph(keep).copy()

    pos = nx.spring_layout(G, seed=seed)
    ws = np.array([float(d["weight"]) for _, _, d in G.edges(data=True)], dtype=float)
    vmin, vmax = 0.0, 1.0
    widths = 0.8 + 5.0 * np.clip(ws, 0.0, 1.0)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="black")
    ax.set_facecolor("black")
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, edge_color=ws,
        edge_cmap=plt.cm.RdYlGn, edge_vmin=vmin, edge_vmax=vmax, alpha=0.9,
    )
    deg = dict(G.degree())
    node_sizes = [340 + 160 * deg.get(n, 0) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="#ECEFF1", edgecolors="#263238")
    labels = {n: _pretty_name(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=11, font_color="white", font_weight="bold")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Affinity score (0→1)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(title, color="white", pad=16)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, facecolor="black")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot affinity graphs from a prediction CSV")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--layout-csv", default=None, help="CSV for union layout only (e.g. GT for fixed layout)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chapter-a", default=None)
    ap.add_argument("--chapter-b", default=None)
    ap.add_argument("--file-prefix", default="graph_pred")
    ap.add_argument("--title-label", default="pred")
    ap.add_argument("--top-k-edges", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    layout_path = Path(args.layout_csv) if args.layout_csv else csv_path
    layout_df = pd.read_csv(layout_path)

    df = pd.read_csv(csv_path)
    ch = _chapter_cols(df)
    if len(ch) < 2:
        raise ValueError("Need at least 2 chapters")

    t0 = -1
    if args.chapter_a and args.chapter_b:
        col_a, col_b = str(args.chapter_a), str(args.chapter_b)
        if col_a not in df.columns or col_b not in df.columns:
            raise ValueError(f"Chapter columns not in {csv_path}: {col_a}, {col_b}")
        t0 = ch.index(col_a) if col_a in ch else -1
    else:
        t0 = _best_transition(df)
        if t0 is None:
            raise ValueError("Could not find a valid transition")
        col_a = ch[t0]
        col_b = ch[t0 + 1]

    meta = plot_two_chapters(
        csv_path,
        Path(args.out_dir),
        col_a=col_a,
        col_b=col_b,
        file_prefix=str(args.file_prefix),
        title_label=str(args.title_label),
        top_k_edges=int(args.top_k_edges),
        seed=int(args.seed),
        layout_df=layout_df if layout_path != csv_path else None,
    )
    meta["layout_csv"] = str(layout_path)
    if t0 >= 0:
        meta["transition_index_0based"] = int(t0)

    (Path(args.out_dir) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[plot] wrote: {meta['files']['a']}")
    print(f"[plot] wrote: {meta['files']['b']}")


if __name__ == "__main__":
    main()
