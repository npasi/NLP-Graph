"""Export a compact "friend pack": graphs + a single metrics JSON.

Creates:
  data/share_with_friend/
    graphs/{train|test}/{book_id}/{model}/graph_pred.png
    metrics_3models_train_test.json

Models included (3-way):
  - pretrained_base
  - old_1ep_MSE
  - new_3ep_MSE

Metrics include the existing regression metrics (MSE/RMSE/MAE/Pearson/Spearman)
plus two NLP-style metrics for graph/ranking use-cases:
  - nDCG@K (default K=10), averaged over chapters
  - F1 for "strong edges" (GT>=tau_gt, Pred>=tau_pred)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402


def _book_title_from_gt_path(gt_path: Path) -> str:
    # e.g. "95_ulysses_affinity.csv" -> "Ulysses"
    stem = gt_path.stem
    parts = stem.split("_", 1)
    rest = parts[1] if len(parts) == 2 else stem
    rest = rest.replace("_affinity", "").replace("_pairwise_affinity", "").replace("_sparknotes", "")
    rest = rest.replace("_", " ").strip()
    return rest.title() if rest else stem


def _pretty_name(name: str, *, max_len: int = 18) -> str:
    """Make node labels compact and readable."""
    s = str(name).strip()
    # Remove parenthetical aliases: "Little Father Time (Little Jude)" -> "Little Father Time"
    s = s.split("(", 1)[0].strip()
    s = " ".join(s.split())
    # Shorten overly long labels
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


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


def _dcg(rels: Sequence[float]) -> float:
    # DCG with gains = rel (already in [0,1]) and log2 discount.
    out = 0.0
    for i, r in enumerate(rels, start=1):
        out += float(r) / math.log2(i + 1)
    return out


def _ndcg_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    # gt/pred are 1D arrays for the same set of pairs.
    if gt.size == 0:
        return float("nan")
    k_eff = int(min(k, gt.size))
    if k_eff <= 1:
        return float("nan")
    order = np.argsort(-pred)[:k_eff]
    ideal = np.argsort(-gt)[:k_eff]
    dcg = _dcg(gt[order].tolist())
    idcg = _dcg(gt[ideal].tolist())
    return float(dcg / idcg) if idcg > 0 else float("nan")


def _f1_strong_edges(gt: np.ndarray, pred: np.ndarray, tau_gt: float, tau_pred: float) -> Dict[str, float]:
    if gt.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "support_pos": 0}
    y_true = gt >= tau_gt
    y_hat = pred >= tau_pred
    tp = int(np.sum(y_true & y_hat))
    fp = int(np.sum((~y_true) & y_hat))
    fn = int(np.sum(y_true & (~y_hat)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "support_pos": int(np.sum(y_true))}


def _regression_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    # gt/pred are 1D arrays aligned, in [0,1]
    if gt.size == 0:
        return {
            "n_cells": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
            "gt_mean": float("nan"),
            "gt_std": float("nan"),
        }
    mse = float(mean_squared_error(gt, pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(gt, pred))
    pred_std = float(np.std(pred))
    gt_std = float(np.std(gt))
    pearson = float(pearsonr(gt, pred)[0]) if gt_std > 1e-9 and pred_std > 1e-9 else 0.0
    spearman = float(spearmanr(gt, pred).correlation) if gt_std > 1e-9 and pred_std > 1e-9 else 0.0
    return {
        "n_cells": int(gt.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "pred_mean": float(np.mean(pred)),
        "pred_std": pred_std,
        "gt_mean": float(np.mean(gt)),
        "gt_std": gt_std,
    }


def _aligned_cells(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Align GT pairs to predicted pairs using evaluate.py fuzzy finder.

    Returns (chapter_cols, gt_matrix, pred_matrix) with same shape [pairs x chapters].
    Only keeps GT rows that can be matched to some predicted row.
    """
    from longformer_pipeline.evaluate import _find_pred_row  # local import, keeps CLI snappy

    ch_cols = _chapter_cols(gt_df)
    pred_ch_cols = _chapter_cols(pred_df)
    # Use GT chapter columns as the reference; predicted files are generated in GT-style,
    # so they should have the same chapter_n names.
    if not ch_cols or not pred_ch_cols:
        return [], np.zeros((0, 0)), np.zeros((0, 0))

    gt_rows: List[np.ndarray] = []
    pred_rows: List[np.ndarray] = []
    for _, gt_row in gt_df.iterrows():
        g1 = str(gt_row["character_1"]).strip()
        g2 = str(gt_row["character_2"]).strip()
        pr = _find_pred_row(g1, g2, pred_df)
        if pr is None:
            continue
        gt_vals: List[float] = []
        pr_vals: List[float] = []
        for c in ch_cols:
            gv = float(gt_row[c]) if _cell_in_unit_interval(gt_row[c]) else np.nan
            pv = float(pr[c]) if _cell_in_unit_interval(pr[c]) else np.nan
            gt_vals.append(gv)
            pr_vals.append(pv)
        gt_rows.append(np.array(gt_vals, dtype=float))
        pred_rows.append(np.array(pr_vals, dtype=float))
    if not gt_rows:
        return ch_cols, np.zeros((0, len(ch_cols))), np.zeros((0, len(ch_cols)))
    return ch_cols, np.vstack(gt_rows), np.vstack(pred_rows)


def _flatten_comparable(gt_mat: np.ndarray, pred_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(gt_mat) & np.isfinite(pred_mat)
    g = gt_mat[valid].astype(float)
    p = pred_mat[valid].astype(float)
    return g, p


def _static_edge_weights(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    ch = _chapter_cols(df)
    if not ch:
        return {}
    weights: Dict[Tuple[str, str], List[float]] = {}
    for _, row in df.iterrows():
        a = str(row["character_1"]).strip()
        b = str(row["character_2"]).strip()
        key = _pair_key(a, b)
        vals = []
        for c in ch:
            if _cell_in_unit_interval(row[c]):
                vals.append(float(row[c]))
        if vals:
            weights.setdefault(key, []).extend(vals)
    return {k: float(np.mean(v)) for k, v in weights.items() if v}


def _delta_edge_weights_for_best_transition(df: pd.DataFrame) -> Tuple[Optional[int], Dict[Tuple[str, str], float]]:
    """Return (best_transition_idx, edge_weights) where weights are |Δ| from chapter t->t+1.

    best_transition_idx is 1-based in the plot's convention (transition 1 means chapter_1->chapter_2).
    """
    ch = _chapter_cols(df)
    if len(ch) < 2:
        return None, {}

    # Build per-pair time series with NaNs for invalid cells.
    pairs: List[Tuple[str, str]] = []
    mat: List[np.ndarray] = []
    for _, row in df.iterrows():
        a = str(row["character_1"]).strip()
        b = str(row["character_2"]).strip()
        vals = []
        for c in ch:
            vals.append(float(row[c]) if _cell_in_unit_interval(row[c]) else np.nan)
        arr = np.array(vals, dtype=float)
        if np.all(~np.isfinite(arr)):
            continue
        pairs.append(_pair_key(a, b))
        mat.append(arr)
    if not mat:
        return None, {}

    M = np.vstack(mat)  # [pairs x chapters]
    diffs = np.abs(M[:, 1:] - M[:, :-1])
    valid = np.isfinite(M[:, 1:]) & np.isfinite(M[:, :-1])
    diffs = np.where(valid, diffs, np.nan)
    mean_d = np.nanmean(diffs, axis=0)  # [transitions]
    if not np.any(np.isfinite(mean_d)):
        return None, {}

    best_t0 = int(np.nanargmax(mean_d))  # 0-based transition index
    weights: Dict[Tuple[str, str], float] = {}
    for idx, key in enumerate(pairs):
        d = diffs[idx, best_t0]
        if np.isfinite(d):
            weights[key] = float(d)
    return best_t0 + 1, weights  # 1-based for humans


def _plot_graph(
    *,
    weights: Dict[Tuple[str, str], float],
    title: str,
    out_path: Path,
    top_k_edges: int = 10,
    min_weight: Optional[float] = None,
    max_nodes: int = 12,
    style: str = "clean",
    seed: int = 42,
) -> None:
    if not weights:
        return

    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    if min_weight is not None:
        items = [kv for kv in items if kv[1] >= min_weight]
    if top_k_edges is not None and top_k_edges > 0:
        items = items[: int(top_k_edges)]

    G = nx.Graph()
    for (a, b), w in items:
        G.add_edge(a, b, weight=float(w))

    if G.number_of_edges() == 0:
        return

    # Keep only the largest connected component to avoid scattered micro-graphs.
    if G.number_of_nodes() > 1:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if comps:
            G = G.subgraph(comps[0]).copy()

    # If still too many nodes, keep the most "central" characters by strength.
    if G.number_of_nodes() > max_nodes:
        strength: Dict[str, float] = {n: 0.0 for n in G.nodes()}
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 0.0))
            strength[u] += w
            strength[v] += w
        keep = set(sorted(strength.keys(), key=lambda n: strength[n], reverse=True)[:max_nodes])
        G = G.subgraph(keep).copy()
        # Drop isolated nodes after pruning
        isolates = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolates)

    # Layout (stable for small graphs)
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=seed, k=None)

    ws = np.array([float(d["weight"]) for _, _, d in G.edges(data=True)], dtype=float)

    if style == "legacy":
        # Absolute [0,1] scale + plasma (older plots).
        vmin, vmax = 0.0, 1.0
        widths = 1.0 + 5.0 * np.clip(ws, 0.0, 1.0)
        cmap = plt.cm.plasma
        add_colorbar = False
        node_color = "#B0BEC5"
        label_all = False
    else:
        # Clean: per-graph scaling so red=min, green=max, informative even when clustered.
        vmin = float(np.nanmin(ws))
        vmax = float(np.nanmax(ws))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if abs(vmax - vmin) < 1e-6:
            vmin = max(0.0, vmin - 0.05)
            vmax = min(1.0, vmax + 0.05)
        widths = 0.8 + 5.0 * np.clip((ws - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        cmap = plt.cm.RdYlGn
        add_colorbar = True
        node_color = "#ECEFF1"
        label_all = True
    colors = ws

    degrees = dict(G.degree())
    node_sizes = [320 + 160 * degrees.get(n, 0) for n in G.nodes()]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="black")
    ax.set_facecolor("black")
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=widths,
        edge_color=colors,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        alpha=0.85,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_color,
        alpha=0.98,
        linewidths=1.0,
        edgecolors="#263238",
    )

    # Labeling strategy: label all if small; otherwise label only top nodes by degree.
    labels: Dict[str, str] = {}
    if label_all and G.number_of_nodes() <= 14:
        labels = {n: _pretty_name(n) for n in G.nodes()}
    else:
        top = sorted(G.nodes(), key=lambda n: degrees.get(n, 0), reverse=True)[:14]
        labels = {n: (_pretty_name(n) if label_all else n) for n in top}
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels=labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
    )

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label(f"Affinity score (min={vmin:.2f}, max={vmax:.2f})", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(title, color="white", pad=16)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, facecolor="black")
    plt.close()


@dataclass(frozen=True)
class ModelCfg:
    name: str
    # where to find predictions for train/test
    pred_dir_train: str
    pred_dir_test: str


MODELS: List[ModelCfg] = [
    ModelCfg("pretrained_base", "data/predictions/pretrained_base", "data/predictions/pretrained_base"),
    ModelCfg("old_1ep_MSE", "data/predictions/prevprefix_full_1ep", "data/predictions/prevprefix_test_1ep"),
    ModelCfg("new_3ep_MSE", "data/predictions/prevprefix_full_3ep_mse", "data/predictions/prevprefix_test_3ep_mse"),
]


def _gt_path(gt_dir: Path, book_id: str) -> Optional[Path]:
    for p in gt_dir.glob(f"{book_id}_*.csv"):
        return p
    return None


def _pred_path(pred_dir: Path, book_id: str) -> Optional[Path]:
    p = pred_dir / f"{book_id}_predicted.csv"
    return p if p.is_file() else None


def export_pack(
    *,
    split_path: Path,
    gt_dir: Path,
    out_root: Path,
    k_ndcg: int,
    tau_gt: float,
    tau_pred: float,
    top_k_edges: int,
    max_nodes: int,
    style: str,
    include_delta: bool,
) -> Dict:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    out: Dict = {
        "params": {
            "k_ndcg": k_ndcg,
            "tau_gt_strong": tau_gt,
            "tau_pred_strong": tau_pred,
            "top_k_edges": top_k_edges,
            "models": [m.name for m in MODELS],
            "note": "Metrics computed on comparable cells where both GT and pred are in [0,1] after fuzzy pair alignment.",
        },
        "splits": {},
    }

    for split_name in ("train", "test"):
        book_ids: List[str] = [str(x) for x in split.get(split_name, [])]
        out["splits"][split_name] = {}

        for bid in book_ids:
            gt_path = _gt_path(gt_dir, bid)
            if gt_path is None:
                continue

            gt_df = pd.read_csv(gt_path)
            book_title = _book_title_from_gt_path(gt_path)

            out["splits"][split_name].setdefault(bid, {"gt_csv": str(gt_path), "models": {}})

            for m in MODELS:
                pred_dir = Path(m.pred_dir_train if split_name == "train" else m.pred_dir_test)
                pred_path = _pred_path(pred_dir, bid)
                if pred_path is None:
                    continue

                pred_df = pd.read_csv(pred_path)
                pred_weights = _static_edge_weights(pred_df)
                pred_best_t, pred_delta_weights = _delta_edge_weights_for_best_transition(pred_df)
                pred_graph_path = out_root / "graphs" / split_name / bid / m.name / "graph_pred.png"
                _plot_graph(
                    weights=pred_weights,
                    title=f"{split_name.upper()} • {book_title} (id={bid}) — {m.name} mean predicted affinity",
                    out_path=pred_graph_path,
                    top_k_edges=top_k_edges,
                    max_nodes=max_nodes,
                    style=style,
                )

                if include_delta:
                    pred_best_t, pred_delta_weights = _delta_edge_weights_for_best_transition(pred_df)
                    if pred_best_t is not None and pred_delta_weights:
                        pred_delta_path = out_root / "graphs" / split_name / bid / m.name / "graph_delta.png"
                        _plot_graph(
                            weights=pred_delta_weights,
                            title=f"{split_name.upper()} • {book_title} (id={bid}) — {m.name} change graph |Δ| (best transition={pred_best_t})",
                            out_path=pred_delta_path,
                            top_k_edges=top_k_edges,
                            max_nodes=max_nodes,
                            style=style,
                        )

                ch_cols, gt_mat, pred_mat = _aligned_cells(gt_df, pred_df)
                gt_flat, pred_flat = _flatten_comparable(gt_mat, pred_mat)
                base = _regression_metrics(gt_flat, pred_flat)

                # nDCG@K averaged across chapter columns
                ndcgs: List[float] = []
                if gt_mat.size and pred_mat.size:
                    for j, _c in enumerate(ch_cols):
                        gcol = gt_mat[:, j]
                        pcol = pred_mat[:, j]
                        v = np.isfinite(gcol) & np.isfinite(pcol)
                        if int(v.sum()) < 2:
                            continue
                        nd = _ndcg_at_k(gcol[v], pcol[v], k_ndcg)
                        if np.isfinite(nd):
                            ndcgs.append(float(nd))
                ndcg_mean = float(np.mean(ndcgs)) if ndcgs else float("nan")

                f1 = _f1_strong_edges(gt_flat, pred_flat, tau_gt=tau_gt, tau_pred=tau_pred)

                artifacts: Dict[str, str] = {"graph_pred_png": str(pred_graph_path)}
                if include_delta:
                    artifacts["graph_pred_delta_png"] = str(
                        out_root / "graphs" / split_name / bid / m.name / "graph_delta.png"
                    )

                out["splits"][split_name][bid]["models"][m.name] = {
                    "pred_csv": str(pred_path),
                    "metrics": {
                        **base,
                        "ndcg_at_k": ndcg_mean,
                        "ndcg_k": int(k_ndcg),
                        **{f"strong_{k}": v for k, v in f1.items()},
                    },
                    "artifacts": artifacts,
                }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", default="data/splits/train_val_test.json")
    ap.add_argument("--gt-dir", default="data/dataset/ground_truth")
    ap.add_argument("--out-root", default="data/share_with_friend")
    ap.add_argument("--k-ndcg", type=int, default=10)
    ap.add_argument("--tau-gt", type=float, default=0.7, help="GT threshold for a 'strong' edge (for F1).")
    ap.add_argument("--tau-pred", type=float, default=0.5, help="Prediction threshold for a 'strong' edge (for F1).")
    ap.add_argument("--top-k-edges", type=int, default=10, help="Top-K edges to draw in each graph PNG.")
    ap.add_argument("--max-nodes", type=int, default=12, help="Max nodes to keep in each graph PNG.")
    ap.add_argument("--style", choices=["clean", "legacy"], default="clean")
    ap.add_argument("--no-delta", action="store_true", help="Do not export change graphs (only mean affinity graphs).")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    payload = export_pack(
        split_path=Path(args.split_path),
        gt_dir=Path(args.gt_dir),
        out_root=out_root,
        k_ndcg=int(args.k_ndcg),
        tau_gt=float(args.tau_gt),
        tau_pred=float(args.tau_pred),
        top_k_edges=int(args.top_k_edges),
        max_nodes=int(args.max_nodes),
        style=str(args.style),
        include_delta=not bool(args.no_delta),
    )

    out_json = out_root / "metrics_3models_train_test.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[export] Wrote: {out_json}")


if __name__ == "__main__":
    main()

