"""Generate an HTML *listing* from BookNLP native outputs (Option A).

This does NOT modify BookNLP outputs. It only reads:
  - <run_id>.book (JSON) for character role/event extraction (agent/patient/poss/mod)
  - <run_id>.tokens (TSV) for token-level event tags (column 'event')

It writes a separate HTML file next to the BookNLP HTML.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _read_tokens_event_counts(tokens_path: Path) -> Dict[str, int]:
    if not tokens_path.exists():
        return {}
    # Only load the event column to keep it light.
    df = pd.read_csv(tokens_path, sep="\t", quoting=3, usecols=["event"], keep_default_na=False)
    if df.empty:
        return {}
    c = Counter(str(x) for x in df["event"].tolist())
    return dict(c.most_common())


def _canonical_from_mentions(m: Dict[str, Any]) -> str:
    # Mentions dict has 'proper/common/pronoun' lists of {n,c}
    for bucket in ("proper", "common", "pronoun"):
        arr = m.get(bucket) or []
        if arr:
            return str(arr[0].get("n", "")).strip()
    return ""


def _mentions_string(m: Dict[str, Any], max_items: int = 20) -> str:
    items: List[Tuple[str, int]] = []
    for bucket in ("proper", "common", "pronoun"):
        for it in (m.get(bucket) or []):
            n = str(it.get("n", "")).strip()
            c = int(it.get("c", 0) or 0)
            if n:
                items.append((n, c))
    items.sort(key=lambda x: -x[1])
    parts = [f"{html.escape(n)} ({c})" for n, c in items[:max_items]]
    return " / ".join(parts)


def _format_pred_list(items: List[Dict[str, Any]], *, max_items: int = 80) -> str:
    # items look like {"w": token_text, "i": token_id}
    if not items:
        return "<span class='muted'>—</span>"
    c = Counter(str(x.get("w", "")).strip() for x in items if str(x.get("w", "")).strip())
    top = c.most_common(max_items)
    return ", ".join(f"{html.escape(w)} <span class='muted'>({n})</span>" for w, n in top)


def _render_character_block(ch: Dict[str, Any]) -> str:
    cid = ch.get("id")
    count = int(ch.get("count", 0) or 0)
    gender = ch.get("g", "unknown")
    mentions = ch.get("mentions") or {}
    canonical = _canonical_from_mentions(mentions) or f"character_{cid}"
    mention_str = _mentions_string(mentions, max_items=20)

    agent = _format_pred_list(ch.get("agent") or [])
    patient = _format_pred_list(ch.get("patient") or [])
    poss = _format_pred_list(ch.get("poss") or [])
    mod = _format_pred_list(ch.get("mod") or [])

    return f"""
    <details class="char" open>
      <summary>
        <span class="name">{html.escape(canonical)}</span>
        <span class="meta">coref_id={html.escape(str(cid))} · mentions={count} · g={html.escape(str(gender))}</span>
      </summary>
      <div class="box">
        <div><b>Mentions</b>: {mention_str or "<span class='muted'>—</span>"}</div>
        <div class="grid">
          <div class="cell"><b>Agent (subject → verbs)</b><div class="list">{agent}</div></div>
          <div class="cell"><b>Patient (object → verbs)</b><div class="list">{patient}</div></div>
          <div class="cell"><b>Poss (owned nouns)</b><div class="list">{poss}</div></div>
          <div class="cell"><b>Mod (attributes/modifiers)</b><div class="list">{mod}</div></div>
        </div>
      </div>
    </details>
    """


def generate_listings_html(*, run_dir: Path, run_id: str, output_html: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    book_path = run_dir / f"{run_id}.book"
    tokens_path = run_dir / f"{run_id}.tokens"
    triplets_tsv_path = run_dir / f"{run_id}.event_triplets.tsv"
    triplets_jsonl_path = run_dir / f"{run_id}.event_triplets.jsonl"
    output_html = output_html or (run_dir / f"{run_id}.book.listings.html")

    data = json.loads(book_path.read_text(encoding="utf-8"))
    chars = list(data.get("characters") or [])

    # Sort characters by mention count (desc).
    chars.sort(key=lambda c: -int(c.get("count", 0) or 0))

    event_counts = _read_tokens_event_counts(tokens_path)
    ev_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td class='num'>{int(v)}</td></tr>"
        for k, v in list(event_counts.items())[:25]
    )

    triplets_rows_html = ""
    triplets_summary_html = "<span class='muted'>Triplets file not found.</span>"
    if triplets_tsv_path.exists():
        try:
            tdf = pd.read_csv(triplets_tsv_path, sep="\t", quoting=3, keep_default_na=False)
            triplets_summary_html = (
                f"<div class='muted' style='margin-top:6px'>"
                f"Loaded <code>{html.escape(triplets_tsv_path.name)}</code>: "
                f"{int(tdf.shape[0])} triplets.</div>"
            )
            # Show a preview (first 200) to keep HTML reasonable.
            preview = tdf.head(200)
            triplets_rows_html = preview.to_html(index=False, escape=True, border=0)
        except Exception as exc:
            triplets_summary_html = f"<span class='muted'>Failed to read triplets TSV: {html.escape(str(exc))}</span>"

    css = """
    :root{--bg:#0b0d12;--panel:#111522;--muted:#9aa4b2;--fg:#e8edf3;--line:#243045;--accent:#7aa2f7;}
    *{box-sizing:border-box} body{margin:0;font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--fg)}
    a{color:var(--accent)}
    .wrap{max-width:1200px;margin:0 auto;padding:24px}
    h1{font-size:20px;margin:0 0 8px 0}
    .sub{color:var(--muted);margin:0 0 16px 0}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin:12px 0}
    .muted{color:var(--muted)}
    table{width:100%;border-collapse:collapse}
    th,td{border-bottom:1px solid var(--line);padding:8px 10px;font-size:13px;text-align:left}
    .num{text-align:right;font-variant-numeric:tabular-nums}
    details.char{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:12px 0;overflow:hidden}
    details.char summary{cursor:pointer;padding:12px 14px;font-weight:650;display:flex;gap:10px;align-items:baseline}
    .name{font-size:14px}
    .meta{color:var(--muted);font-weight:500;font-size:12px}
    .box{padding:12px 14px;border-top:1px solid var(--line)}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
    .cell{border:1px solid var(--line);border-radius:12px;padding:10px;background:#0d1220}
    .list{margin-top:6px;line-height:1.45}
    code{background:#0d1220;border:1px solid var(--line);padding:2px 6px;border-radius:8px;color:#cfe1ff}
    .pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted);margin-right:6px;font-size:12px}
    """

    html_out = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BookNLP listings — {html.escape(run_id)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="wrap">
    <h1>BookNLP listings (native outputs) — {html.escape(run_id)}</h1>
    <p class="sub">
      This file is generated by reading <code>{html.escape(book_path.name)}</code> and <code>{html.escape(tokens_path.name)}</code>.
      It does not change BookNLP outputs.
    </p>

    <div class="card">
      <div><b>Event tags (from .tokens → column <code>event</code>)</b></div>
      <div class="muted" style="margin-top:6px">Top values + counts (first 25).</div>
      <table>
        <thead><tr><th>tag</th><th class="num">count</th></tr></thead>
        <tbody>{ev_rows or "<tr><td colspan='2' class='muted'>No data.</td></tr>"}</tbody>
      </table>
    </div>

    <div class="card">
      <div><b>Event triplets (agent, predicate, patient)</b></div>
      <div class="muted" style="margin-top:6px">
        This section is generated from derived files next to BookNLP outputs:
        <span class="pill"><a href="{html.escape(triplets_tsv_path.name)}">{html.escape(triplets_tsv_path.name)}</a></span>
        <span class="pill"><a href="{html.escape(triplets_jsonl_path.name)}">{html.escape(triplets_jsonl_path.name)}</a></span>
      </div>
      {triplets_summary_html}
      <div style="margin-top:10px; overflow:auto; max-height:520px; border:1px solid var(--line); border-radius:12px; padding:8px; background:#0d1220">
        {triplets_rows_html or "<div class='muted'>No preview available.</div>"}
      </div>
    </div>

    <div class="card">
      <div><b>Characters + Event/role extraction</b></div>
      <div class="muted" style="margin-top:6px">
        For each character cluster: agent/patient/poss/mod are taken from <code>{html.escape(book_path.name)}</code>.
      </div>
    </div>

    {''.join(_render_character_block(c) for c in chars)}
  </div>
</body>
</html>
"""

    output_html.write_text(html_out, encoding="utf-8")
    return output_html


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate an HTML listing from BookNLP .book + .tokens outputs.")
    p.add_argument("--run-dir", required=True, help="Folder that contains <run_id>.book and <run_id>.tokens")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", default=None, help="Output HTML path (default: <run_id>.book.listings.html in run-dir)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out = generate_listings_html(
        run_dir=Path(args.run_dir),
        run_id=args.run_id,
        output_html=Path(args.output) if args.output else None,
    )
    logger.info("Wrote listings HTML to %s", out)


if __name__ == "__main__":
    main()

