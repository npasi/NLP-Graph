"""Run BookNLP on a single full-book .txt (no splitting, no graphs).

This is the simplest way to "analyze a book using only BookNLP":
  - tokens (.tokens)
  - entities (.entities)
  - quotes (.quotes, if produced)
  - book metadata (.book)

Plus a small JSON summary (top character clusters) for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_builder import BookNLPGraphBuilder
from utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


def _character_summary(builder: BookNLPGraphBuilder, book_meta: Dict[str, Any], top_k: int = 25) -> List[Dict[str, Any]]:
    chars = book_meta.get("characters", []) or []
    out: List[Dict[str, Any]] = []
    for c in chars:
        canonical, aliases = builder._canonical_character_info(c)  # stable across the project
        mentions = c.get("mentions", {}) or {}
        proper = sum(int(m.get("c", 0) or 0) for m in (mentions.get("proper", []) or []))
        common = sum(int(m.get("c", 0) or 0) for m in (mentions.get("common", []) or []))
        pronoun = sum(int(m.get("c", 0) or 0) for m in (mentions.get("pronoun", []) or []))
        out.append(
            {
                "coref_id": int(c.get("id", -1)) if str(c.get("id", "")).isdigit() else c.get("id"),
                "canonical": canonical,
                "mentions_total": proper + common + pronoun,
                "mentions_proper": proper,
                "mentions_common": common,
                "mentions_pronoun": pronoun,
                "aliases": aliases[:20],
            }
        )
    out.sort(key=lambda r: -int(r.get("mentions_total", 0)))
    return out[: max(1, int(top_k))]


def run_booknlp_only(
    *,
    input_txt: Path,
    output_root: Path,
    run_id: str,
    model_size: str = "big",
    top_k: int = 25,
) -> Dict[str, Any]:
    text = input_txt.read_text(encoding="utf-8", errors="replace")
    builder = BookNLPGraphBuilder(
        output_root=ensure_dir(output_root),
        tmp_root=ensure_dir(output_root / "_tmp"),
        model_size=model_size,
        filter_mode="curated",
    )
    outputs = builder._run_booknlp(text, run_id=run_id)

    run_dir = Path(output_root) / run_id
    summary = {
        "input_txt": str(input_txt),
        "output_dir": str(run_dir),
        "run_id": run_id,
        "model_size": model_size,
        "chars_top": _character_summary(builder, outputs.book_meta, top_k=top_k),
        "tokens_rows": int(outputs.tokens.shape[0]),
        "entities_rows": int(outputs.entities.shape[0]),
        "quotes_rows": int(outputs.quotes.shape[0]),
        "book_meta_keys": sorted(list(outputs.book_meta.keys())),
    }
    write_json(run_dir / f"{run_id}.summary.json", summary, indent=2)
    logger.info("Wrote BookNLP summary to %s", run_dir / f"{run_id}.summary.json")

    # Note: we intentionally do NOT overwrite BookNLP's native HTML report.
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BookNLP on a full-book .txt (no splitting).")
    p.add_argument("--input", required=True, help="Path to a UTF-8 .txt file.")
    p.add_argument("--output-root", default="data/output/booknlp_full", help="Where to cache BookNLP outputs.")
    p.add_argument("--run-id", default=None, help="Run id / folder name (default: input filename stem).")
    p.add_argument("--model-size", choices=["small", "big"], default="big")
    p.add_argument("--top-k", type=int, default=25, help="How many top character clusters to keep in the summary.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_txt = Path(args.input)
    if not input_txt.exists():
        raise FileNotFoundError(str(input_txt))

    output_root = Path(args.output_root)
    run_id = args.run_id or input_txt.stem
    rep = run_booknlp_only(
        input_txt=input_txt,
        output_root=output_root,
        run_id=run_id,
        model_size=args.model_size,
        top_k=args.top_k,
    )
    print(json.dumps(rep["chars_top"][:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

