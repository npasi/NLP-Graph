"""CLI wrapper for the Character Identity Layer.

Usage:
    # Fully automatic (default):
    python -m src.step2b_character_identity --book-id 46

    # Human-refined (requires explicit alias file):
    python -m src.step2b_character_identity --book-id the_trial \\
        --identity-mode human_refined \\
        --alias-file data/aliases/the_trial.json --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.character_identity_layer import CharacterIdentityLayer
from src.utils.io import data_dir

IDENTITY_MODES = ("auto_conservative", "human_refined")


def _default_booknlp_root(book_id: str) -> Path:
    return data_dir() / "booknlp_chapter_output" / book_id


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Character Identity Layer on per-chapter BookNLP outputs."
    )
    p.add_argument("--book-id", required=True, help="Book id (e.g. 46 or the_trial).")
    p.add_argument(
        "--booknlp-root", default=None,
        help="Root dir with per-chapter BookNLP folders "
             "(default: data/booknlp_chapter_output/<book-id>/).",
    )
    p.add_argument(
        "--identity-mode",
        default="auto_conservative",
        choices=IDENTITY_MODES,
        help="auto_conservative (default): fully automatic, no alias file used. "
             "human_refined: requires --alias-file.",
    )
    p.add_argument(
        "--alias-file", default=None,
        help="Path to alias JSON file. Only valid with --identity-mode human_refined.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # Validate identity mode / alias file combination first (before filesystem checks).
    if args.identity_mode == "human_refined" and not args.alias_file:
        raise ValueError("--identity-mode human_refined requires --alias-file")
    if args.identity_mode == "auto_conservative" and args.alias_file:
        raise ValueError(
            "Manual alias files are only allowed with --identity-mode human_refined"
        )

    book_id = str(args.book_id)
    booknlp_root = (
        Path(args.booknlp_root) if args.booknlp_root else _default_booknlp_root(book_id)
    )
    if not booknlp_root.exists():
        raise FileNotFoundError(
            f"BookNLP root not found: {booknlp_root}\n"
            "Run run_booknlp_per_chapter.py first."
        )

    alias_file: Optional[Path] = None
    if args.alias_file:
        alias_file = Path(args.alias_file)
        if not alias_file.exists():
            raise FileNotFoundError(f"Alias file not found: {alias_file}")

    layer = CharacterIdentityLayer(
        booknlp_root=booknlp_root,
        alias_file=alias_file,
        identity_mode=args.identity_mode,
    )
    report = layer.run()
    log.info("Report:\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
