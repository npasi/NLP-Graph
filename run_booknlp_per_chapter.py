from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Optional

from booknlp.booknlp import BookNLP

logger = logging.getLogger(__name__)

_MODEL_BASE_URL = "https://people.ischool.berkeley.edu/~dbamman/booknlp_models"

_BIG_MODELS = [
    "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model",
    "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model",
    "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model",
]

_SMALL_MODELS = [
    "entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model",
    "coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model",
    "speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model",
]


def _ensure_booknlp_models(model_path: Path, model_size: str) -> None:
    """BookNLP upstream uses http:// URLs which may 403; we prefetch via https://."""
    model_path.mkdir(parents=True, exist_ok=True)
    needed = _BIG_MODELS if model_size == "big" else _SMALL_MODELS
    for fname in needed:
        out = model_path / fname
        if out.is_file() and out.stat().st_size > 0:
            continue
        url = f"{_MODEL_BASE_URL}/{fname}"
        logger.info("Downloading BookNLP model: %s", fname)
        urllib.request.urlretrieve(url, out)


def _default_chapters_dir(book_id: str) -> Path:
    return Path("data") / "books" / book_id / "chapters"


def _default_output_root(book_id: str) -> Path:
    return Path("data") / "booknlp_chapter_output" / book_id


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BookNLP per chapter (one output folder per chapter).")
    p.add_argument("--book-id", required=True, help="Book id used by the splitter (e.g. 46).")
    p.add_argument("--chapters-dir", default=None, help="Override chapters dir (default: data/books/<book-id>/chapters).")
    p.add_argument("--output-root", default=None, help="Override output root (default: data/booknlp_chapter_output/<book-id>/).")
    p.add_argument("--model-path", default=None, help="Cache dir for BookNLP model weights (default: data/booknlp_models).")
    p.add_argument("--model-size", choices=["small", "big"], default="big")
    p.add_argument("--pipeline", default="entity,quote,supersense,event,coref")
    p.add_argument("--max-chapters", type=int, default=None)
    p.add_argument("--language", default="en")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def run_booknlp_per_chapter(
    *,
    book_id: str,
    chapters_dir: Path,
    output_root: Path,
    language: str = "en",
    model_size: str = "big",
    pipeline: str = "entity,quote,supersense,event,coref",
    model_path: Optional[Path] = None,
    max_chapters: Optional[int] = None,
) -> None:
    model_params = {"pipeline": pipeline, "model": model_size}
    if model_path is not None:
        model_params["model_path"] = str(model_path)
        _ensure_booknlp_models(model_path, model_size)

    booknlp = BookNLP(language, model_params)

    chapter_files = sorted(chapters_dir.glob("chapter_*.txt"))
    if not chapter_files:
        raise FileNotFoundError(f"No chapter_*.txt found in {chapters_dir.resolve()}")
    if max_chapters is not None:
        chapter_files = chapter_files[: int(max_chapters)]

    output_root.mkdir(parents=True, exist_ok=True)

    for ch_path in chapter_files:
        chapter_id = int(ch_path.stem.split("_")[-1])  # chapter_000 -> 0
        run_id = f"booknlp_{book_id}_chapter_{chapter_id:04d}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        booknlp.process(str(ch_path), str(run_dir), run_id)
        logger.info("OK: %s -> %s", run_id, run_dir.resolve())


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    book_id = str(args.book_id)
    chapters_dir = Path(args.chapters_dir) if args.chapters_dir else _default_chapters_dir(book_id)
    output_root = Path(args.output_root) if args.output_root else _default_output_root(book_id)
    model_path = Path(args.model_path) if args.model_path else (Path("data") / "booknlp_models")
    model_path.mkdir(parents=True, exist_ok=True)

    run_booknlp_per_chapter(
        book_id=book_id,
        chapters_dir=chapters_dir,
        output_root=output_root,
        language=str(args.language),
        model_size=str(args.model_size),
        pipeline=str(args.pipeline),
        model_path=model_path,
        max_chapters=args.max_chapters,
    )


if __name__ == "__main__":
    main()

