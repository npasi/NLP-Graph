from __future__ import annotations

import argparse
import logging
import os
import re as _re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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


_BOOKNLP_PATCHED = False


def _patch_booknlp_for_windows() -> None:
    """Patch BookNLP to work on Windows paths + modern transformers.

    BookNLP upstream derives the HuggingFace model id via `model_file.split("/")[-1]`.
    On Windows, backslashes leak into the model name and HF rejects it.

    Also, BookNLP checkpoints may contain stale keys removed by newer transformers;
    loading with `strict=False` prevents hard failures.
    """
    global _BOOKNLP_PATCHED
    if _BOOKNLP_PATCHED:
        return

    from booknlp.english import entity_tagger as _et
    from booknlp.english import litbank_coref as _lc
    from booknlp.english import bert_qa as _qa
    from booknlp.english.tagger import Tagger
    from booknlp.english.bert_coref_quote_pronouns import BERTCorefTagger
    from booknlp.english.speaker_attribution import BERTSpeakerID
    import booknlp.common.sequence_layered_reader as seq_reader
    import pkg_resources
    import torch

    def _basename_model_key(model_file: str) -> str:
        base = os.path.basename(model_file)
        base = _re.sub("google_bert", "google/bert", base)
        base = _re.sub(r"\.model$", "", base)
        return base

    def _patched_entity_init(self, model_file, model_tagset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tagset = seq_reader.read_tagset(model_tagset)
        supersense_path = pkg_resources.resource_filename(_et.__name__, "data/supersense.tagset")
        self.supersense_tagset = seq_reader.read_tagset(supersense_path)
        self.model = Tagger(
            freeze_bert=False,
            base_model=_basename_model_key(model_file),
            tagset_flat={"EVENT": 1, "O": 1},
            supersense_tagset=self.supersense_tagset,
            tagset=self.tagset,
            device=device,
        )
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
        wns_path = pkg_resources.resource_filename(_et.__name__, "data/wordnet.first.sense")
        self.wns = self.read_wn(wns_path)

    _et.LitBankEntityTagger.__init__ = _patched_entity_init

    def _patched_coref_init(self, modelFile, gender_cats, pronominalCorefOnly=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERTCorefTagger(
            gender_cats=gender_cats,
            freeze_bert=True,
            base_model=_basename_model_key(modelFile),
            pronominalCorefOnly=pronominalCorefOnly,
        )
        self.model.load_state_dict(torch.load(modelFile, map_location=device), strict=False)
        self.model.to(device)
        self.model.eval()

    _lc.LitBankCoref.__init__ = _patched_coref_init

    def _patched_qa_init(self, modelFile):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERTSpeakerID(base_model=_basename_model_key(modelFile))
        self.model.load_state_dict(torch.load(modelFile, map_location=device), strict=False)
        self.model.to(device)
        self.model.eval()

    _qa.QuotationAttribution.__init__ = _patched_qa_init

    _BOOKNLP_PATCHED = True
    logger.info("Applied Windows + transformers compatibility patches to BookNLP.")


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
    return Path("data") / "output" / "chapters" / book_id


def _default_output_root(book_id: str) -> Path:
    return Path("data") / "output" / "booknlp" / book_id


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BookNLP per chapter (one output folder per chapter).")
    p.add_argument("--book-id", required=True, help="Book id used by the splitter (e.g. 46).")
    p.add_argument("--chapters-dir", default=None, help="Override chapters dir (default: data/output/chapters/<book-id>).")
    p.add_argument("--output-root", default=None, help="Override output root (default: data/output/booknlp/<book-id>).")
    p.add_argument("--model-path", default=None, help="Cache dir for BookNLP model weights (default: data/models).")
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
    _patch_booknlp_for_windows()
    model_params = {"pipeline": pipeline, "model": model_size}
    if model_path is not None:
        model_params["model_path"] = str(model_path)
        _ensure_booknlp_models(model_path, model_size)

    # Import after project-root sys.path insertion so our `pkg_resources.py` shim
    # (repo root) is discoverable even when running this file as `python src/...`.
    from booknlp.booknlp import BookNLP

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
    model_path = Path(args.model_path) if args.model_path else (Path("data") / "models")
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

