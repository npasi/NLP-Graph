"""Step 2 of the Longformer pipeline: scoring wrapper.

This module loads a (fine-tuned) Longformer regression model and produces a
continuous sentiment score in [0, 1] for inputs produced by
`input_builder.build_longformer_input()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import torch
from transformers import LongformerForSequenceClassification, LongformerTokenizer


class LongformerScorer:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        *,
        norm: Literal["sigmoid", "clip"] = "sigmoid",
    ):
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.norm: Literal["sigmoid", "clip"] = norm

        print(f"[scorer] Loading model from: {model_path}")
        print(f"[scorer] Device: {self.device}")
        print(f"[scorer] Normalization: {self.norm}")

        self.tokenizer = LongformerTokenizer.from_pretrained(model_path)
        self.model = LongformerForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e1_end_id = self.tokenizer.convert_tokens_to_ids("[/E1]")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        self.e2_end_id = self.tokenizer.convert_tokens_to_ids("[/E2]")

        print(f"[scorer] Vocab size: {len(self.tokenizer)}")
        print(
            "[scorer] Special token IDs: "
            f"E1={self.e1_id}, /E1={self.e1_end_id}, "
            f"E2={self.e2_id}, /E2={self.e2_end_id}"
        )
        print(
            "[scorer] Model loaded successfully. Parameters: "
            f"{sum(p.numel() for p in self.model.parameters()):,}"
        )

        assert (
            self.e1_id != self.tokenizer.unk_token_id
        ), "[E1] not in vocabulary! Model was not trained with special tokens."
        assert (
            self.e2_id != self.tokenizer.unk_token_id
        ), "[E2] not in vocabulary! Model was not trained with special tokens."

    def _normalize(self, raw_logit: torch.Tensor) -> torch.Tensor:
        """Map raw logits to [0, 1] consistently across baseline + finetuned."""
        if self.norm == "sigmoid":
            return torch.sigmoid(raw_logit)
        # "clip" (legacy / per-spec): directly clamp raw logit into [0,1]
        return torch.clamp(raw_logit, 0.0, 1.0)

    def _global_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create Longformer global attention mask for a batch."""
        # input_ids: [batch, seq]
        gam = torch.zeros_like(input_ids, dtype=torch.long)
        # CLS / <s> is position 0 for LongformerTokenizer.
        gam[:, 0] = 1

        for tok_id in (self.e1_id, self.e1_end_id, self.e2_id, self.e2_end_id):
            if tok_id is None:
                continue
            gam = torch.where(input_ids == tok_id, torch.ones_like(gam), gam)

        return gam

    def score(self, text: str) -> float:
        """Score a single Longformer-formatted input text."""
        if text is None or not str(text).strip():
            print("[score] Warning: empty input text. Returning -1.0")
            return -1.0

        inputs = self.tokenizer(
            text,
            max_length=4096,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        global_attention_mask = self._global_attention_mask(inputs["input_ids"])
        inputs["global_attention_mask"] = global_attention_mask

        with torch.no_grad():
            outputs = self.model(**inputs)

        raw_logit = outputs.logits.squeeze()
        score = self._normalize(raw_logit).squeeze().item()
        score_f = float(score)

        n_tokens = (inputs["attention_mask"] == 1).sum().item()
        n_global = (global_attention_mask == 1).sum().item()
        print(f"[score] Input tokens: {n_tokens}, Global attention tokens: {n_global}")
        print(f"[score] Raw logit: {float(raw_logit.item()):.4f}, Score: {score_f:.4f}")

        assert 0.0 <= score_f <= 1.0, f"Score out of range: {score_f}"
        return score_f

    def score_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """Score multiple texts efficiently in batches."""
        if texts is None:
            return []
        if len(texts) == 0:
            return []

        print(f"[batch] Processing {len(texts)} texts in batches of {batch_size}")

        scores: List[float] = []
        valid_scores: List[float] = []

        for i, batch_start in enumerate(range(0, len(texts), batch_size)):
            batch_end = min(batch_start + batch_size, len(texts))
            print(f"[batch] Batch {i+1}: texts {batch_start+1}-{batch_end}")

            batch_texts = texts[batch_start:batch_end]

            # Keep empty texts as -1.0 (special pipeline value).
            non_empty_idx = [j for j, t in enumerate(batch_texts) if t and str(t).strip()]
            if not non_empty_idx:
                scores.extend([-1.0 for _ in batch_texts])
                continue

            non_empty_texts = [batch_texts[j] for j in non_empty_idx]

            inputs = self.tokenizer(
                non_empty_texts,
                max_length=4096,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            global_attention_mask = self._global_attention_mask(inputs["input_ids"])
            inputs["global_attention_mask"] = global_attention_mask

            with torch.no_grad():
                outputs = self.model(**inputs)

            raw_logits = outputs.logits.squeeze(-1).detach().float()
            normed = self._normalize(raw_logits).detach().float().cpu().tolist()
            if isinstance(normed, float):
                clipped = [float(normed)]
            else:
                clipped = [float(x) for x in normed]

            batch_scores: List[float] = [-1.0 for _ in batch_texts]
            for j, s in zip(non_empty_idx, clipped, strict=True):
                batch_scores[j] = s

            scores.extend(batch_scores)
            valid_scores.extend(clipped)

        if valid_scores:
            print(
                "[batch] Done. Scores: "
                f"min={min(valid_scores):.3f}, "
                f"max={max(valid_scores):.3f}, "
                f"mean={sum(valid_scores)/len(valid_scores):.3f}"
            )
        else:
            print("[batch] Done. No non-empty texts scored.")

        return scores

    @staticmethod
    def setup_model(output_dir: str):
        """Create a fresh base Longformer model with special tokens and save it."""
        print("[setup] Creating new Longformer model with special tokens...")
        print("[setup] Base model: allenai/longformer-base-4096")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"[setup] Added {num_added} special tokens. New vocab size: {len(tokenizer)}")

        model = LongformerForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels=1,
            problem_type="regression",
        )
        model.resize_token_embeddings(len(tokenizer))
        print(f"[setup] Resized embeddings to {len(tokenizer)}")

        tokenizer.save_pretrained(str(out))
        model.save_pretrained(str(out))
        print(f"[setup] Model and tokenizer saved to: {out}")


def _write_scorer_report(
    *,
    interactions_path: str,
    model_dir: str,
    results: List[dict],
    scores: List[float],
    out_path: str,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("SCORER REPORT")
    lines.append("=" * 80)
    lines.append(f"Interactions: {interactions_path}")
    lines.append(f"Model dir:    {model_dir}")
    lines.append(f"Pairs:        {len(results)}")
    lines.append("")

    for i, (r, s) in enumerate(zip(results, scores, strict=True), 1):
        char_a = r["char_a"]
        char_b = r["char_b"]
        raw_text = r["raw_text"]
        lf = r["longformer_input"]

        lines.append("-" * 80)
        lines.append(f"PAIR {i}: {char_a} ◄──► {char_b}")
        lines.append("-" * 80)
        lines.append(f"Score: {s:.4f}")
        lines.append(f"Raw text length: {len(raw_text)}")
        lines.append(f"Longformer input length: {len(lf)}")
        lines.append(f"[E1] count: {lf.count('[E1]')}   [E2] count: {lf.count('[E2]')}")
        lines.append("")
        lines.append("LONGFORMER INPUT (first 600 chars):")
        lines.append(lf[:600])
        lines.append("")

    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[report] Wrote scorer report to: {out}")


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup")
    p_setup.add_argument("output_dir", nargs="?", default="data/models/longformer_base")

    p_test = sub.add_parser("test")
    p_test.add_argument("model_dir")
    p_test.add_argument("text", nargs="?", default=None)
    p_test.add_argument("--norm", choices=["sigmoid", "clip"], default="sigmoid")

    p_ch = sub.add_parser("chapter")
    p_ch.add_argument("model_dir")
    p_ch.add_argument("interactions_txt")
    p_ch.add_argument("out_txt", nargs="?", default=None)
    p_ch.add_argument("--norm", choices=["sigmoid", "clip"], default="sigmoid")

    args = ap.parse_args()

    if args.cmd == "setup":
        LongformerScorer.setup_model(args.output_dir)
        raise SystemExit(0)

    if args.cmd == "test":
        test_text = (
            args.text
            if args.text is not None
            else (
                "Characters: [E1] Marley [/E1] and [E2] Scrooge [/E2].\n\n"
                "[E2] Scrooge [/E2] never painted out Old [E1] Marley [/E1]'s name."
            )
        )
        scorer = LongformerScorer(args.model_dir, norm=args.norm)
        score = scorer.score(test_text)
        print(f"\n[test] Final score: {score:.4f}")
        print("[test] (Note: baseline model is not fine-tuned)")
        raise SystemExit(0)

    if args.cmd == "chapter":
        model_dir = args.model_dir
        interactions_txt = args.interactions_txt
        out_txt = args.out_txt

        # Allow running as `python src/longformer_pipeline/scorer.py ...`
        # by ensuring `src/` is on sys.path.
        this_file = Path(__file__).resolve()
        src_dir = this_file.parents[1]
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from longformer_pipeline.input_builder import process_chapter_file  # noqa: PLC0415

        built = process_chapter_file(interactions_txt)
        texts = [r["longformer_input"] for r in built]

        scorer = LongformerScorer(model_dir, norm=args.norm)
        scores = scorer.score_batch(texts, batch_size=8)

        print("\n[chapter] Scores")
        for r, s in zip(built, scores, strict=True):
            print(f"  {r['char_a']} ◄──► {r['char_b']}: {s:.4f}")

        if out_txt:
            _write_scorer_report(
                interactions_path=interactions_txt,
                model_dir=model_dir,
                results=built,
                scores=scores,
                out_path=out_txt,
            )
        raise SystemExit(0)

    raise SystemExit(f"Unknown command: {args.cmd}")

