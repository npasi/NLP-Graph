"""SHAP interpretability: region-level Shapley + token top-10 removal delta.

Computes (per model, aggregated over a fixed test sample):
  - attr_previous_pct, attr_characters_pct, attr_narrative_pct
  - top10_removal_delta

Region SHAP uses shap.KernelExplainer on 3 binary features (Previous / Characters / Narrative).
Top-10 delta uses integrated gradients on input embeddings (truncated inputs for speed).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import LongformerForSequenceClassification, LongformerTokenizer

# ---------------------------------------------------------------------------
# Region parsing
# ---------------------------------------------------------------------------


def split_regions(text: str) -> Tuple[str, str, str]:
    """Split Longformer input into Previous / Characters / Narrative blocks."""
    t = str(text).strip()
    prev, rest = "", t
    if t.lower().startswith("previous:"):
        sep = t.find("\n\n")
        if sep >= 0:
            prev, rest = t[:sep].strip(), t[sep + 2 :]
        else:
            prev, rest = t, ""

    char_block, narr = "", rest
    if rest.lower().startswith("characters:"):
        sep = rest.find("\n\n")
        if sep >= 0:
            char_block, narr = rest[:sep].strip(), rest[sep + 2 :].strip()
        else:
            char_block, narr = rest.strip(), ""

    return prev, char_block, narr


def compose_regions(prev: str, char_block: str, narr: str, mask: Sequence[int]) -> str:
    """mask = (prev_on, char_on, narr_on) each 0/1."""
    parts: List[str] = []
    if mask[0] and prev:
        parts.append(prev)
    if mask[1] and char_block:
        parts.append(char_block)
    if mask[2] and narr:
        parts.append(narr)
    if not parts:
        return "Characters: [E1] ? [/E1] and [E2] ? [/E2]."
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class AffinityModel:
    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.tokenizer = LongformerTokenizer.from_pretrained(model_dir)
        self.model = LongformerForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.e_ids = [
            self.tokenizer.convert_tokens_to_ids(t)
            for t in ("[E1]", "[/E1]", "[E2]", "[/E2]")
        ]

    def _global_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        gam = torch.zeros_like(input_ids, dtype=torch.long)
        gam[:, 0] = 1
        for tid in self.e_ids:
            if tid is not None:
                gam = torch.where(input_ids == tid, torch.ones_like(gam), gam)
        return gam

    def score_text(self, text: str) -> float:
        if not str(text).strip():
            return float("nan")
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["global_attention_mask"] = self._global_attention_mask(inputs["input_ids"])
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        return float(torch.sigmoid(logits).item())

    def score_batch_masks(
        self,
        prev: str,
        char_block: str,
        narr: str,
        masks: np.ndarray,
    ) -> np.ndarray:
        """masks shape (n, 3) binary."""
        out = []
        for row in masks:
            text = compose_regions(prev, char_block, narr, row)
            out.append(self.score_text(text))
        return np.asarray(out, dtype=np.float64)

    def token_attribution(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Input x gradient attribution per token (non-pad positions)."""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        gam = self._global_attention_mask(input_ids)
        active = int(attention_mask[0].sum().item())

        emb_layer = self.model.get_input_embeddings()
        emb = emb_layer(input_ids)
        emb.retain_grad()
        outputs = self.model(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            global_attention_mask=gam,
        )
        score = torch.sigmoid(outputs.logits.squeeze())
        score.backward()
        grad = emb.grad[0]
        attrs = (grad * emb[0]).norm(dim=-1).abs().detach().cpu().numpy()
        toks = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return attrs[:active], toks[:active]


# ---------------------------------------------------------------------------
# Metrics per example
# ---------------------------------------------------------------------------


def exact_shapley_3(
    model: AffinityModel,
    prev: str,
    char_block: str,
    narr: str,
) -> np.ndarray:
    """Exact Shapley values for 3 binary region features (8 coalitions)."""

    def f(mask: Tuple[int, int, int]) -> float:
        return float(model.score_batch_masks(prev, char_block, narr, np.array([mask]))[0])

    # feature order: 0=Previous, 1=Characters, 2=Narrative
    n = 3
    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                s = [0, 0, 0]
                for j in subset:
                    s[j] = 1
                s_with_i = list(s)
                s_with_i[i] = 1
                s = tuple(s)
                s_with_i = tuple(s_with_i)
                weight = (
                    math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                )
                phi[i] += weight * (f(s_with_i) - f(s))
    return phi


def region_shap_values(
    model: AffinityModel,
    prev: str,
    char_block: str,
    narr: str,
) -> np.ndarray:
    """Exact Shapley for 3 region features (Previous / Characters / Narrative)."""
    return exact_shapley_3(model, prev, char_block, narr)


def shap_to_pct(phi: np.ndarray) -> Tuple[float, float, float]:
    denom = float(np.sum(np.abs(phi)))
    if denom <= 0:
        return 0.0, 0.0, 0.0
    return tuple((float(abs(p)) / denom * 100.0 for p in phi))


def top10_removal_delta(model: AffinityModel, text: str) -> float:
    y0 = model.score_text(text)
    attrs, toks = model.token_attribution(text)
    if len(attrs) == 0:
        return 0.0
    k = min(10, len(attrs))
    top_idx = np.argsort(-attrs)[:k]

    enc = model.tokenizer(
        text,
        max_length=model.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].clone()
    pad_id = model.tokenizer.pad_token_id or 0
    for i in top_idx:
        input_ids[0, int(i)] = pad_id

    attention_mask = enc["attention_mask"]
    gam = model._global_attention_mask(input_ids.to(model.device))
    with torch.no_grad():
        logits = model.model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            global_attention_mask=gam,
        ).logits.squeeze(-1)
    y1 = float(torch.sigmoid(logits).item())
    return abs(y0 - y1)


@dataclass
class ExampleMetrics:
    attr_prev: float
    attr_char: float
    attr_narr: float
    top10_delta: float


def eval_example(model: AffinityModel, text: str) -> ExampleMetrics:
    prev, char_b, narr = split_regions(text)
    phi = region_shap_values(model, prev, char_b, narr)
    p_prev, p_char, p_narr = shap_to_pct(phi)
    d10 = top10_removal_delta(model, text)
    return ExampleMetrics(p_prev, p_char, p_narr, d10)


def load_test_sample(jsonl_path: Path, book_ids: Sequence[str], n_total: int, seed: int) -> List[dict]:
    rows: List[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if str(ex.get("book_id")) in book_ids:
                rows.append(ex)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n_total]


def aggregate(metrics: List[ExampleMetrics]) -> Dict[str, float]:
    if not metrics:
        return {}
    return {
        "n": len(metrics),
        "attr_previous_pct": float(np.mean([m.attr_prev for m in metrics])),
        "attr_characters_pct": float(np.mean([m.attr_char for m in metrics])),
        "attr_narrative_pct": float(np.mean([m.attr_narr for m in metrics])),
        "top10_removal_delta": float(np.mean([m.top10_delta for m in metrics])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP region attribution + top-10 removal delta")
    ap.add_argument("--test-jsonl", default="data/training/test.jsonl")
    ap.add_argument("--book-ids", default="95,99,110")
    ap.add_argument("--n-samples", type=int, default=18, help="Examples per model (same IDs for both)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--out-json",
        default="data/share_with_friend_clean/shap_pretrained_vs_1ep.json",
    )
    ap.add_argument(
        "--pretrained-dir",
        default="data/models/longformer_base",
    )
    ap.add_argument(
        "--finetuned-dir",
        default="data/models/longformer_prevprefix_full_1ep/best",
    )
    args = ap.parse_args()

    book_ids = [b.strip() for b in str(args.book_ids).split(",") if b.strip()]
    sample = load_test_sample(Path(args.test_jsonl), book_ids, int(args.n_samples), int(args.seed))
    if not sample:
        raise SystemExit("No test examples found.")

    configs = [
        ("pretrained_base", str(args.pretrained_dir)),
        ("old_1ep_MSE", str(args.finetuned_dir)),
    ]

    out: Dict = {
        "params": {
            "test_jsonl": args.test_jsonl,
            "book_ids": book_ids,
            "n_samples": len(sample),
            "seed": args.seed,
            "max_length": args.max_length,
            "region_shap": "exact Shapley values on 3 binary region masks (Previous/Characters/Narrative)",
            "top10_delta": "input x gradient attribution, mask top-10 tokens",
        },
        "sample_ids": [
            {
                "book_id": ex["book_id"],
                "chapter": ex.get("chapter"),
                "char_a": ex.get("char_a"),
                "char_b": ex.get("char_b"),
            }
            for ex in sample
        ],
        "models": {},
    }

    texts = [ex["text"] for ex in sample]

    for name, model_dir in configs:
        print(f"[shap] Loading {name} from {model_dir}")
        am = AffinityModel(model_dir, max_length=int(args.max_length))
        per_ex: List[ExampleMetrics] = []
        for i, text in enumerate(texts):
            print(f"  [{name}] example {i+1}/{len(texts)}")
            per_ex.append(eval_example(am, text))
        out["models"][name] = aggregate(per_ex)
        del am
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[shap] Wrote {out_path}")
    for mk, agg in out["models"].items():
        print(mk, agg)


if __name__ == "__main__":
    main()
