"""Step 4: fine-tune Longformer for pairwise sentiment regression.

Labels are in [0, 1].

Supported losses (--loss flag):
  mse       MSE on sigmoid(logits)  [original, collapses to 0.5]
  pearson   (1 - Pearson corr) + lambda * WeightedMSE  [recommended]
            WeightedMSE up-weights examples far from 0.5 so the model
            cannot minimise loss by always predicting the mean.

The Pearson loss directly penalises "flat" predictions: predicting 0.5
everywhere gives Pearson=0 and loss=1, the worst possible value.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import (
    LongformerForSequenceClassification,
    LongformerTokenizer,
    Trainer,
    TrainingArguments,
)


class SentimentDataset(Dataset):
    """JSONL rows with ``text`` and ``score``; tokenization + global attention."""

    def __init__(self, jsonl_path: str, tokenizer: LongformerTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts: List[str] = []
        self.scores: List[float] = []

        e1 = tokenizer.convert_tokens_to_ids("[E1]")
        e1e = tokenizer.convert_tokens_to_ids("[/E1]")
        e2 = tokenizer.convert_tokens_to_ids("[E2]")
        e2e = tokenizer.convert_tokens_to_ids("[/E2]")
        self.special_token_ids = [e1, e1e, e2, e2e]

        print(f"[dataset] Loading {jsonl_path}...")
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(jsonl_path)
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.texts.append(row["text"])
                self.scores.append(float(row["score"]))

        print(f"[dataset] Loaded {len(self.texts)} examples")
        if self.scores:
            print(
                "[dataset] Score stats: "
                f"min={min(self.scores):.3f}, max={max(self.scores):.3f}, "
                f"mean={sum(self.scores) / len(self.scores):.3f}"
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        global_attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        global_attention_mask[0] = 1
        for tok_id in self.special_token_ids:
            global_attention_mask[input_ids == tok_id] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "labels": torch.tensor(self.scores[idx], dtype=torch.float),
        }


class MSESigmoidTrainer(Trainer):
    """Regression head outputs logits; we apply sigmoid then optimize MSE vs
    targets in [0,1]. Inference uses the same sigmoid, so train/eval match.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        preds = torch.sigmoid(logits)
        loss = F.mse_loss(preds, labels.float())
        return (loss, outputs) if return_outputs else loss


class PearsonWeightedTrainer(Trainer):
    """Loss = (1 - Pearson(pred, gt)) + lambda * WeightedMSE(pred, gt).

    WeightedMSE uses per-example weights = 1 + alpha * |gt - 0.5| so that
    pairs with extreme affinity scores (clearly positive/negative) pull the
    model harder than neutral pairs. This prevents the model from collapsing
    to predicting 0.5 for every example.

    Pearson loss ensures the *shape* (rank ordering) of predictions matches
    the GT even when the absolute scale is off.
    """

    lam: float = 0.3    # weight of WeightedMSE term relative to Pearson term
    alpha: float = 2.0  # strength of example weighting by distance from 0.5

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        preds = torch.sigmoid(logits)

        # --- Pearson loss (1 - corr): forces rank ordering, penalises flat ---
        n = preds.shape[0]
        if n < 2 or preds.std() < 1e-8 or labels.std() < 1e-8:
            # fallback to MSE when batch is too small or degenerate
            pearson_loss = F.mse_loss(preds, labels)
        else:
            vp = preds - preds.mean()
            vl = labels - labels.mean()
            corr = (vp * vl).sum() / (
                torch.sqrt((vp ** 2).sum() * (vl ** 2).sum()) + 1e-8
            )
            pearson_loss = 1.0 - corr

        # --- Weighted MSE: up-weight examples far from 0.5 ---
        weights = 1.0 + self.alpha * torch.abs(labels - 0.5)
        weighted_mse = (weights * (preds - labels) ** 2).mean()

        loss = pearson_loss + self.lam * weighted_mse
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """MSE/MAE/Pearson on ``sigmoid(logits)`` vs labels (aligned with scorer)."""
    predictions, labels = eval_pred
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    lab = np.asarray(labels, dtype=np.float64).reshape(-1)
    pred_prob = 1.0 / (1.0 + np.exp(-np.clip(pred, -50, 50)))

    mse = float(mean_squared_error(lab, pred_prob))
    mae = float(mean_absolute_error(lab, pred_prob))
    if np.std(lab) < 1e-9 or np.std(pred_prob) < 1e-9:
        pearson = 0.0
    else:
        pearson = float(pearsonr(lab, pred_prob)[0])

    print(f"[metrics] MSE={mse:.4f}, MAE={mae:.4f}, Pearson={pearson:.4f}")

    return {"mse": mse, "mae": mae, "pearson": pearson}


def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 4096,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    seed: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    loss: str = "pearson",
) -> None:
    print("[train] Loading base model: allenai/longformer-base-4096")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"[train] Added {num_added} special tokens")

    model = LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096",
        num_labels=1,
        problem_type="regression",
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"[train] Model ready. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = SentimentDataset(train_path, tokenizer, max_length)
    val_dataset = SentimentDataset(val_path, tokenizer, max_length)
    print(f"[train] Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty")
    if len(val_dataset) == 0:
        raise ValueError("Val dataset is empty (need at least one val example for eval)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # transformers 4.x: evaluation_strategy; 5.x+: eval_strategy
    import inspect

    _sig = inspect.signature(TrainingArguments.__init__).parameters
    _eval_kw = (
        {"evaluation_strategy": "epoch"}
        if "evaluation_strategy" in _sig
        else {"eval_strategy": "epoch"}
    )

    # Loss selection: pearson (default, recommended) or mse (original)
    if loss == "pearson":
        trainer_cls = PearsonWeightedTrainer
        metric_for_best = "pearson"
        higher_better = True
    else:
        trainer_cls = MSESigmoidTrainer
        metric_for_best = "mse"
        higher_better = False

    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        **_eval_kw,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=higher_better,
        fp16=fp16,
        gradient_checkpointing=True,
        logging_steps=10,
        seed=seed,
        report_to="none",
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n{'=' * 60}")
    print("[train] STARTING TRAINING")
    loss_desc = "(1-Pearson) + 0.3*WeightedMSE" if loss == "pearson" else "MSE on sigmoid(logits)"
    print(f"[train] Loss: {loss_desc}")
    print(f"[train] Epochs: {epochs}")
    print(
        f"[train] Batch size: {batch_size} x {gradient_accumulation_steps} = "
        f"{batch_size * gradient_accumulation_steps} effective"
    )
    print(f"[train] Learning rate: {learning_rate}")
    print(f"[train] Max length: {max_length}")
    print(f"[train] FP16: {fp16}")
    if resume_from_checkpoint:
        print(f"[train] Resuming from checkpoint: {resume_from_checkpoint}")
    print(f"{'=' * 60}\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n[train] Best model saved to: {best_dir}")

    final_metrics = trainer.evaluate()
    print(f"\n{'=' * 60}")
    print("[train] FINAL METRICS")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"[train]   {k}: {v:.4f}")
        else:
            print(f"[train]   {k}: {v}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Longformer (MSE on sigmoid)")
    parser.add_argument("--train-path", default="data/training/train.jsonl")
    parser.add_argument("--val-path", default="data/training/val.jsonl")
    parser.add_argument("--output-dir", default="data/models/longformer_finetuned")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Effective batch = batch-size * grad-accum (default 4*4=16)",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 (e.g. on CPU)")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from a checkpoint dir (e.g. data/models/.../checkpoint-121)",
    )
    parser.add_argument(
        "--loss",
        choices=["pearson", "mse"],
        default="pearson",
        help="pearson = (1-Pearson)+0.3*WeightedMSE [default, fixes regression-to-mean]; mse = original MSE",
    )
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=not args.no_fp16,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        loss=args.loss,
    )
