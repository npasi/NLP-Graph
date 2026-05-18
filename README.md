# Book Graph Pipeline

End-to-end pipeline that turns raw Project Gutenberg novels into chapter-level
character-interaction graphs (BookNLP + PMI + VADER) and fine-tunes a
Longformer regression model to predict pairwise affinity scores in `[0, 1]`.

Course: Language Technology — Bocconi University, 2025/2026.

---

## 1. Installation

Use **Python 3.11** (required by BookNLP + `tokenizers==0.13.3`).

### Linux / macOS

```bash
git clone https://github.com/npasi/NLP-Graph.git
cd NLP-Graph
git checkout BRANCH-CON-I-RAGAZZI

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Windows (PowerShell)

```powershell
git clone https://github.com/npasi/NLP-Graph.git
cd NLP-Graph
git checkout BRANCH-CON-I-RAGAZZI

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### GPU (optional, recommended for Longformer training)

Install a CUDA-matched PyTorch build before `requirements.txt`:

```bash
# CUDA 12.1 example
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Caches (recommended)

Keep HuggingFace and BookNLP model weights local to the repo:

```bash
export HF_HOME="$PWD/data/hf_cache"
export TRANSFORMERS_CACHE="$PWD/data/hf_cache"
export XDG_CACHE_HOME="$PWD/data/hf_cache"
```

PowerShell equivalent:

```powershell
$env:HF_HOME       = "$PWD\data\hf_cache"
$env:TRANSFORMERS_CACHE = "$PWD\data\hf_cache"
$env:XDG_CACHE_HOME = "$PWD\data\hf_cache"
```

### Verify the install

```bash
python -c "import torch, transformers, booknlp, spacy, networkx; print('ok')"
```

---

## 2. Repository layout

```
.
├── data/                      # folder skeleton only (all artifacts are .gitignored)
│   ├── archive/               # legacy / bulky duplicates
│   ├── dataset/               # ground_truth/, texts/, interactions/
│   ├── evaluation/            # all_results/, compare/, diagnostics/
│   ├── models/                # BookNLP weights + fine-tuned Longformer
│   ├── output/                # BookNLP chapter outputs
│   ├── predictions/           # per-book affinity CSVs (per model)
│   ├── share_with_friend_clean/  # graphs + metrics export
│   ├── splits/                # train/val/test split JSON
│   └── training/              # training/val/test JSONL
├── results_and_plots/         # paper-ready figures and tables (committed)
├── src/                       # graph-building pipeline (BookNLP + lexicon)
├── src/longformer_pipeline/   # Longformer fine-tuning + inference + plots
├── requirements.txt
└── README.md
```

> **Note on `data/`**: only the folder skeleton (via `.gitkeep`) is tracked.
> Datasets, model weights, predictions, BookNLP outputs and evaluation
> artifacts are not committed — regenerate them with the scripts below.

---

## 3. End-to-end pipeline

Place a raw book at `data/raw/<book_id>.txt` (Project Gutenberg works well).

### 3.1 Chapter splitting

```bash
python src/step1_split_only.py --input data/raw/46.txt --book-id 46 --log-level INFO
# -> data/output/chapters/46/chapters/chapter_000.txt ...
```

### 3.2 BookNLP per chapter (entity, quote, supersense, event, coref)

```bash
python src/run_booknlp_per_chapter.py --book-id 46 --model-size big \
    --pipeline entity,quote,supersense,event,coref
# -> data/output/booknlp/46/booknlp_46_chapter_NNNN/*
```

### 3.3 Derived artifacts (no new NLP)

```bash
python src/step2_character_list_per_chapter.py       --book-id 46
python src/step3_predicates_between_characters.py    --book-id 46
python src/step4_sentences_by_character_pair.py      --book-id 46
```

### 3.4 Chapter graphs (PMI + VADER lexicon)

```bash
python -m src.run_pipeline_all_books --book 46
# -> data/graphs/chapters/<slug>.pkl + metadata
```

---

## 4. Longformer fine-tuning

All scripts live under `src/longformer_pipeline/` (set `PYTHONPATH=src`).

### 4.1 Build training JSONL from GT + chapter interactions

```bash
python -m longformer_pipeline.build_dataset \
    --gt-dir data/dataset/ground_truth \
    --interactions-dir data/dataset/interactions \
    --output-dir data/training \
    --texts-dir data/dataset/texts
```

Produces `data/training/{train,val,test}.jsonl`.

### 4.2 Train (MSE or Pearson + Weighted-MSE composite loss)

```bash
python -m longformer_pipeline.train \
    --train-path data/training/train.jsonl \
    --val-path   data/training/val.jsonl \
    --output-dir data/models/longformer_prevprefix_full_1ep \
    --epochs 1 --batch-size 4 --learning-rate 2e-5
```

### 4.3 Book-level inference (carry-forward for absent pairs)

```bash
python -m longformer_pipeline.inference \
    --book-id 95 \
    --interactions-dir data/dataset/interactions \
    --model-dir data/models/longformer_prevprefix_full_1ep/best \
    --output-csv data/predictions/prevprefix_test_1ep/95_predicted.csv
```

### 4.4 Evaluation, comparison plots, SHAP

```bash
python -m longformer_pipeline.export_friend_pack         # metrics + graph PNGs
python -m longformer_pipeline.plot_two_chapters_batch    # max-change transition
python -m longformer_pipeline.plot_all_chapters_batch    # per-chapter graphs
python -m longformer_pipeline.explain_shap               # region + token SHAP
```

---

## 5. Results and figures

All paper-ready figures and tables are committed under `results_and_plots/`:

| folder | content |
|---|---|
| `Sample_output_graph/` | predicted graphs for test books (95, 99, 110) |
| `Sample_Improvement/`  | full graph set for book 110 (all models) |
| `training_testing_results/` | train/val/test comparison metrics + plots |
| `sample_book_improvements/` | val books 86, 91 + aggregate (pretrained vs FT) |
| `histplots/` | GT vs prediction value and Δ distributions |

---

## 6. BookNLP compatibility patches

`src/graph_builder.py` monkey-patches two known issues at import time:

1. **Windows paths.** `entity_tagger.py`, `litbank_coref.py`, `bert_qa.py`
   use `model_file.split("/")[-1]` which breaks on Windows back-slashes;
   replaced with `os.path.basename`.
2. **New `transformers` releases.** BookNLP checkpoints still carry the
   stale `bert.embeddings.position_ids` buffer; loaded with `strict=False`
   so the key is silently dropped.

Both fixes are no-ops on Linux/macOS and on older `transformers`.

---

## 7. License & attribution

Academic project. BookNLP © David Bamman et al.; LitBank, VADER, Longformer
under their respective licenses.
