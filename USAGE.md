# NLP-Graph – Usage Guide

This repository provides an end-to-end pipeline to transform a raw book (.txt)
into a structured character graph using BookNLP and a custom Character Identity Layer.

---

## Quick Start (Legacy mode)

Run the full pipeline with a single command:

```bash
python -m src.run_full_pipeline --input data/raw/book.txt --book-id book
```

Outputs land in the historic `data/` layout:

```
data/books/<book_id>/chapters/
data/booknlp_chapter_output/<book_id>/
data/reports/<book_id>/
data/graphs/chapters/
data/ml/<book_id>/
data/evaluation_logs/
```

This mode is fully preserved and requires no additional arguments.

---

## Recommended: Run-based workflow

Use `--output-root` and `--run-id` to isolate each experiment in its own
directory. This prevents outputs from different runs from overwriting each
other and makes reproducibility easy.

### 1. Choose a run ID

Pick a descriptive, unique name:

```bash
RUN_ID=run_2026_05_07_eval_v1
```

### 2. Run the pipeline for one book

```bash
python -m src.run_full_pipeline \
  --input data/raw/9_The_Great_Gatsby_F._Scott_Fitzgerald.txt \
  --book-id gatsby \
  --output-root outputs/runs \
  --run-id "$RUN_ID"
```

Outputs are written to:

```
outputs/runs/$RUN_ID/
  books/gatsby/chapters/
  booknlp_chapter_output/gatsby/
  reports/gatsby/
  graphs/chapters/gatsby_*.pkl
  ml/gatsby/
  logs/
  manifest.json
```

The `manifest.json` records the run ID, git branch/commit, input path,
and the status of each processed book.

### 3. Generate evaluation summary

```bash
python scripts/make_evaluation_summary.py --run-dir outputs/runs/$RUN_ID
```

Writes `outputs/runs/$RUN_ID/evaluation_summary.csv` with one row per book.

### 4. Archive old legacy outputs (optional)

Preview what would be moved (safe, prints only):

```bash
bash scripts/archive_old_outputs.sh --dry-run
```

Actually move legacy outputs to `archive/old_outputs/<timestamp>/`:

```bash
bash scripts/archive_old_outputs.sh
```

Nothing is deleted. To undo, move directories back from the archive.

---

## Directory layout

| Directory | Role | Committed? |
|-----------|------|------------|
| `data/raw/` | Raw input `.txt` files | Yes |
| `data/aliases/` | Per-book alias JSON config | Yes |
| `data/booknlp_models/` | Downloaded model weights | Yes |
| `data/books/` … `data/ml/` | Legacy generated outputs | No |
| `outputs/runs/<run_id>/` | Run-based generated outputs | No |
| `cache/` | Cleaned text cache | No |
| `archive/` | Archived old outputs | No |

---

## Individual pipeline steps

Each step can also be run manually. All scripts preserve backward-compatible
defaults; new path arguments are optional overrides.

| Step | Script | Key optional args |
|------|--------|-------------------|
| Chapter splitting | `src/step1_split_only.py` | `--books-root` |
| BookNLP per chapter | `run_booknlp_per_chapter.py` | `--chapters-dir`, `--output-root` |
| Character identity | `src/step2b_character_identity.py` | `--booknlp-root` |
| Normalized predicates | `src/step3b_normalized_predicates.py` | `--booknlp-root` |
| Normalized evidence | `src/step4b_normalized_evidence.py` | `--booknlp-root`, `--chapters-root` |
| Build graphs | `src/build_normalized_graphs.py` | `--booknlp-root`, `--graphs-dir` |
| Score interactions | `src/score_interactions.py` | `--booknlp-root`, `--output-dir` |
| Filtered graphs | `src/build_filtered_graphs.py` | `--booknlp-root`, `--graphs-root` |
| BERT dataset | `src/export_bert_dataset.py` | `--booknlp-root`, `--output-dir` |
| Quality report | `src/build_quality_report.py` | `--booknlp-root`, `--graphs-root`, `--output-dir` |
| Alias suggestions | `src/suggest_aliases.py` | `--booknlp-root`, `--output-dir` |
| Tables export | `src/export_tables.py` | `--booknlp-root`, `--reports-root` |
| Interaction reports | `src/export_interaction_reports.py` | `--booknlp-root`, `--output-dir` |

---

## Notes

- `data/raw`, `data/aliases`, and `data/booknlp_models` are **stable inputs**
  and are never touched by cleanup or archive scripts.
- Generated outputs (`data/books`, `data/reports`, etc.) are gitignored.
  Use run mode to keep experiments separated.
- Manual cleanup should use `scripts/archive_old_outputs.sh`, not `rm -rf`.
- The evaluation summary script (`scripts/make_evaluation_summary.py`) auto-
  discovers book IDs from the run directory in run mode.
