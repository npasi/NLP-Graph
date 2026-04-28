## Repo overview

This repository contains a research pipeline for extracting **character-centric graphs** from long-form English narrative text, built around **BookNLP**.

- **Primary goals**
  - Run BookNLP on a full book (or per chapter) and cache its outputs
  - Derive character inventories and per-chapter relation graphs
  - Provide inspectable intermediate artefacts (TSVs/JSON/HTML)

- **Key output philosophy**
  - **Commit code, not caches**: heavy/replicable outputs (BookNLP caches, generated graphs, reports) are excluded via `.gitignore`.
  - Keep raw book text only if small; otherwise treat as external input.

## Top-level layout

- `src/`: main library + CLI entrypoints
- `notebooks/`: exploratory analysis
- `data/`: local cache (ignored in git for replicability)
- `requirements.txt`: Python dependencies

## Main modules and what they do

### `src/graph_builder.py`

Implements `BookNLPGraphBuilder`, a wrapper that:

- loads BookNLP lazily (so weights are loaded once)
- applies Windows + Transformers compatibility patches (`_patch_booknlp_for_windows`)
- runs BookNLP with caching (`_run_booknlp`)
- provides helpers to select character chains and build graph features

**Key functions/classes**
- **`BookNLPGraphBuilder.__init__`**: configure cache roots, model size, filtering mode, scoring.
- **`BookNLPGraphBuilder._ensure_booknlp`**: initialise BookNLP with `"entity,quote,supersense,event,coref"`.
- **`BookNLPGraphBuilder._run_booknlp(text, run_id)`**: run (or reuse cached) BookNLP and return parsed outputs.
- **`_patch_booknlp_for_windows()`**: monkey-patches upstream BookNLP for Windows path handling + `transformers` strictness.

### `src/run_booknlp_only.py`

Minimal “BookNLP only” runner for a single full-book `.txt`, without chapter splitting and without graph construction.

**Key functions**
- **`run_booknlp_only(input_txt, output_root, run_id, model_size, top_k)`**: executes BookNLP and writes a `*.summary.json`.
- **CLI**: `python -X utf8 -m src.run_booknlp_only --input ... --run-id ...`

### `src/booknlp_listings_html.py`

Generates a lightweight HTML “listing” page by **reading** BookNLP native outputs:

- `<run_id>.book` for character clusters and **event/role extraction** lists:
  - `agent`, `patient`, `poss`, `mod`
- `<run_id>.tokens` for token-level event tags (`event` column)
- optional derived event triplets preview (if present)

**Key functions**
- **`generate_listings_html(run_dir, run_id)`** → writes `<run_id>.book.listings.html`

### `src/booknlp_event_triplets.py`

Extracts **(agent, predicate, patient)** triplets suitable for graph construction using:

- dependency parse from `<run_id>.tokens`
- PER mention spans + COREF ids from `<run_id>.entities`
- canonical cluster names from `<run_id>.book`

**Key functions**
- **`extract_event_triplets(run_dir, run_id, only_person_person)`**: returns a DataFrame + JSONL objects
- **`write_outputs(...)`**: writes `<run_id>.event_triplets.tsv` and `<run_id>.event_triplets.jsonl`
- **CLI**: `python -m src.booknlp_event_triplets --run-dir ... --run-id ...`

### `src/pipeline_steps/*` (4-step pipeline)

This directory contains the more structured pipeline:

- **`step1_characters.py`**: run BookNLP book-level, create `characters.json` from chains.
- **`step2_alias_resolution.py`**: merge aliases, recompute coref confidences, optional bridging and cleanup.
- **`step3_relations.py`**: derive per-chapter edges from sentence proximity / relation scoring.
- **`step4_render.py`**: render interactive HTML visualisations.

Shared utilities:
- **`_common.py`**: path conventions, chapter loading, concatenation, book-level BookNLP wrapper.
- **`_bridge.py`**: deterministic heuristic NOM→PROP bridging.
- **`_llm_bridge.py`**: optional DeepSeek LLM-based bridge (uses env `DEEPSEEK_API_KEY`).

## How to run (common)

- BookNLP-only full book:
  - `python -X utf8 -m src.run_booknlp_only --input data/raw/46.txt --run-id christmas_carol_full --model-size big`

- Generate character/predicate listing HTML (from BookNLP outputs):
  - `python -m src.booknlp_listings_html --run-dir data/booknlp_only_output/christmas_carol_full --run-id christmas_carol_full`

- Extract event triplets:
  - `python -m src.booknlp_event_triplets --run-dir data/booknlp_only_output/christmas_carol_full --run-id christmas_carol_full`

## Notes on “Event/role extraction”

BookNLP stores the role/event extraction lists **inside** the `.book` JSON for each character cluster:

- `agent`: verbs where the character is syntactic subject
- `patient`: verbs where the character is object / passive subject
- `poss`: possessed nouns (from `poss` dependency)
- `mod`: modifiers/attributes (from copular patterns, etc.)

The native BookNLP `*.book.html` does not render these lists by default; this repo keeps them as native JSON and provides derived listings/triplets without changing BookNLP outputs.

