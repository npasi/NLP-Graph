# Book Graph Pipeline

A research pipeline for extracting **character graphs** from literary works.

For each selected book the pipeline:

1. Clones [LitBank](https://github.com/dbamman/litbank) and parses its
   gold annotations (entities + coreference chains).
2. Downloads the **full** book from Project Gutenberg using the ID in the
   LitBank filename.
3. Splits the full book into chapters.
4. Runs **BookNLP** on every chapter to extract characters / quotes.
5. Builds one **NetworkX** graph per chapter (100-token co-occurrence
   window, sentiment per edge).
6. Builds a **gold mini-graph** from the LitBank annotations (only the
   ~2 000 annotated words) as a reference.
7. Saves chapter graphs, gold graph and per-chapter metadata.

---

## Project layout (this repo)

```
book_graph_pipeline/
├── data/
│   ├── raw/                      # put your input full-book .txt here (optional)
│   ├── booknlp_only_output/      # BookNLP outputs for full-book runs (local cache)
│   ├── booknlp_output/           # other BookNLP caches (local cache)
│   └── books/                    # optional per-book assets (kept local)
├── script validati/              # validated scripts (kept in repo)
├── src/                          # code (runners + extractors)
├── requirements.txt
└── README.md
```

---

## Installation

Python 3.10+. Git must be on the PATH (needed to clone LitBank).

```powershell
cd book_graph_pipeline
python -m venv .venv
.venv\Scripts\activate              # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

BookNLP pulls ~1–2 GB of model weights the first time it runs.

---

## Usage

> **Windows users: launch Python with `-X utf8`.** BookNLP opens its own
> data files without specifying an encoding, which triggers
> ``UnicodeDecodeError`` on Windows (cp1252) for books that contain
> smart quotes or em dashes. ``-X utf8`` makes ``open()`` default to
> UTF-8 and sidesteps the problem entirely. The pipeline warns if the
> flag is missing.

### Run BookNLP on a full book (classic outputs)

Put a text file in `data/raw/` (or pass any path), then run:

```powershell
python -X utf8 -m src.run_booknlp_only --input data/raw/<your_book>.txt --run-id my_run --model-size big
```

This writes native BookNLP artefacts under:

- `data/booknlp_only_output/my_run/`
  - `my_run.tokens` (includes token-level `event` tag)
  - `my_run.entities`
  - `my_run.quotes`
  - `my_run.supersense`
  - `my_run.book` (includes `agent/patient/poss/mod` lists per character)
  - `my_run.book.html` (native BookNLP HTML)

### Generate listings HTML (read-only)

```powershell
python -m src.booknlp_listings_html --run-dir data/booknlp_only_output/my_run --run-id my_run
```

Produces:

- `data/booknlp_only_output/my_run/my_run.book.listings.html`

### Extract (agent, predicate, patient) triplets

```powershell
python -m src.booknlp_event_triplets --run-dir data/booknlp_only_output/my_run --run-id my_run
```

Produces:

- `my_run.event_triplets.tsv`
- `my_run.event_triplets.jsonl`

### About data/ in git

This repo keeps the same folder structure, but **does not commit** heavy/replicable caches.
Empty `data/*` folders are tracked via `.gitkeep`.

### Legacy pipeline notes

The README below contains older notes from a larger pipeline version. The current repo focuses on the BookNLP runners and graph-ready outputs above.

### End-to-end on every LitBank book

```powershell
python -X utf8 -m src.pipeline
```

The pipeline is driven directly by the LitBank checkout: it processes
**every** book shipped in `data/litbank/entities/tsv/` (100 books). There
is no hand-curated list.

Each successful book produces:

- `data/graphs/chapters/<slug>.pkl` — `list[networkx.Graph]`, one per chapter.
- `data/graphs/chapters/<slug>_metadata.json` — chapter stats.
- `data/graphs/gold/<slug>_gold.pkl` — single `networkx.Graph` from
  LitBank gold annotations.

Books that fail (download error, BookNLP crash) are logged and skipped —
the pipeline moves on.

### Smaller / selective runs

```powershell
# only one Gutenberg id (must be present in LitBank)
python -X utf8 -m src.pipeline --book 1342

# only the first N LitBank books (handy for a smoke test)
python -X utf8 -m src.pipeline --limit 3

# smaller / faster BookNLP model
python -X utf8 -m src.pipeline --model-size small

# mix them
python -X utf8 -m src.pipeline --limit 3 --model-size small
```

### Programmatic use

```python
from src.pipeline import process_book, process_catalog

# one book by LitBank slug
result = process_book("1342_pride_and_prejudice")
print(result["chapter_graphs"][0].nodes(data=True))

# several books by Gutenberg id
all_results = process_catalog(gutenberg_ids=[1342, 84], model_size="small")
```

---

## Graph schemas

### Chapter graphs (`chapters/<slug>.pkl`)

Undirected `networkx.Graph`, one per chapter.

- **Nodes** — one per BookNLP coref cluster classified as PER:
  - `name` — canonical name (most frequent proper mention).
  - `aliases` — all surface forms seen (proper + common).
  - `mention_count` — mention count.
  - `gender` — BookNLP-inferred gender (`unknown` when absent).
- **Edges** — two characters within a 100-token window at least once:
  - `weight` — number of co-occurrence windows.
  - `sentiment` — `"positive" / "negative" / "neutral"` (VADER averaged
    over the co-occurrence windows; BookNLP does not emit sentiment).
  - `sentiment_score` — raw VADER compound score in `[-1, 1]`.
  - `description` — short snippet from the most charged window.
- **Graph-level** — `chapter_id`, `chapter_title`, `num_tokens`,
  `short_chapter` (`True` when `num_tokens < 200`), `book_title`,
  `gutenberg_id`.

### Gold graphs (`gold/<slug>_gold.pkl`)

Single `networkx.Graph` built only from the ~2 000 annotated words in
LitBank (PER entities + coref chains).

- **Nodes** — one per LitBank coref chain that has at least one PER
  mention:
  - `name` — longest mention text in the chain.
  - `aliases` — every mention surface form in the chain.
  - `mention_count` — number of mentions of the chain.
- **Edges** — two chains appearing in the same sentence at least once:
  - `weight` — number of co-occurring sentences.
- **Graph-level** — `source` = `"litbank_gold"`, `gutenberg_id`,
  `book_title`, `num_tokens`.

---

## Notes & caveats

- LitBank covers ~2 000 words per book — that's the first few pages. The
  gold graph is therefore tiny and is not directly comparable to a
  full-chapter graph; it's a reference for evaluating BookNLP on the
  same passage.
- BookNLP is run **per chapter** (simpler, restartable). Global
  coreference across chapters is not attempted in this phase.
- Downloads, LitBank clones and BookNLP outputs are all cached — reruns
  are cheap.
- English only (all preset books qualify).

## BookNLP compatibility patches

BookNLP 1.0.7.1 ships with two upstream issues that the pipeline
monkey-patches transparently at load time (see
``_patch_booknlp_for_windows`` in ``src/graph_builder.py``):

1. **Windows paths.** ``entity_tagger.py``, ``litbank_coref.py`` and
   ``bert_qa.py`` use ``model_file.split("/")[-1]`` to derive a model
   basename. On Windows the backslashes in the path leak into the
   HuggingFace repo id and the call is rejected. The patch swaps the
   logic for ``os.path.basename``.
2. **New transformers releases.** BookNLP checkpoints still carry the
   ``bert.embeddings.position_ids`` buffer that recent ``transformers``
   versions have removed. The patch reloads the state dict with
   ``strict=False`` so the stale key is silently dropped.

Both fixes are no-ops on Linux / macOS and on older ``transformers``
versions — they just become irrelevant.
