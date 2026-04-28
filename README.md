# NLP-Graph (BookNLP per-chapter pipeline)

This repo contains a small, reproducible pipeline to go from a raw book `.txt`
to per-chapter BookNLP outputs and a few derived JSON artefacts.

## Setup

Use **Python 3.11** for BookNLP + tokenizers compatibility.

```bash
cd NLP-Graph
python3.11 -m venv .venv311
. .venv311/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Important: BERT / model downloads

Running the **big** model requires downloading BookNLP model weights the first time:

- cached under `data/booknlp_models/` (downloaded automatically by `run_booknlp_per_chapter.py`)
- plus HuggingFace tokenizer cache (recommended to keep local to the repo):

```bash
export HF_HOME="$PWD/data/hf_cache"
export TRANSFORMERS_CACHE="$PWD/data/hf_cache"
export XDG_CACHE_HOME="$PWD/data/hf_cache"
```

## End-to-end pipeline (raw book -> where we are now)

Assuming your raw input is:

- `data/raw/46.txt`

### 1) Split into chapters (and strip Gutenberg footer)

```bash
python src/step1_split_only.py --input data/raw/46.txt --book-id 46 --debug --log-level INFO
```

Outputs:
- `data/books/46/chapters/chapter_000.txt` ... `chapter_004.txt`
- `data/books/46/chapters/chapters.json`

### 2) Run BookNLP per chapter (big model, full pipeline)

```bash
python run_booknlp_per_chapter.py --book-id 46 --model-size big --pipeline entity,quote,supersense,event,coref --log-level INFO
```

Outputs (one folder per chapter):
- `data/booknlp_chapter_output/46/booknlp_46_chapter_0000/*`
- ...
- `data/booknlp_chapter_output/46/booknlp_46_chapter_0004/*`

### 3) Derived JSON artefacts (no new NLP; reads BookNLP outputs only)

Characters (counts + aliases), per chapter:

```bash
python src/step2_character_list_per_chapter.py --book-id 46
```

Predicates between character pairs (unordered), per chapter:

```bash
python src/step3_predicates_between_characters.py --book-id 46
```

Concatenated co-occurrence sentences for every possible unordered character pair, per chapter:

```bash
python src/step4_sentences_by_character_pair.py --book-id 46
```

## Notes

- This repo ignores `data/**` by default in git, except `data/raw/46.txt` as a reference input.
- BookNLP is CPU by default; first run is slower due to model downloads.

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
