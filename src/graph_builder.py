"""Build one character graph per chapter using BookNLP + NetworkX.

For each chapter we:

1. Run BookNLP (entity + coref + event + quote pipeline) to produce
   per-token / per-entity TSVs plus a ``.book`` JSON.
2. Resolve character clusters into nodes using a **three-mode filter**:

   * ``strict``    — only chains with at least one proper-name mention.
   * ``curated``   — proper-name chains + common-noun chains that are
     *definite* (e.g. ``"the governess"``, ``"the Admiral"``) and have
     at least ``min_common_mentions`` mentions. This is the default.
   * ``permissive`` — proper or common-noun chains, any mention count.
3. Rewire pronoun-only chains into strong chains via quote attribution.
4. For each pair of characters, derive an edge with:

   * ``weight``    — raw co-occurrence count (sentence-level).
   * ``pmi``       — normalised PMI, the structural affinity.
   * ``polarity``  — -1..+1 aggregated from relational verbs.
   * ``relation``  — dominant class (affection / cooperation /
     dialogue / conflict / hostility / neutral).
   * ``evidence``  — top-k snippets behind the score, for inspection.

The heavy BookNLP model is loaded lazily — instantiate
:class:`BookNLPGraphBuilder` once and reuse it across chapters / books
so the weights are only loaded from disk a single time.
"""

from __future__ import annotations

import json
import locale
import logging
import os
import re as _re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.relation_scorer import EdgeScores, RelationScorer, polarity_color_label

logger = logging.getLogger(__name__)

# Filter modes for character selection.
FILTER_MODES = ("strict", "curated", "permissive")
# Default is "curated": proper-name chains AND definite common-noun chains
# (≥ N NOM mentions, "the X" form). The bridge merge in Step 2 then folds
# common-noun chains into their proper-name owners ("the clerk" → Bob),
# and the unbridged ones (truly nameless side characters, "the gentleman")
# are dropped, so the final per-chapter graph only shows proper names.
DEFAULT_FILTER_MODE = "curated"

# Articles/determiners that mark a noun as "definite" — we accept those.
_DEFINITE_STARTS = {"the", "a", "an"}
# Possessives / demonstratives that we *don't* want to keep as characters.
_POSSESSIVE_STARTS = {
    "his", "her", "their", "my", "our", "your", "its",
    "this", "that", "these", "those",
}


# --------------------------------------------------------------------------- #
#  Windows compatibility patch for BookNLP                                     #
# --------------------------------------------------------------------------- #

_BOOKNLP_PATCHED = False


def _patch_booknlp_for_windows() -> None:
    """Patch BookNLP to work on Windows and with modern ``transformers``.

    Two upstream issues are fixed here:

    1. **Windows paths** — BookNLP uses ``model_file.split("/")[-1]`` in
       three constructors to get a model basename. On Windows the whole
       path (backslashes and all) leaks into the downstream HuggingFace
       repo id and hits ``HFValidationError``.
    2. **New transformers releases** — BookNLP checkpoints still carry
       the ``bert.embeddings.position_ids`` buffer that ``transformers``
       has since removed. We switch ``load_state_dict`` to
       ``strict=False`` so the extra key is simply ignored.

    The fix is idempotent: calling this function twice is a no-op.
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

    # --- entity_tagger.LitBankEntityTagger ---------------------------------
    def _patched_entity_init(self, model_file, model_tagset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tagset = seq_reader.read_tagset(model_tagset)
        supersense_path = pkg_resources.resource_filename(
            _et.__name__, "data/supersense.tagset"
        )
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
        self.model.load_state_dict(
            torch.load(model_file, map_location=device), strict=False,
        )
        wns_path = pkg_resources.resource_filename(
            _et.__name__, "data/wordnet.first.sense"
        )
        self.wns = self.read_wn(wns_path)

    _et.LitBankEntityTagger.__init__ = _patched_entity_init

    # --- litbank_coref.LitBankCoref ----------------------------------------
    def _patched_coref_init(self, modelFile, gender_cats, pronominalCorefOnly=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERTCorefTagger(
            gender_cats=gender_cats,
            freeze_bert=True,
            base_model=_basename_model_key(modelFile),
            pronominalCorefOnly=pronominalCorefOnly,
        )
        self.model.load_state_dict(
            torch.load(modelFile, map_location=device), strict=False,
        )
        self.model.to(device)
        self.model.eval()

    _lc.LitBankCoref.__init__ = _patched_coref_init

    # --- bert_qa.QuotationAttribution --------------------------------------
    def _patched_qa_init(self, modelFile):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERTSpeakerID(base_model=_basename_model_key(modelFile))
        self.model.load_state_dict(
            torch.load(modelFile, map_location=device), strict=False,
        )
        self.model.to(device)
        self.model.eval()

    _qa.QuotationAttribution.__init__ = _patched_qa_init

    _BOOKNLP_PATCHED = True
    logger.info("Applied Windows + transformers compatibility patches to BookNLP.")




# --------------------------------------------------------------------------- #
#  BookNLP wrapper                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class _BookNLPOutputs:
    """Paths + parsed frames from a single BookNLP run."""

    tokens: pd.DataFrame
    entities: pd.DataFrame
    quotes: pd.DataFrame
    book_meta: dict


# Pronouns we treat as "first-person narrator" markers. Chains dominated
# by these are dropped because they represent either a first-person
# narrator (which we don't want as a character node) or a BookNLP coref
# failure that couldn't be healed via quote-attribution merging.
_FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}

# Second-person pronouns — always garbage for our graph.
_SECOND_PERSON_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves",
}


class BookNLPGraphBuilder:
    """Run BookNLP once per chapter and turn the outputs into a graph.

    Parameters
    ----------
    output_root : Path
        Directory where BookNLP artefacts are cached (one subfolder per
        chapter → reruns skip the heavy model call).
    tmp_root : Path
        Scratch directory for BookNLP *input* files. Files here are
        removed after processing; only the cached outputs survive.
    model_size : str
        ``"big"`` (default, more accurate) or ``"small"`` (faster).
    min_mentions : int
        Drop character clusters with fewer than this many mentions.
    """

    def __init__(
        self,
        output_root: Path,
        tmp_root: Optional[Path] = None,
        model_size: str = "big",
        min_mentions: int = 2,
        filter_mode: str = DEFAULT_FILTER_MODE,
        min_common_mentions: int = 4,
        scorer: Optional[RelationScorer] = None,
    ) -> None:
        """Parameters
        ----------
        filter_mode : str
            ``"strict"`` (proper name required), ``"curated"`` (default:
            proper *or* definite common with ≥ ``min_common_mentions``
            occurrences — recovers things like ``"the governess"`` but
            not ``"his wife"``), or ``"permissive"`` (proper OR any
            common).
        min_common_mentions : int
            Threshold used by ``curated`` mode for common-noun chains.
        scorer : RelationScorer, optional
            Replace the default lexicon-based scorer (for tests).
        """
        if filter_mode not in FILTER_MODES:
            raise ValueError(f"filter_mode must be one of {FILTER_MODES}")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root = Path(tmp_root) if tmp_root is not None else self.output_root / "_tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.min_mentions = min_mentions
        self.filter_mode = filter_mode
        self.min_common_mentions = min_common_mentions
        self.scorer = scorer or RelationScorer()
        self._booknlp = None  # lazy

    # --------------------------------------------------------------------- #
    #  BookNLP lifecycle                                                     #
    # --------------------------------------------------------------------- #

    def _ensure_booknlp(self):
        if self._booknlp is None:
            _patch_booknlp_for_windows()
            from booknlp.booknlp import BookNLP  # heavy import (pulls torch)

            logger.info("Loading BookNLP (model=%s)…", self.model_size)
            model_params = {
                "pipeline": "entity,quote,supersense,event,coref",
                "model": self.model_size,
            }
            self._booknlp = BookNLP("en", model_params)
            logger.info("BookNLP ready.")
        return self._booknlp

    def _run_booknlp(self, text: str, run_id: str) -> _BookNLPOutputs:
        """Execute BookNLP on *text* and return parsed outputs.

        Skips the model run when cached output files already exist
        (makes the pipeline restartable).
        """
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        tokens_path = run_dir / f"{run_id}.tokens"
        entities_path = run_dir / f"{run_id}.entities"
        quotes_path = run_dir / f"{run_id}.quotes"
        supersense_path = run_dir / f"{run_id}.supersense"
        book_path = run_dir / f"{run_id}.book"
        html_path = run_dir / f"{run_id}.book.html"

        force = os.environ.get("BOOKNLP_FORCE_RERUN", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        cached_ok = (
            tokens_path.exists()
            and entities_path.exists()
            and book_path.exists()
            and quotes_path.exists()
            and supersense_path.exists()
            and html_path.exists()
        )

        if cached_ok and not force:
            logger.info("Re-using cached BookNLP output at %s", run_dir)
        else:
            if force:
                logger.info("BOOKNLP_FORCE_RERUN enabled — regenerating outputs in %s", run_dir)
            tmp_input = self.tmp_root / f"{run_id}.txt"
            # BookNLP opens the input file with the *default* system
            # encoding (cp1252 on Windows-IT), so we match it here to
            # avoid UnicodeDecodeError on smart quotes / em dashes.
            # ``errors='replace'`` swaps any truly out-of-range character
            # for ``?`` — extremely rare for English fiction.
            bn_encoding = locale.getpreferredencoding(False) or "utf-8"
            tmp_input.write_text(text, encoding=bn_encoding, errors="replace")
            try:
                booknlp = self._ensure_booknlp()
                logger.info("Running BookNLP on %s (%d chars)…", run_id, len(text))
                booknlp.process(str(tmp_input), str(run_dir), run_id)
            finally:
                try:
                    tmp_input.unlink()
                except OSError:
                    pass  # best-effort cleanup

        tokens = pd.read_csv(tokens_path, sep="\t", quoting=3, keep_default_na=False)
        entities = pd.read_csv(entities_path, sep="\t", quoting=3, keep_default_na=False)
        if quotes_path.exists():
            quotes = pd.read_csv(quotes_path, sep="\t", quoting=3, keep_default_na=False)
        else:
            quotes = pd.DataFrame(columns=[
                "quote_start", "quote_end", "mention_start", "mention_end",
                "mention_phrase", "char_id", "quote",
            ])
        book_meta = json.loads(book_path.read_text(encoding="utf-8"))
        return _BookNLPOutputs(
            tokens=tokens, entities=entities, quotes=quotes, book_meta=book_meta,
        )

    # --------------------------------------------------------------------- #
    #  Character resolution                                                  #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _canonical_character_info(character: dict) -> Tuple[str, List[str]]:
        """Pick a canonical name + alias list from a BookNLP character dict.

        Preference order:

        1. Most frequent proper-noun mention (e.g. ``"Elizabeth"``).
        2. Most frequent common-noun mention (e.g. ``"the governess"``).
        3. Most frequent pronoun (e.g. ``"I"`` for first-person narrators).
        4. Fallback ``character_<id>`` if the cluster has nothing we can use.
        """
        mentions = character.get("mentions", {}) or {}
        proper = mentions.get("proper", []) or []
        common = mentions.get("common", []) or []
        pronoun = mentions.get("pronoun", []) or []

        def _sorted(items: Iterable[dict]) -> List[Tuple[str, int]]:
            return sorted(
                ((m.get("n", "").strip(), m.get("c", 0)) for m in items if m.get("n")),
                key=lambda x: -x[1],
            )

        proper_sorted = _sorted(proper)
        common_sorted = _sorted(common)
        pronoun_sorted = _sorted(pronoun)

        if proper_sorted:
            canonical = proper_sorted[0][0]
        elif common_sorted:
            canonical = common_sorted[0][0]
        elif pronoun_sorted:
            canonical = pronoun_sorted[0][0]
        else:
            canonical = f"character_{character.get('id', '?')}"

        aliases = [n for n, _ in proper_sorted + common_sorted + pronoun_sorted]
        return canonical, aliases

    # ---------- Chain "strength" helpers ----------

    @staticmethod
    def _chain_mention_types(entities: pd.DataFrame) -> Dict[int, Dict[str, int]]:
        """Return ``{coref_id: {'PROP': n, 'NOM': n, 'PRON': n}}`` counts.

        Restricted to PER entities — non-person cats don't influence our
        character-ness decision.
        """
        counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"PROP": 0, "NOM": 0, "PRON": 0}
        )
        per_df = entities[entities["cat"] == "PER"]
        for cid, prop in zip(per_df["COREF"].astype(int), per_df["prop"].astype(str)):
            if prop in counts[cid]:
                counts[cid][prop] += 1
        return counts

    @staticmethod
    def _is_first_person_chain(
        coref_id: int, entities: pd.DataFrame,
    ) -> bool:
        """True if a chain is effectively a 1st-person narrator's pronouns."""
        per_df = entities[
            (entities["COREF"].astype(int) == coref_id)
            & (entities["cat"] == "PER")
        ]
        if per_df.empty:
            return False
        props = per_df["prop"].astype(str).tolist()
        texts = per_df["text"].astype(str).str.lower().tolist()
        if any(p != "PRON" for p in props):
            return False  # has a proper or common mention → not a bare narrator
        fp_hits = sum(1 for t in texts if t.strip() in _FIRST_PERSON_PRONOUNS)
        return fp_hits / max(len(texts), 1) >= 0.7  # strong first-person majority

    # ---------- Quote-attribution merge ----------

    def _merge_weak_via_quotes(
        self,
        outputs: _BookNLPOutputs,
        strong_ids: set,
    ) -> pd.DataFrame:
        """Return an updated entities DataFrame with "weak" pronoun mentions
        re-routed to the speaker of the enclosing quote (if the speaker is
        a strong chain).

        The strategy is:
          * Index BookNLP quotes by ``quote_start`` for a binary search.
          * For every entity row with ``prop == PRON`` in a weak chain,
            check whether its start token falls inside a quote and whether
            that quote's speaker is a strong chain.
          * If so, reassign the mention's COREF to the speaker's id.

        This rescues things like a dialogue "I" that BookNLP failed to
        link to the speaker's proper-name chain.
        """
        if outputs.quotes.empty or not strong_ids:
            return outputs.entities

        # Build a sorted list of (quote_start, quote_end, speaker_id).
        q = outputs.quotes.copy()
        q["quote_start"] = pd.to_numeric(q["quote_start"], errors="coerce")
        q["quote_end"] = pd.to_numeric(q["quote_end"], errors="coerce")
        q["char_id"] = pd.to_numeric(q["char_id"], errors="coerce")
        q = q.dropna(subset=["quote_start", "quote_end", "char_id"])
        q_starts = q["quote_start"].astype(int).tolist()
        q_ends = q["quote_end"].astype(int).tolist()
        q_speakers = q["char_id"].astype(int).tolist()

        # Sort by start for a simple linear scan (quotes rarely overlap).
        order = sorted(range(len(q_starts)), key=lambda i: q_starts[i])
        q_starts = [q_starts[i] for i in order]
        q_ends = [q_ends[i] for i in order]
        q_speakers = [q_speakers[i] for i in order]

        def _speaker_for_token(tok: int) -> Optional[int]:
            # Linear scan is fast enough for per-chapter quote counts.
            for s, e, sp in zip(q_starts, q_ends, q_speakers):
                if s <= tok <= e:
                    return sp if sp in strong_ids else None
                if s > tok:
                    break
            return None

        entities = outputs.entities.copy()
        entities["COREF"] = entities["COREF"].astype(int)

        reassigned = 0
        for idx, row in entities.iterrows():
            if row["COREF"] in strong_ids:
                continue
            if row.get("cat") != "PER":
                continue
            if str(row.get("prop")) != "PRON":
                continue
            speaker = _speaker_for_token(int(row["start_token"]))
            if speaker is not None:
                entities.at[idx, "COREF"] = speaker
                reassigned += 1

        if reassigned:
            logger.info(
                "Quote-attribution merge: reassigned %d pronoun mention(s) "
                "to their speakers' chains.", reassigned,
            )
        return entities

    # ---------- Public character selection ----------

    def _select_person_characters(
        self, outputs: _BookNLPOutputs,
    ) -> Tuple[Dict[int, dict], pd.DataFrame]:
        """Return ``(strong_clusters, updated_entities_df)``.

        A cluster is *strong* when it contains at least one PER mention
        that is either a proper name (``PROP``) or a common noun
        (``NOM``). Pronoun-only chains are dropped, after first trying
        to merge them into the speaker's chain via quote attribution.
        """
        if "cat" not in outputs.entities.columns or outputs.entities.empty:
            return {}, outputs.entities

        # Types counts before any merge.
        initial_type_counts = self._chain_mention_types(outputs.entities)

        def _is_strong(cid: int, counts: Dict[str, int],
                       entities_df: pd.DataFrame) -> bool:
            if counts["PROP"] > 0:
                return True   # proper-name chains always kept
            if self.filter_mode == "strict":
                return False
            if counts["NOM"] == 0:
                return False
            if self.filter_mode == "permissive":
                return True
            # --- curated mode --------------------------------------------
            if counts["NOM"] < self.min_common_mentions:
                return False
            # Must be *definite*: at least one NOM mention starts with
            # an article or a capitalised token (role-noun like "Admiral"),
            # and no NOM mention starts with a possessive/demonstrative.
            nom_texts = entities_df.loc[
                (entities_df["COREF"].astype(int) == cid)
                & (entities_df["prop"] == "NOM"),
                "text",
            ].astype(str).tolist()
            if not nom_texts:
                return False
            has_definite = False
            for t in nom_texts:
                first = (t.strip().split() or [""])[0].lower()
                if first in _POSSESSIVE_STARTS:
                    return False       # possessive chains are thrown away
                if first in _DEFINITE_STARTS:
                    has_definite = True
                elif first and first[0].isupper():
                    has_definite = True
            return has_definite

        strong_ids_initial = {
            cid for cid, c in initial_type_counts.items()
            if _is_strong(cid, c, outputs.entities)
        }

        # Step 1 + 2: try to rescue weak (pronoun-only) chains via quotes.
        entities = self._merge_weak_via_quotes(outputs, strong_ids_initial)

        # Recompute after the merge.
        type_counts = self._chain_mention_types(entities)
        strong_ids = {
            cid for cid, c in type_counts.items()
            if _is_strong(cid, c, entities)
        }

        # Step 3: drop chains that remained weak (true pronoun-only).
        # Also drop explicit first-person narrator chains (defensive:
        # the weak-drop above already catches them in practice).
        narrator_ids = {
            cid for cid in set(type_counts) - strong_ids
            if self._is_first_person_chain(cid, entities)
        }
        if narrator_ids:
            logger.info("Dropping %d first-person narrator chain(s): %s",
                        len(narrator_ids), sorted(narrator_ids))

        # Count mentions per chain (post-merge) for min_mentions check.
        per_entities = entities[entities["cat"] == "PER"]
        post_counts = per_entities["COREF"].astype(int).value_counts().to_dict()

        selected: Dict[int, dict] = {}
        book_characters_by_id = {
            int(c.get("id", -1)): c for c in outputs.book_meta.get("characters", [])
        }
        for cid in strong_ids:
            if post_counts.get(cid, 0) < self.min_mentions:
                continue
            base = dict(book_characters_by_id.get(cid, {"id": cid}))
            # Override the raw BookNLP count with the post-merge count so
            # that quote-merged pronouns contribute to node size.
            base["count"] = post_counts.get(cid, base.get("count", 0))
            selected[cid] = base

        logger.info(
            "Kept %d / %d chains (strong=%d, weak=%d, narrator-dropped=%d, "
            "min_mentions=%d).",
            len(selected), len(type_counts),
            len(strong_ids),
            len(set(type_counts) - strong_ids),
            len(narrator_ids),
            self.min_mentions,
        )
        return selected, entities

    # --------------------------------------------------------------------- #
    #  Graph assembly                                                        #
    # --------------------------------------------------------------------- #

    def build_chapter_graph(
        self,
        chapter: dict,
        book_id: str,
    ) -> nx.Graph:
        """Run BookNLP on a chapter and return its character graph.

        Parameters
        ----------
        chapter : dict
            A record from :func:`src.chapter_splitter.split_into_chapters`
            with keys ``chapter_id``, ``title``, ``text``.
        book_id : str
            Slug identifying the book (used in output paths). Different
            books must use different ``book_id`` strings to keep BookNLP
            caches separate.

        Returns
        -------
        networkx.Graph
        """
        chapter_id = int(chapter["chapter_id"])
        chapter_text = chapter["text"]
        chapter_title = chapter.get("title", "")
        n_tokens_whitespace = len(chapter_text.split())

        graph = nx.Graph()
        graph.graph.update({
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "num_tokens": n_tokens_whitespace,
            "short_chapter": bool(chapter.get("short", n_tokens_whitespace < 200)),
            "book_id": book_id,
        })

        if len(chapter_text.strip()) < 50:
            logger.warning("Chapter %d is effectively empty — skipping BookNLP.", chapter_id)
            graph.graph["note"] = "empty_chapter"
            return graph

        run_id = f"{book_id}__chapter_{chapter_id:04d}"
        outputs = self._run_booknlp(chapter_text, run_id=run_id)
        characters, merged_entities = self._select_person_characters(outputs)

        # Per-chain mention-type counts on the merged entities (used to
        # tag nodes with ``has_proper`` / ``has_common`` for the viz).
        post_type_counts = self._chain_mention_types(merged_entities)

        # ------- Nodes -------
        for cid, info in characters.items():
            canonical, aliases = self._canonical_character_info(info)
            gender = (info.get("g") or {}).get("inference", "unknown")
            types = post_type_counts.get(cid, {"PROP": 0, "NOM": 0, "PRON": 0})
            graph.add_node(
                cid,
                name=canonical,
                aliases=aliases,
                mention_count=int(info.get("count", 0)),
                gender=gender,
                has_proper=bool(types["PROP"] > 0),
                has_common=bool(types["NOM"] > 0),
            )

        if graph.number_of_nodes() < 2:
            logger.info("Chapter %d has < 2 characters — no edges to add.", chapter_id)
            return graph

        # ------- Edges (sentence-level co-occurrence + relational scoring) ----
        pair_scores = self.scorer.score_chapter(
            tokens=outputs.tokens,
            entities=merged_entities,
            character_ids=characters.keys(),
        )

        for (a, b), s in pair_scores.items():
            # Keep the top-1 snippet as a compact 'description' attribute
            # for back-compat / quick inspection.
            description = ""
            if s.evidence:
                top = max(s.evidence, key=lambda e: abs(e.polarity))
                description = top.snippet
            graph.add_edge(
                a, b,
                weight=s.weight,
                pmi=s.pmi,
                polarity=s.polarity,
                relation=s.relation,
                sentiment=polarity_color_label(s.polarity),  # viz-friendly
                sentiment_score=s.polarity,
                description=description,
                evidence=[asdict(e) for e in s.evidence],
            )

        logger.info(
            "Chapter %d graph: %d nodes, %d edges (relational scoring).",
            chapter_id, graph.number_of_nodes(), graph.number_of_edges(),
        )
        return graph


# --------------------------------------------------------------------------- #
#  Functional convenience                                                      #
# --------------------------------------------------------------------------- #

_DEFAULT_BUILDER: Optional[BookNLPGraphBuilder] = None


def build_chapter_graph(
    chapter: dict,
    book_id: str,
    *,
    output_root: Optional[Path] = None,
    tmp_root: Optional[Path] = None,
    model_size: str = "big",
    filter_mode: str = DEFAULT_FILTER_MODE,
) -> nx.Graph:
    """Convenience wrapper that reuses a module-level builder."""
    global _DEFAULT_BUILDER
    project_root = Path(__file__).resolve().parent.parent
    if _DEFAULT_BUILDER is None:
        _DEFAULT_BUILDER = BookNLPGraphBuilder(
            output_root=Path(output_root) if output_root else project_root / "data" / "booknlp_output",
            tmp_root=Path(tmp_root) if tmp_root else project_root / "data" / "tmp",
            model_size=model_size,
            filter_mode=filter_mode,
        )
    return _DEFAULT_BUILDER.build_chapter_graph(chapter, book_id)


__all__ = ["BookNLPGraphBuilder", "build_chapter_graph", "FILTER_MODES"]
