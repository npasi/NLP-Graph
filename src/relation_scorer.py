"""Score character-pair relations from BookNLP output.

This is the "smart" layer on top of BookNLP that replaces the old
``N-token sliding window + generic DistilBERT`` approach. For every
chapter we:

1. Index BookNLP tokens by sentence.
2. Find, per character, the sentence-IDs it's mentioned in.
3. For every *pair* of characters, take the set of sentences where
   **both** appear and enrich it with a ±1-sentence context (spec 1-B).
4. Compute:

   * ``weight``      — raw number of co-occurrence events (kept so old
     metrics still work).
   * ``pmi``         — Pointwise Mutual Information, the **preferred**
     structural affinity score. Positive ⇒ they appear together more
     than random chance; negative ⇒ they avoid each other.
   * ``polarity``    — average ``[-1, +1]`` polarity of the evidence
     sentences, from :class:`RelationLexicon`.
   * ``relation``    — the dominant class among the verbs observed:
     ``affection / cooperation / dialogue / conflict / hostility``.
   * ``evidence``    — list of ``(chapter_offset_frase, snippet, pol,
     relation_class)`` — small but enough for a “why is this rosso?”
     inspection.

The module is pure-Python + pandas; no torch / transformers needed here.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from relation_lexicon import LexiconHit, RelationLexicon

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Context / scoring tunables                                                  #
# --------------------------------------------------------------------------- #

CONTEXT_RADIUS = 1               # sentences before/after the target sentence
MAX_EVIDENCE_PER_EDGE = 5        # remember at most this many snippets per pair
POLARITY_LABEL_THRESHOLD = 0.1   # absolute polarity below → neutral
PMI_SMOOTHING = 0.5              # Laplace-like smoothing for rare pairs


# --------------------------------------------------------------------------- #
#  Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class EdgeEvidence:
    sentence_id: int
    snippet: str
    polarity: float
    relation: str                   # dominant class of verbs in this snippet
    cues: List[str] = field(default_factory=list)


@dataclass
class EdgeScores:
    weight: int                     # raw co-occurrence count
    pmi: float                      # pointwise mutual information
    polarity: float                 # avg polarity across evidence
    relation: str                   # dominant class across evidence
    evidence: List[EdgeEvidence] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _build_sentence_index(tokens: pd.DataFrame) -> Tuple[
    Dict[int, str], Dict[int, int]
]:
    """Return ``(sentence_id -> text, token_id -> sentence_id)``."""
    sent_col = "sentence_ID"
    tok_col = "token_ID_within_document"
    word_col = "word"

    token_to_sent: Dict[int, int] = dict(
        zip(tokens[tok_col].astype(int), tokens[sent_col].astype(int))
    )
    sentence_text: Dict[int, str] = {}
    for sid, group in tokens.groupby(sent_col):
        sentence_text[int(sid)] = " ".join(group[word_col].astype(str).tolist())
    return sentence_text, token_to_sent


def _character_sentences(
    entities: pd.DataFrame,
    token_to_sent: Dict[int, int],
    character_ids: Iterable[int],
) -> Dict[int, Set[int]]:
    """Return ``{coref_id: {sentence_id, ...}}`` for every kept character."""
    keep = set(int(c) for c in character_ids)
    out: Dict[int, Set[int]] = {cid: set() for cid in keep}
    for _, row in entities.iterrows():
        cid = int(row["COREF"])
        if cid not in keep:
            continue
        sid = token_to_sent.get(int(row["start_token"]))
        if sid is None:
            continue
        out[cid].add(sid)
    return out


def _cleanup_sentence(text: str) -> str:
    """Light cosmetic cleaning for evidence display."""
    # Collapse multiple whitespaces, strip BookNLP sentinel spaces
    # around punctuation, limit length.
    t = re.sub(r"\s+([,.;:!?])", r"\1", text)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 260:
        t = t[:257].rstrip() + "…"
    return t


def _aggregate_class(cues: List[LexiconHit]) -> str:
    """Return the dominant relation class among a list of cues."""
    if not cues:
        return "neutral"
    counts: Counter = Counter()
    for hit in cues:
        if hit.klass == "vader":
            counts["neutral"] += 0.5
        else:
            counts[hit.klass] += 1
    return counts.most_common(1)[0][0]


def _label_polarity(score: float) -> str:
    if score >= POLARITY_LABEL_THRESHOLD:
        return "positive"
    if score <= -POLARITY_LABEL_THRESHOLD:
        return "negative"
    return "neutral"


# --------------------------------------------------------------------------- #
#  Main scorer                                                                 #
# --------------------------------------------------------------------------- #

class RelationScorer:
    """Compute :class:`EdgeScores` for every pair of characters in a chapter."""

    def __init__(
        self,
        lexicon: Optional[RelationLexicon] = None,
        context_radius: int = CONTEXT_RADIUS,
        max_evidence: int = MAX_EVIDENCE_PER_EDGE,
    ) -> None:
        self.lexicon = lexicon or RelationLexicon()
        self.context_radius = context_radius
        self.max_evidence = max_evidence

    # ------------------------------------------------------------------ #
    #  Public                                                             #
    # ------------------------------------------------------------------ #

    def score_chapter(
        self,
        tokens: pd.DataFrame,
        entities: pd.DataFrame,
        character_ids: Iterable[int],
    ) -> Dict[Tuple[int, int], EdgeScores]:
        """Return ``{(cid_a, cid_b): EdgeScores}`` for every qualifying pair."""

        sentence_text, token_to_sent = _build_sentence_index(tokens)
        char_sents = _character_sentences(entities, token_to_sent, character_ids)
        char_list = [cid for cid, s in char_sents.items() if s]

        # Raw per-character frequency (how many sentences mention each).
        total_sentences = len(sentence_text) or 1
        char_freq = {cid: len(s) for cid, s in char_sents.items()}

        result: Dict[Tuple[int, int], EdgeScores] = {}
        for i in range(len(char_list)):
            a = char_list[i]
            for j in range(i + 1, len(char_list)):
                b = char_list[j]
                shared = char_sents[a] & char_sents[b]
                if not shared:
                    continue
                pair = (a, b)

                # ---- weight (raw) ----
                weight = len(shared)

                # ---- PMI ----
                p_a = char_freq[a] / total_sentences
                p_b = char_freq[b] / total_sentences
                p_ab = (weight + PMI_SMOOTHING) / (total_sentences + PMI_SMOOTHING)
                denom = p_a * p_b
                pmi = math.log((p_ab) / denom) if denom > 0 else 0.0
                # Normalize against the self-information for interpretability.
                self_info = -math.log(p_ab) if p_ab > 0 else 1.0
                npmi = pmi / self_info if self_info > 0 else 0.0
                npmi = max(-1.0, min(1.0, npmi))

                # ---- polarity & evidence (sentence + context) ----
                polarities: List[float] = []
                all_cues: List[LexiconHit] = []
                evidence: List[EdgeEvidence] = []

                seen_sids: Set[int] = set()
                for sid in sorted(shared):
                    # Expand ±context_radius around every co-occurrence sentence.
                    window_sids = [
                        s for s in range(sid - self.context_radius,
                                         sid + self.context_radius + 1)
                        if s in sentence_text and s not in seen_sids
                    ]
                    seen_sids.update(window_sids)
                    snippet = " ".join(sentence_text[s] for s in window_sids)
                    pol, cues = self.lexicon.score_sentence(snippet)
                    polarities.append(pol)
                    all_cues.extend(cues)
                    if len(evidence) < self.max_evidence:
                        evidence.append(EdgeEvidence(
                            sentence_id=sid,
                            snippet=_cleanup_sentence(snippet),
                            polarity=round(pol, 4),
                            relation=_aggregate_class(cues),
                            cues=[h.word for h in cues],
                        ))

                avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
                dominant = _aggregate_class(all_cues)
                # Map "dialogue" class (polarity 0) to "neutral" if polarity is low.
                if dominant == "dialogue" and abs(avg_polarity) < POLARITY_LABEL_THRESHOLD:
                    dominant = "neutral"

                result[pair] = EdgeScores(
                    weight=weight,
                    pmi=round(npmi, 4),
                    polarity=round(avg_polarity, 4),
                    relation=dominant,
                    evidence=evidence,
                )

        return result


def polarity_color_label(polarity: float) -> str:
    """Symmetric label used by the viz — same thresholds as ``_label_polarity``."""
    return _label_polarity(polarity)


__all__ = [
    "RelationScorer",
    "EdgeScores",
    "EdgeEvidence",
    "polarity_color_label",
]
