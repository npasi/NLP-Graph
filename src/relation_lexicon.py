"""Lexicons used to score *character–character relations* in prose.

The previous version of the pipeline ran a generic sentiment classifier
on the text of a 100-token window. That measures the sentiment of the
**paragraph**, not of the **relation**. Austen's satirical prose in
``Persuasion`` is negative on the surface because the narrator
criticises Sir Walter — but that does not mean Anne ↔ Lady Russell is
a hostile pair.

To fix it we build edges from evidence that actually *involves both
characters*: the sentences where they co-occur, and the verbs /
adjectives appearing in those sentences. This module provides:

1. A **curated list of ~150 relational English verbs** grouped in five
   semantic classes (affection, cooperation, hostility, conflict,
   dialogue-neutral). Each verb has a polarity in ``[-1, +1]``.
2. A loader for VADER's lexicon (``~7500 words`` with valence in
   ``[-4, +4]``) used as a **fallback** for any content word not in the
   curated list.

Lookups are case-insensitive and lemma-aware (we strip simple suffixes
like ``-ed``, ``-ing``, ``-s``). The exposed API is tiny::

    from src.relation_lexicon import RelationLexicon

    lex = RelationLexicon()
    pol, cls = lex.lookup("hated")     # (-0.9, "hostility")
    pol, cls = lex.lookup("comforted") # (+0.8, "affection")

The scoring layer (``relation_scorer.py``) sits on top of this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Curated relational-verb lexicon                                             #
# --------------------------------------------------------------------------- #

# Class → (polarity, verbs). Polarity is roughly calibrated on a -1..+1
# scale so the numbers can be averaged alongside the VADER fallback
# (which we re-scale into the same range). Keep the verbs **base form
# only** — the ``_stem_candidates`` helper handles -ed/-ing/-s variants.

_CURATED: Dict[str, Tuple[float, List[str]]] = {
    "affection": (+0.9, [
        "love", "adore", "cherish", "admire", "esteem", "worship",
        "revere", "like", "delight", "befriend", "kiss", "embrace",
        "hug", "caress", "enamor", "idolize", "prize", "treasure",
        "favour", "favor",
    ]),
    "cooperation": (+0.6, [
        "help", "aid", "assist", "support", "comfort", "console",
        "reassure", "encourage", "welcome", "thank", "greet", "visit",
        "accompany", "join", "agree", "bless", "forgive", "pardon",
        "rescue", "save", "protect", "defend", "shelter", "provide",
        "share", "nurse", "tend", "serve", "obey", "trust", "respect",
        "praise", "commend", "approve", "compliment", "invite",
        "promise",
    ]),
    "dialogue": (0.0, [
        "say", "tell", "ask", "answer", "reply", "speak", "talk",
        "address", "inquire", "whisper", "remark", "observe", "mention",
        "mutter", "declare", "announce", "respond", "continue",
        "explain", "repeat", "pronounce", "utter",
    ]),
    "conflict": (-0.6, [
        "argue", "quarrel", "dispute", "disagree", "scold", "rebuke",
        "chide", "criticise", "criticize", "blame", "accuse", "reject",
        "refuse", "deny", "oppose", "resist", "rival", "avoid",
        "snub", "ignore", "neglect", "slight", "dismiss", "disapprove",
        "complain", "object", "contradict", "mock", "ridicule",
    ]),
    "hostility": (-0.9, [
        "hate", "despise", "loathe", "abhor", "detest", "scorn",
        "insult", "offend", "wound", "hurt", "strike", "attack",
        "assault", "fight", "beat", "threaten", "menace", "frighten",
        "terrify", "curse", "betray", "deceive", "cheat", "mislead",
        "abandon", "desert", "disown", "kill", "murder", "stab",
        "shoot", "slay", "poison", "injure", "torment", "persecute",
        "punish", "revenge", "suspect",
    ]),
}


# --------------------------------------------------------------------------- #
#  Stemming helpers                                                            #
# --------------------------------------------------------------------------- #

def _stem_candidates(word: str) -> Iterable[str]:
    """Yield possible base forms of *word* (lowercased).

    Simple rule-based stemming that handles the most common English
    suffixes you actually see in 19th-century prose. Not perfect, but
    good enough for lexicon lookup where the keyed form is always the
    base verb.
    """
    w = word.lower().strip()
    yield w
    if len(w) < 4:
        return
    if w.endswith("ied"):
        yield w[:-3] + "y"
    if w.endswith("ed"):
        yield w[:-2]
        yield w[:-1]            # "loved" -> "love"
    if w.endswith("ing"):
        yield w[:-3]
        yield w[:-3] + "e"      # "loving" -> "love"
    if w.endswith("es"):
        yield w[:-2]
        yield w[:-1]
    elif w.endswith("s") and not w.endswith("ss"):
        yield w[:-1]


# --------------------------------------------------------------------------- #
#  VADER lexicon loader (fallback)                                             #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_vader_lexicon() -> Dict[str, float]:
    """Return ``{word: valence_in_[-1,+1]}`` for the VADER lexicon.

    VADER ships a 7 500-entry file where valence is already a float in
    roughly ``[-4, +4]``; we divide by 4 to align with the curated
    scale. Loaded once per process.
    """
    try:
        import vaderSentiment
        import os
        base = os.path.dirname(vaderSentiment.__file__)
        lex_path = os.path.join(base, "vader_lexicon.txt")
        if not os.path.isfile(lex_path):
            logger.warning("VADER lexicon not found at %s", lex_path)
            return {}

        out: Dict[str, float] = {}
        with open(lex_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                out[word] = max(-1.0, min(1.0, score / 4.0))
        logger.info("Loaded VADER lexicon fallback (%d entries).", len(out))
        return out
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load VADER fallback lexicon: %s", exc)
        return {}


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LexiconHit:
    word: str
    polarity: float
    klass: str           # "affection" / "cooperation" / "conflict" /
                         # "hostility" / "dialogue" / "vader"
    source: str          # "curated" / "vader"


class RelationLexicon:
    """Two-layer lookup: curated verbs first, VADER lexicon as fallback."""

    def __init__(self, use_vader_fallback: bool = True) -> None:
        self._curated: Dict[str, Tuple[float, str]] = {}
        for klass, (polarity, verbs) in _CURATED.items():
            for v in verbs:
                self._curated[v.lower()] = (polarity, klass)
        self._vader = _load_vader_lexicon() if use_vader_fallback else {}
        logger.info(
            "RelationLexicon ready: %d curated verbs, %d VADER fallback entries.",
            len(self._curated), len(self._vader),
        )

    # ---- single-word lookup ----
    def lookup(self, token: str) -> Optional[LexiconHit]:
        for cand in _stem_candidates(token):
            if cand in self._curated:
                polarity, klass = self._curated[cand]
                return LexiconHit(word=cand, polarity=polarity, klass=klass, source="curated")
        for cand in _stem_candidates(token):
            if cand in self._vader:
                return LexiconHit(
                    word=cand, polarity=self._vader[cand],
                    klass="vader", source="vader",
                )
        return None

    # ---- whole-sentence scoring ----
    _WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")

    def score_sentence(self, sentence: str) -> Tuple[float, List[LexiconHit]]:
        """Return ``(avg_polarity, hits)`` over content words in *sentence*.

        The polarity is the simple average of matched word polarities
        (curated words count double — they're more reliable on literary
        text than VADER's generic lexicon).
        """
        hits: List[LexiconHit] = []
        total = 0.0
        weight = 0.0
        for m in self._WORD_RE.finditer(sentence):
            word = m.group(0)
            hit = self.lookup(word)
            if hit is None:
                continue
            hits.append(hit)
            w = 2.0 if hit.source == "curated" else 1.0
            total += hit.polarity * w
            weight += w
        if weight == 0:
            return 0.0, hits
        return total / weight, hits


__all__ = [
    "RelationLexicon",
    "LexiconHit",
]
