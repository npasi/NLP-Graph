"""Character Identity Layer for BookNLP-based narrative graph pipeline.

Core invariant: never treat raw coref_id as a global identity.
Always key by (chapter_id, local_coref_id) -> canonical_character_id.

Cluster classification types:
  individual_candidate   – proper name present, single person
  group_candidate        – collective noun (children, policemen…)
  narrator_candidate     – first-person pronoun dominated, no proper/common names
  agent_nonhuman         – ghost, spirit, creature…
  review_role_candidate  – common-noun-only description ("the supervisor",
                           "a businessman", "the child")
  generic_noise          – everybody, dear, someone…
  abstract_noise         – Christmas, God, fate…
  ambiguous              – pronoun-only, not first-person dominated

Promotion policy  (PRECISION > RECALL):
  A cluster becomes a canonical character ONLY if:
    (a) alias_resolved  — explicit curator knowledge in the alias file, OR
    (b) individual_candidate  AND  at least one proper name, OR
    (c) agent_nonhuman        AND  at least one proper name

  The following types are NEVER promoted without (a):
    review_role_candidate, ambiguous, narrator_candidate, group_candidate,
    generic_noise, abstract_noise.

Decision types: AUTO_MERGE | REJECT | REVIEW | ABSTAIN
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary / classifier constants
# ---------------------------------------------------------------------------

_GENERIC_NOISE_EXACT: Set[str] = {
    "everybody", "everyone", "nobody", "no one", "no-one",
    "anybody", "anyone", "somebody", "someone",
    "one another", "each other", "each",
    "dear", "my dear",
    "all", "none", "both", "some", "many", "few", "most", "other", "others",
    "people", "the people", "the world",
    "you", "your", "yourself", "yourselves",
    "not everyone", "none of you", "none of them",
    "no one",
}

_ABSTRACT_NOISE_EXACT: Set[str] = {
    "christmas", "god", "heaven", "hell", "church", "death",
    "fate", "fortune", "nature", "society", "time",
    "the lord", "the almighty", "the creator",
    "business", "sunday", "justice",
}

_GROUP_WORD_MARKERS: Set[str] = {
    "children", "boys", "girls", "men", "women",
    "folks", "crowd", "crowds",
    "spirits", "ghosts", "creatures", "monsters",
    "gentlemen", "ladies",
    "passengers", "guests", "friends",
    "mourners", "clerks", "servants",
    "family", "families", "relations", "neighbors", "neighbours",
    "people", "persons", "beings",
    "choir", "band", "group", "company", "party",
    "officials", "guards", "officers", "defendants", "lawyers",
    "policemen", "attendants",
}

_FIRST_PERSON_FORMS: Set[str] = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}

_AGENT_NONHUMAN_WORDS: Set[str] = {
    "ghost", "spirit", "phantom", "spectre", "specter",
    "creature", "beast", "monster", "demon", "angel",
    "fairy", "witch", "wizard", "dragon", "shadow",
    "apparition", "presence",
}

# Title prefix → gender hint ("M" / "F" / None)
_TITLE_GENDER: Dict[str, Optional[str]] = {
    "mr.": "M", "mrs.": "F", "miss": "F", "ms.": "F",
    "sir": "M", "lord": None, "lady": "F",
    "uncle": "M", "aunt": "F",
    "dr.": None, "prof.": None, "master": None,
    "captain": None, "colonel": None, "general": None, "rev.": None,
}

# Pairs that indicate opposite-gender titles (block AUTO_MERGE)
_GENDERED_OPPOSITE_PAIRS: List[Tuple[str, str]] = [
    ("mr.", "mrs."), ("mr.", "ms."), ("mr.", "miss"), ("sir", "lady"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LocalCluster:
    chapter_id: int
    coref_id: str
    display_name: str
    proper_names: List[str]   # sorted by frequency desc
    common_names: List[str]
    pronoun_forms: List[str]
    mention_count: int
    cluster_type: str = "ambiguous"
    gender_hint: Optional[str] = None
    title_prefix: Optional[str] = None
    match_key: str = ""       # normalised base name used for grouping


@dataclass
class CanonicalCharacter:
    canonical_id: str
    name: str
    type: str
    aliases: List[str]
    source_clusters: List[Dict[str, Any]]  # [{chapter_id, coref_id}]
    confidence: float


@dataclass
class IdentityDecision:
    chapter_id: int
    coref_id: str
    surface: str
    target_canonical_id: Optional[str]
    decision: str     # AUTO_MERGE | REJECT | REVIEW | ABSTAIN
    confidence: float
    reasons: List[str]


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    n = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"\s+", " ", n)


def _strip_title(name: str) -> Tuple[str, Optional[str]]:
    """Return (base_name, title_lower_or_None) after stripping a leading title."""
    low = _norm(name)
    for title in sorted(_TITLE_GENDER, key=len, reverse=True):
        if low == title or low.startswith(title + " "):
            base = name[len(title):].strip()
            return base if base else name, title
    return name, None


def _make_canonical_id(name: str, existing: Set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _norm(name)).strip("_") or "char"
    cid = f"char_{slug}"
    if cid not in existing:
        return cid
    i = 2
    while f"{cid}_{i}" in existing:
        i += 1
    return f"{cid}_{i}"


# ---------------------------------------------------------------------------
# Cluster classification
# ---------------------------------------------------------------------------

def _is_first_person_dominated(pronoun_forms: List[str], total: int) -> bool:
    if not pronoun_forms or total == 0:
        return False
    fp = sum(1 for p in pronoun_forms if _norm(p) in _FIRST_PERSON_FORMS)
    return fp / len(pronoun_forms) >= 0.7


def _contains_group_marker(names: List[str]) -> bool:
    return any(w in _GROUP_WORD_MARKERS
               for n in names for w in _norm(n).split())


def _contains_nonhuman_marker(names: List[str]) -> bool:
    return any(w in _AGENT_NONHUMAN_WORDS
               for n in names for w in _norm(n).split())


def _is_generic_noise(display: str, all_names: List[str]) -> bool:
    return any(_norm(n) in _GENERIC_NOISE_EXACT for n in [display] + all_names)


def _is_abstract_noise(display: str, all_names: List[str]) -> bool:
    return any(_norm(n) in _ABSTRACT_NOISE_EXACT for n in [display] + all_names)


def classify_cluster(cl: LocalCluster) -> str:
    """Assign a cluster_type to a LocalCluster (pure function)."""
    all_names = cl.proper_names + cl.common_names

    if _is_generic_noise(cl.display_name, all_names):
        return "generic_noise"
    if _is_abstract_noise(cl.display_name, all_names):
        return "abstract_noise"

    # Pronoun-only clusters
    if not cl.proper_names and not cl.common_names:
        if _is_first_person_dominated(cl.pronoun_forms, cl.mention_count):
            return "narrator_candidate"
        return "ambiguous"

    if _contains_group_marker(all_names):
        return "group_candidate"

    if cl.proper_names:
        return "agent_nonhuman" if _contains_nonhuman_marker(cl.proper_names) else "individual_candidate"

    # Common-noun-only clusters
    if _contains_nonhuman_marker(cl.common_names):
        return "agent_nonhuman"

    _ambiguous_roles = {
        "the child", "the man", "the woman", "the girl", "the boy",
        "the gentleman", "the lady", "the old man", "the young man",
        "a man", "a woman",
    }
    if any(_norm(n) in _ambiguous_roles for n in cl.common_names):
        return "review_role_candidate"

    # Default for any common-noun-only cluster: always review_role_candidate.
    # This covers "a businessman", "a client", "the supervisor", etc.
    return "review_role_candidate"


# ---------------------------------------------------------------------------
# Promotion policy  (the key precision gate)
# ---------------------------------------------------------------------------

def _qualifies_for_promotion(cl: LocalCluster, alias_resolved: bool) -> bool:
    """
    Returns True ONLY if this cluster should become a canonical character.

    Rules (precision > recall — if uncertain, return False):
      - narrator_candidate:         NEVER promoted (not even via alias)
      - alias_resolved:             always promote (curator overrides everything)
      - individual_candidate:       promote only if proper_names is non-empty
      - agent_nonhuman:             promote only if proper_names is non-empty
      - everything else:            DO NOT promote

    Clusters that are never promoted without alias:
      review_role_candidate, ambiguous, group_candidate,
      generic_noise, abstract_noise.
    """
    # Hard exclusion — narrator pronouns (I/me/my…) are never canonical characters.
    if cl.cluster_type == "narrator_candidate":
        return False
    if alias_resolved:
        return True
    if cl.cluster_type == "individual_candidate":
        return bool(cl.proper_names)
    if cl.cluster_type == "agent_nonhuman":
        return bool(cl.proper_names)
    return False


def _non_promotable_decision(cluster_type: str) -> Tuple[str, str]:
    """Return (decision_str, reason) for a cluster that fails promotion."""
    if cluster_type in ("generic_noise", "abstract_noise"):
        return "REJECT", f"classified as {cluster_type}"
    if cluster_type == "narrator_candidate":
        return "ABSTAIN", "narrator_candidate — first-person pronoun dominated"
    if cluster_type == "ambiguous":
        return "ABSTAIN", "ambiguous — pronoun-only, no proper or common names"
    if cluster_type == "group_candidate":
        return "REVIEW", "group_candidate — collective noun, not a single character"
    if cluster_type == "review_role_candidate":
        return "REVIEW", (
            "review_role_candidate — no proper name and not alias-resolved; "
            "generic role description excluded by default"
        )
    if cluster_type == "agent_nonhuman":
        return "REVIEW", "agent_nonhuman — no proper name, not alias-resolved"
    return "REVIEW", f"unresolved type {cluster_type!r}"


# ---------------------------------------------------------------------------
# BookNLP .book file parsing
# ---------------------------------------------------------------------------

def _parse_book_file(book_path: Path, chapter_id: int) -> List[LocalCluster]:
    try:
        meta = json.loads(book_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        logger.error("Cannot parse %s: %s", book_path, exc)
        return []

    clusters: List[LocalCluster] = []
    for char in meta.get("characters", []) or []:
        cid = str(char.get("id", "?"))
        count = int(char.get("count", 0) or 0)
        mentions = char.get("mentions", {}) or {}

        def _extract(key: str) -> List[str]:
            return [
                str(m["n"]).strip()
                for m in sorted(
                    mentions.get(key, []) or [],
                    key=lambda x: -int(x.get("c", 0) or 0),
                )
                if str(m.get("n", "")).strip()
            ]

        proper  = _extract("proper")
        common  = _extract("common")
        pronoun = _extract("pronoun")
        display = (proper or common or pronoun or [f"coref_{cid}"])[0]

        _, title = _strip_title(display)
        gender_from_title = _TITLE_GENDER.get(title) if title else None
        # BookNLP gender: g.argmax is a string like "he/him/his" or "she/her"
        g_block = char.get("g") or {}
        gender_raw = str(g_block.get("argmax") or "").lower()
        if gender_from_title:
            gender_hint = gender_from_title
        elif "he" in gender_raw:
            gender_hint = "M"
        elif "she" in gender_raw:
            gender_hint = "F"
        else:
            gender_hint = None

        base, _ = _strip_title(display)
        match_key = _norm(base) if base else _norm(display)

        cl = LocalCluster(
            chapter_id=chapter_id,
            coref_id=cid,
            display_name=display,
            proper_names=proper,
            common_names=common,
            pronoun_forms=pronoun,
            mention_count=count,
            gender_hint=gender_hint,
            title_prefix=title,
            match_key=match_key,
        )
        cl.cluster_type = classify_cluster(cl)
        clusters.append(cl)
    return clusters


def _parse_chapter_id_from_dirname(name: str) -> Optional[int]:
    parts = name.split("_")
    return int(parts[-1]) if parts and parts[-1].isdigit() else None


# ---------------------------------------------------------------------------
# Alias file
# ---------------------------------------------------------------------------

def _load_alias_file(path: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Return (global_aliases, chapter_aliases)."""
    if not path or not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    global_aliases = {str(k): str(v) for k, v in (data.get("aliases") or {}).items()}
    chapter_aliases = {
        str(ch): {str(k): str(v) for k, v in ch_map.items()}
        for ch, ch_map in (data.get("chapter_aliases") or {}).items()
    }
    return global_aliases, chapter_aliases


def _resolve_alias(
    cl: LocalCluster,
    global_aliases: Dict[str, str],
    chapter_aliases: Dict[str, Dict[str, str]],
) -> Optional[str]:
    """Return the resolved canonical name from the alias file, or None."""
    ch_map = chapter_aliases.get(str(cl.chapter_id), {})
    for surface in [cl.display_name] + cl.proper_names + cl.common_names:
        if surface in ch_map:
            return ch_map[surface]
        if surface in global_aliases:
            return global_aliases[surface]
    return None


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _gendered_title_conflict(ta: Optional[str], tb: Optional[str]) -> bool:
    if not ta or not tb:
        return False
    return any(
        (ta == p[0] and tb == p[1]) or (ta == p[1] and tb == p[0])
        for p in _GENDERED_OPPOSITE_PAIRS
    )


def _decide_group_merge(
    clusters: List[LocalCluster],
) -> Tuple[str, float, List[str]]:
    if len(clusters) == 1:
        return "AUTO_MERGE", 1.0, ["single cluster"]

    types = {c.cluster_type for c in clusters}
    if "group_candidate" in types and types & {"individual_candidate", "agent_nonhuman"}:
        return "REVIEW", 0.5, ["mix of group and individual types"]

    titles = [c.title_prefix for c in clusters]
    for i, ta in enumerate(titles):
        for tb in titles[i + 1:]:
            if _gendered_title_conflict(ta, tb):
                return "REVIEW", 0.4, [
                    f"gendered title conflict: {ta!r} vs {tb!r} "
                    "(e.g. Mr. X vs Mrs. X — likely different people)"
                ]

    return "AUTO_MERGE", 0.9, ["same normalised name across chapters"]


# ---------------------------------------------------------------------------
# CharacterIdentityLayer
# ---------------------------------------------------------------------------

class CharacterIdentityLayer:
    """Maps per-chapter BookNLP coref clusters to canonical characters."""

    def __init__(
        self,
        booknlp_root: Path,
        alias_file: Optional[Path] = None,
        identity_mode: str = "auto_conservative",
    ) -> None:
        self.booknlp_root = Path(booknlp_root)
        self.alias_file = Path(alias_file) if alias_file else None
        self.identity_mode = identity_mode
        self._global_aliases: Dict[str, str] = {}
        self._chapter_aliases: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        if self.alias_file:
            self._global_aliases, self._chapter_aliases = _load_alias_file(self.alias_file)
            logger.info(
                "Loaded %d global aliases, %d chapter alias sets.",
                len(self._global_aliases), len(self._chapter_aliases),
            )

        all_clusters = self._read_all_clusters()
        logger.info("Read %d local clusters across all chapters.", len(all_clusters))

        decisions, canon_chars, unresolved = self._resolve_identities(all_clusters)
        mapping = self._build_mapping(decisions)
        report  = self._build_report(all_clusters, decisions, canon_chars)
        self._write_outputs(canon_chars, mapping, decisions, unresolved, report)

        logger.info(
            "Done: %d canonical characters | %d unresolved | %d rejected/abstained",
            len(canon_chars),
            sum(1 for d in decisions if d.decision in ("REVIEW", "ABSTAIN")),
            sum(1 for d in decisions if d.decision == "REJECT"),
        )
        return report

    # ------------------------------------------------------------------
    def _read_all_clusters(self) -> List[LocalCluster]:
        clusters: List[LocalCluster] = []
        for run_dir in sorted(
            (p for p in self.booknlp_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        ):
            ch_id = _parse_chapter_id_from_dirname(run_dir.name)
            if ch_id is None:
                continue
            book_files = sorted(run_dir.glob("*.book"))
            if not book_files:
                logger.warning("No .book in %s — skipping", run_dir)
                continue
            ch_clusters = _parse_book_file(book_files[0], ch_id)
            clusters.extend(ch_clusters)
            logger.debug("Chapter %d: %d clusters", ch_id, len(ch_clusters))
        return clusters

    # ------------------------------------------------------------------
    def _resolve_identities(
        self, all_clusters: List[LocalCluster]
    ) -> Tuple[List[IdentityDecision], List[CanonicalCharacter], List[Dict[str, Any]]]:

        decisions:    List[IdentityDecision]  = []
        canon_chars:  List[CanonicalCharacter] = []
        unresolved:   List[Dict[str, Any]]    = []
        used_ids:     Set[str]                = set()

        # ---- Phase 1: classify each cluster and decide fate ----
        # promotable: (cluster, resolved_name, alias_resolved)
        promotable: List[Tuple[LocalCluster, str, bool]] = []

        for cl in all_clusters:
            alias_name    = _resolve_alias(cl, self._global_aliases, self._chapter_aliases)
            alias_resolved = alias_name is not None

            if _qualifies_for_promotion(cl, alias_resolved):
                resolved_name = alias_name if alias_name else cl.display_name
                promotable.append((cl, resolved_name, alias_resolved))
            else:
                decision_str, reason = _non_promotable_decision(cl.cluster_type)
                decisions.append(IdentityDecision(
                    chapter_id=cl.chapter_id,
                    coref_id=cl.coref_id,
                    surface=cl.display_name,
                    target_canonical_id=None,
                    decision=decision_str,
                    confidence=0.0,
                    reasons=[reason],
                ))
                if decision_str in ("REVIEW", "ABSTAIN"):
                    unresolved.append({
                        "chapter_id":     cl.chapter_id,
                        "local_coref_id": cl.coref_id,
                        "surface":        cl.display_name,
                        "type":           cl.cluster_type,
                        "reason":         reason,
                        "candidate_targets": [],
                    })

        # ---- Phase 2: group promotable clusters by resolved match_key ----
        groups: Dict[str, List[Tuple[LocalCluster, str, bool]]] = defaultdict(list)
        for cl, resolved_name, alias_resolved in promotable:
            base, _ = _strip_title(resolved_name)
            key = _norm(base) if base else _norm(resolved_name)
            groups[key].append((cl, resolved_name, alias_resolved))

        # ---- Phase 3: merge decision per group ----
        for _key, members in sorted(groups.items()):
            group_clusters = [m[0] for m in members]
            any_alias = any(m[2] for m in members)

            merge_decision, confidence, reasons = _decide_group_merge(group_clusters)
            if any_alias and merge_decision == "AUTO_MERGE":
                reasons = ["alias_resolution"] + reasons

            # Best canonical name
            if any_alias:
                alias_names = [m[1] for m in members if m[2]]
                best_name_raw = alias_names[0]
            else:
                best_name_raw = max(
                    (m[1] for m in members),
                    key=lambda n: sum(
                        cl.mention_count for cl, rn, _ in members
                        if _norm(_strip_title(rn)[0] or rn) == _norm(_strip_title(n)[0] or n)
                    ),
                )
            best_base, _ = _strip_title(best_name_raw)
            canonical_name = best_base or best_name_raw

            type_votes = [cl.cluster_type for cl in group_clusters]
            dominant_type = max(set(type_votes), key=lambda t: type_votes.count(t))
            alias_set = sorted({n for cl, _, _ in members
                                for n in cl.proper_names + cl.common_names})

            if merge_decision == "AUTO_MERGE":
                cid = _make_canonical_id(canonical_name, used_ids)
                used_ids.add(cid)
                canon_chars.append(CanonicalCharacter(
                    canonical_id=cid,
                    name=canonical_name,
                    type=dominant_type,
                    aliases=alias_set,
                    source_clusters=[
                        {"chapter_id": cl.chapter_id, "coref_id": cl.coref_id}
                        for cl in group_clusters
                    ],
                    confidence=confidence,
                ))
                for cl in group_clusters:
                    decisions.append(IdentityDecision(
                        chapter_id=cl.chapter_id,
                        coref_id=cl.coref_id,
                        surface=cl.display_name,
                        target_canonical_id=cid,
                        decision="AUTO_MERGE",
                        confidence=confidence,
                        reasons=reasons,
                    ))

            else:  # REVIEW — one canonical per cluster, all flagged
                for cl, rname, _ in members:
                    ind_base, _ = _strip_title(rname)
                    ind_name = ind_base or rname
                    ind_id = _make_canonical_id(ind_name, used_ids)
                    used_ids.add(ind_id)
                    canon_chars.append(CanonicalCharacter(
                        canonical_id=ind_id,
                        name=ind_name,
                        type=cl.cluster_type,
                        aliases=sorted(set(cl.proper_names + cl.common_names)),
                        source_clusters=[{"chapter_id": cl.chapter_id, "coref_id": cl.coref_id}],
                        confidence=confidence,
                    ))
                    decisions.append(IdentityDecision(
                        chapter_id=cl.chapter_id,
                        coref_id=cl.coref_id,
                        surface=cl.display_name,
                        target_canonical_id=ind_id,
                        decision="REVIEW",
                        confidence=confidence,
                        reasons=reasons,
                    ))
                    unresolved.append({
                        "chapter_id":     cl.chapter_id,
                        "local_coref_id": cl.coref_id,
                        "surface":        cl.display_name,
                        "type":           cl.cluster_type,
                        "reason":         "; ".join(reasons),
                        "candidate_targets": [
                            _norm(_strip_title(m[1])[0] or m[1])
                            for m in members
                            if m[0] is not cl
                        ],
                    })

        return decisions, canon_chars, unresolved

    # ------------------------------------------------------------------
    def _build_mapping(self, decisions: List[IdentityDecision]) -> Dict[str, Dict[str, str]]:
        """local_coref_to_character: only AUTO_MERGE / REVIEW with a target."""
        mapping: Dict[str, Dict[str, str]] = {}
        for d in decisions:
            if d.target_canonical_id is None or d.decision not in ("AUTO_MERGE", "REVIEW"):
                continue
            mapping.setdefault(str(d.chapter_id), {})[d.coref_id] = d.target_canonical_id
        return mapping

    # ------------------------------------------------------------------
    def _build_report(
        self,
        all_clusters: List[LocalCluster],
        decisions:    List[IdentityDecision],
        canon_chars:  List[CanonicalCharacter],
    ) -> Dict[str, Any]:
        counts: Dict[str, int] = {k: 0 for k in ("AUTO_MERGE", "REJECT", "REVIEW", "ABSTAIN")}
        for d in decisions:
            counts[d.decision] = counts.get(d.decision, 0) + 1

        type_counts: Dict[str, int] = {}
        for cl in all_clusters:
            type_counts[cl.cluster_type] = type_counts.get(cl.cluster_type, 0) + 1

        warnings: List[str] = []
        if counts["REVIEW"] > 0:
            warnings.append(
                f"{counts['REVIEW']} REVIEW clusters — check identity_decisions.json "
                "and unresolved_entities.json"
            )

        return {
            "book_id": self.booknlp_root.name,
            "identity_mode": self.identity_mode,
            "manual_aliases_used": self.alias_file is not None,
            "manual_alias_file": str(self.alias_file) if self.alias_file else None,
            "review_policy": "review_and_ambiguous_entities_excluded_from_graph_by_default",
            "local_clusters_read": len(all_clusters),
            "clusters_kept":       counts["AUTO_MERGE"] + counts["REVIEW"],
            "clusters_discarded":  counts["REJECT"] + counts["ABSTAIN"],
            "canonical_characters": len(canon_chars),
            "decision_counts":     counts,
            "category_counts":     type_counts,
            "warnings":            warnings,
        }

    # ------------------------------------------------------------------
    def _write_outputs(
        self,
        canon_chars: List[CanonicalCharacter],
        mapping:     Dict[str, Dict[str, str]],
        decisions:   List[IdentityDecision],
        unresolved:  List[Dict[str, Any]],
        report:      Dict[str, Any],
    ) -> None:
        root = self.booknlp_root

        def _w(name: str, obj: Any) -> None:
            p = root / name
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            n = len(obj) if hasattr(obj, "__len__") else "?"
            logger.info("Wrote %-45s  (%s items)", p.name, n)

        _w("canonical_characters.json",
           [asdict(c) for c in sorted(canon_chars, key=lambda c: c.canonical_id)])
        _w("local_coref_to_character.json",
           {ch: dict(sorted(m.items())) for ch, m in sorted(mapping.items())})
        _w("identity_decisions.json",
           [asdict(d) for d in sorted(decisions, key=lambda d: (d.chapter_id, d.coref_id))])
        _w("unresolved_entities.json",
           sorted(unresolved, key=lambda x: (x["chapter_id"], x["local_coref_id"])))
        _w("identity_report.json", report)


# ---------------------------------------------------------------------------
# Public helpers for downstream steps
# ---------------------------------------------------------------------------

def load_canonical_characters(booknlp_root: Path) -> List[Dict[str, Any]]:
    p = booknlp_root / "canonical_characters.json"
    if not p.exists():
        raise FileNotFoundError(
            f"canonical_characters.json not found in {booknlp_root}. "
            "Run step2b_character_identity first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_coref_mapping(booknlp_root: Path) -> Dict[str, Dict[str, str]]:
    """Return {chapter_id_str: {local_coref_id_str: canonical_id}}."""
    p = booknlp_root / "local_coref_to_character.json"
    if not p.exists():
        raise FileNotFoundError(
            f"local_coref_to_character.json not found in {booknlp_root}. "
            "Run step2b_character_identity first."
        )
    return json.loads(p.read_text(encoding="utf-8"))
