"""Identity audit script for NLP-Graph pipeline runs.

Usage:
    python scripts/make_identity_audit.py --run-dir outputs/runs/<run_id>

Reads existing run outputs. Does NOT modify pipeline data, graphs, or BERT files.
Writes diagnostic reports only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Thresholds (documented here for easy tuning)
# ---------------------------------------------------------------------------

# canonical_per_10k_words threshold for HIGH_RISK_IDENTITY_NOISE
CANONICAL_PER_10K_HIGH_RISK = 20.0

# minimum suspicious canonical count to trigger HIGH_RISK_IDENTITY_NOISE
SUSPICIOUS_CANONICAL_HIGH_RISK = 5

# review_per_canonical threshold for HIGH_RISK_UNDERMERGE_OR_REVIEW
REVIEW_PER_CANONICAL_HIGH_RISK = 10.0

# unresolved_per_canonical threshold for HIGH_RISK_UNDERMERGE_OR_REVIEW
UNRESOLVED_PER_CANONICAL_HIGH_RISK = 20.0

# mapped_speaker_ratio below which LOW_SPEAKER_MAPPING is flagged
MAPPED_SPEAKER_RATIO_LOW = 0.25

# max mapped mentions before "low mention" flag
LOW_MENTION_THRESHOLD = 2

# minimum mention count for a REVIEW/ABSTAIN cluster to be "high salience"
HIGH_SALIENCE_MIN_MENTIONS = 10

# minimum quote count for a speaker to be "high salience" even without many mentions
HIGH_SALIENCE_MIN_QUOTES = 5

# ---------------------------------------------------------------------------
# Suspicious name lexicon and patterns
# ---------------------------------------------------------------------------

# Lower-cased words/phrases that are clearly not person names
_ABSTRACT_NAMES: frozenset[str] = frozenset({
    "ivory", "adieu", "jove", "god", "good god", "lord",
    "heaven", "hell", "devil", "fate", "fortune", "duty",
    "darkness", "light", "silence", "death", "life", "nature",
    "glory", "honor", "honour", "truth", "beauty", "time",
    "chapter", "prologue", "epilogue", "preface", "introduction",
    "farewell", "amen", "hurrah", "huzza", "alas", "encore",
    "voila", "voilà", "bravo",
})

# Regex: name starts with "the " followed by a national/group noun
_GROUP_PATTERN = re.compile(
    r"^the\s+(french|english|english|spanish|german|russian|american|"
    r"italian|greek|polish|dutch|swedish|danish|turkish|chinese|"
    r"japanese|company|crowd|men|women|mob|natives|savages|crew|"
    r"captain|manager|lawyer|doctor|priest|painter|helmsman|nurse|"
    r"butler|maid|servant|peasant|soldier|guard|villager|villagers|"
    r"bystander|bystanders|audience|public|government|authority|"
    r"committee|board|council|senate|tribe|clan|horde|party|group|"
    r"swede|dane|pole|finn|norwegian|austrian|belgian|portuguese|"
    r"hungarian|romanian|bulgarian|serbian|croatian)\b",
    re.IGNORECASE,
)

# Long alias fragments (> N chars) suggest bad alias contamination
_BAD_ALIAS_MIN_LEN = 60

# Regex to detect punctuation-heavy alias fragments
_PUNCTUATION_HEAVY = re.compile(r"[,;:\"'?!]{2,}|\.{2,}")


def _is_suspicious_name(name: str, aliases: list[str]) -> list[str]:
    """Return list of suspicion reasons for a canonical character name."""
    reasons: list[str] = []
    name_l = name.strip().lower()

    if name_l in _ABSTRACT_NAMES:
        reasons.append(f"abstract/exclamation name: {name!r}")

    if _GROUP_PATTERN.match(name_l):
        reasons.append(f"group/nationality pattern: {name!r}")

    if len(name) > _BAD_ALIAS_MIN_LEN:
        reasons.append(f"name too long ({len(name)} chars) — likely bad alias")

    for alias in aliases:
        if len(alias) > _BAD_ALIAS_MIN_LEN:
            reasons.append(f"long alias: {alias[:70]!r}…")
            break
        if _PUNCTUATION_HEAVY.search(alias):
            reasons.append(f"punctuation-heavy alias: {alias[:60]!r}")
            break

    return reasons


# ---------------------------------------------------------------------------
# Role-surface heuristics (for high-salience excluded roles)
# ---------------------------------------------------------------------------

_ROLE_SURFACE_PATTERN = re.compile(
    r"\b(manager|lawyer|doctor|physician|priest|painter|helmsman|"
    r"captain|colonel|general|sergeant|lieutenant|inspector|detective|"
    r"butler|maid|servant|housekeeper|nurse|cook|gardener|driver|"
    r"secretary|assistant|clerk|attendant|guard|officer|agent|chief|"
    r"director|chairman|commissioner|professor|teacher|judge|magistrate|"
    r"emperor|king|queen|prince|princess|duke|duchess|baron|count|"
    r"the\s+man|the\s+woman|the\s+girl|the\s+boy|the\s+child|"
    r"old\s+man|young\s+man|old\s+woman)\b",
    re.IGNORECASE,
)


def _looks_like_role(surface: str) -> bool:
    return bool(_ROLE_SURFACE_PATTERN.search(surface))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entities file parsing
# ---------------------------------------------------------------------------

def _read_entities(entities_path: Path) -> list[dict]:
    """Parse a BookNLP .entities TSV file into list of dicts."""
    if not entities_path.exists():
        return []
    rows: list[dict] = []
    try:
        lines = entities_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))
    except Exception:
        pass
    return rows


def _read_quotes_file(quotes_path: Path) -> list[dict]:
    """Parse a BookNLP .quotes TSV file into list of dicts."""
    if not quotes_path.exists():
        return []
    rows: list[dict] = []
    try:
        lines = quotes_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t", len(header) - 1)
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Mention-type stats per canonical character
# ---------------------------------------------------------------------------

def _compute_mention_stats(
    chapter_dirs: list[Path],
    coref_map: dict[str, dict[str, str]],
) -> dict[str, dict]:
    """Return per canonical_id mention stats (total/proper/nominal/pronoun).

    coref_map: {chapter_id_str: {coref_id_str: canonical_id}}
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"mapped_mentions": 0, "proper_mentions": 0,
                 "nominal_mentions": 0, "pronoun_mentions": 0}
    )

    for chapter_dir in chapter_dirs:
        chapter_id_str = _chapter_id_from_dir(chapter_dir)
        chapter_coref = coref_map.get(chapter_id_str, {})

        entities_files = list(chapter_dir.glob("*.entities"))
        if not entities_files:
            continue
        entities = _read_entities(entities_files[0])

        for row in entities:
            coref_id = row.get("COREF", "").strip()
            canonical_id = chapter_coref.get(coref_id)
            if not canonical_id:
                continue

            prop = row.get("prop", "").strip().upper()
            stats[canonical_id]["mapped_mentions"] += 1
            if prop == "PROP":
                stats[canonical_id]["proper_mentions"] += 1
            elif prop == "NOM":
                stats[canonical_id]["nominal_mentions"] += 1
            elif prop == "PRON":
                stats[canonical_id]["pronoun_mentions"] += 1

    return dict(stats)


def _chapter_id_from_dir(chapter_dir: Path) -> str:
    """Extract zero-padded chapter number from directory name as string."""
    name = chapter_dir.name  # e.g. booknlp_gatsby_chapter_0003
    m = re.search(r"chapter_(\d+)$", name)
    if m:
        return str(int(m.group(1)))  # strip leading zeros → "3"
    return name


# ---------------------------------------------------------------------------
# High-salience excluded roles
# ---------------------------------------------------------------------------

def _compute_high_salience_excluded(
    decisions: list[dict],
    chapter_dirs: list[Path],
) -> list[dict]:
    """Identify REVIEW/ABSTAIN clusters with high mention or speaker counts."""
    # Index chapter dirs by chapter_id string
    chapter_dir_map: dict[str, Path] = {}
    for d in chapter_dirs:
        cid = _chapter_id_from_dir(d)
        chapter_dir_map[cid] = d

    # Count entity mentions per (chapter_id, coref_id) for excluded clusters
    excluded: list[dict] = []

    for dec in decisions:
        decision = dec.get("decision", "")
        if decision not in ("REVIEW", "ABSTAIN"):
            continue

        chapter_id = str(dec.get("chapter_id", ""))
        coref_id = str(dec.get("coref_id", ""))
        surface = dec.get("surface", "")
        reasons = dec.get("reasons", [])

        chapter_dir = chapter_dir_map.get(chapter_id)
        mention_count = 0
        sample_mentions: list[str] = []
        quote_speaker_count = 0

        if chapter_dir:
            ent_files = list(chapter_dir.glob("*.entities"))
            if ent_files:
                entities = _read_entities(ent_files[0])
                for row in entities:
                    if row.get("COREF", "").strip() == coref_id:
                        mention_count += 1
                        text = row.get("text", "").strip()
                        if text and len(sample_mentions) < 3:
                            sample_mentions.append(text)

            qt_files = list(chapter_dir.glob("*.quotes"))
            if qt_files:
                quotes = _read_quotes_file(qt_files[0])
                for q in quotes:
                    if str(q.get("char_id", "")).strip() == coref_id:
                        quote_speaker_count += 1

        is_high_salience = (
            mention_count >= HIGH_SALIENCE_MIN_MENTIONS
            or quote_speaker_count >= HIGH_SALIENCE_MIN_QUOTES
        )

        if not is_high_salience:
            continue

        why_flagged: list[str] = []
        if mention_count >= HIGH_SALIENCE_MIN_MENTIONS:
            why_flagged.append(f"high mention count ({mention_count})")
        if quote_speaker_count >= HIGH_SALIENCE_MIN_QUOTES:
            why_flagged.append(f"high quote speaker count ({quote_speaker_count})")
        if _looks_like_role(surface):
            why_flagged.append("surface looks like a role/person descriptor")

        excluded.append({
            "chapter_id": chapter_id,
            "coref_id": coref_id,
            "surface": surface,
            "decision": decision,
            "reasons": reasons,
            "mention_count": mention_count,
            "quote_speaker_count": quote_speaker_count,
            "sample_mentions": sample_mentions,
            "why_flagged": why_flagged,
        })

    excluded.sort(key=lambda x: -x["mention_count"])
    return excluded


# ---------------------------------------------------------------------------
# Speaker mapping aggregation
# ---------------------------------------------------------------------------

def _aggregate_speaker_mapping(quote_diag: dict[str, Any]) -> dict:
    """Aggregate quote_evidence_diagnostics_by_chapter into book-level stats."""
    if not quote_diag:
        return {
            "total_quotes": 0,
            "mapped_speaker_quotes": 0,
            "unmapped_speaker_quotes": 0,
            "mapped_speaker_ratio": None,
            "top_unmapped_speakers": [],
        }

    total = 0
    mapped = 0
    unmapped_counter: dict[str, int] = defaultdict(int)

    for _ch, ch_data in quote_diag.items():
        total += int(ch_data.get("quotes_total", 0))
        mapped += int(ch_data.get("mapped_speaker_quotes", 0))
        for entry in ch_data.get("top_unmapped_speakers", []):
            key = entry.get("mention_phrase", entry.get("coref_id", "?"))
            unmapped_counter[key] += int(entry.get("quotes", 0))

    unmapped = total - mapped
    ratio = round(mapped / total, 4) if total > 0 else None

    top_unmapped = sorted(
        [{"phrase": k, "count": v} for k, v in unmapped_counter.items()],
        key=lambda x: -x["count"],
    )[:10]

    return {
        "total_quotes": total,
        "mapped_speaker_quotes": mapped,
        "unmapped_speaker_quotes": unmapped,
        "mapped_speaker_ratio": ratio,
        "top_unmapped_speakers": top_unmapped,
    }


# ---------------------------------------------------------------------------
# Evidence participation per canonical character
# ---------------------------------------------------------------------------

def _compute_evidence_participation(scored: dict[str, Any]) -> dict[str, dict]:
    """Return per canonical_id evidence participation stats from scored pairs."""
    participation: dict[str, dict] = defaultdict(
        lambda: {
            "total_pair_participation": 0,
            "strong_pair_participation": 0,
            "direct_event_participation": 0,
            "dialogue_turn_participation": 0,
            "quote_about_participation": 0,
            "co_presence_only_participation": 0,
        }
    )

    for _chapter_id, pairs in scored.items():
        for pair_key, score in pairs.items():
            parts = pair_key.split("||")
            if len(parts) != 2:
                continue
            char_ids = parts

            de = int(score.get("direct_event_count", 0) or 0)
            dt = int(score.get("dialogue_turn_count", 0) or 0)
            qa = int(score.get("quote_about_count", 0) or 0)
            strong = int(score.get("strong_evidence_count", de + dt + qa) or 0)
            co = int(score.get("co_presence_count", 0) or 0)

            for cid in char_ids:
                p = participation[cid]
                p["total_pair_participation"] += 1
                if strong > 0:
                    p["strong_pair_participation"] += 1
                if de > 0:
                    p["direct_event_participation"] += 1
                if dt > 0:
                    p["dialogue_turn_participation"] += 1
                if qa > 0:
                    p["quote_about_participation"] += 1
                if strong == 0 and co > 0:
                    p["co_presence_only_participation"] += 1

    return dict(participation)


# ---------------------------------------------------------------------------
# Suspicious canonical character analysis
# ---------------------------------------------------------------------------

def _analyse_suspicious_canonicals(
    canonicals: list[dict],
    mention_stats: dict[str, dict],
    participation: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Return (suspicious_list, low_mention_list)."""
    suspicious: list[dict] = []
    low_mention: list[dict] = []

    for char in canonicals:
        cid = char.get("canonical_id", "")
        name = char.get("name", "")
        aliases = char.get("aliases") or []

        mstats = mention_stats.get(cid, {})
        pstats = participation.get(cid, {})

        mapped_mentions = mstats.get("mapped_mentions", 0)
        proper_mentions = mstats.get("proper_mentions", 0)
        nominal_mentions = mstats.get("nominal_mentions", 0)
        pronoun_mentions = mstats.get("pronoun_mentions", 0)
        strong_pairs = pstats.get("strong_pair_participation", 0)

        suspicion_reasons: list[str] = _is_suspicious_name(name, aliases)

        # Low-mention
        if mapped_mentions <= LOW_MENTION_THRESHOLD and strong_pairs == 0:
            suspicion_reasons.append(
                f"very low mapped mentions ({mapped_mentions}) with no strong evidence"
            )

        # Pronoun-dominated
        if mapped_mentions >= 3 and pronoun_mentions > 0:
            pronoun_ratio = pronoun_mentions / mapped_mentions
            proper_ratio = (proper_mentions + nominal_mentions) / mapped_mentions
            if pronoun_ratio >= 0.80 and proper_ratio < 0.10:
                suspicion_reasons.append(
                    f"pronoun-dominated cluster ({pronoun_mentions}/{mapped_mentions} pronouns, "
                    f"only {proper_mentions} proper+nominal)"
                )

        record = {
            "canonical_id": cid,
            "name": name,
            "type": char.get("type", ""),
            "mapped_mentions": mapped_mentions,
            "proper_mentions": proper_mentions,
            "nominal_mentions": nominal_mentions,
            "pronoun_mentions": pronoun_mentions,
            "strong_pair_participation": strong_pairs,
            "suspicion_score": len(suspicion_reasons),
            "suspicion_reasons": suspicion_reasons,
            "aliases": aliases,
        }

        if suspicion_reasons:
            suspicious.append(record)

        # Low-mention list: ≤2 mapped mentions AND no strong evidence AND no quote evidence
        quote_ev = (
            pstats.get("dialogue_turn_participation", 0)
            + pstats.get("quote_about_participation", 0)
            + pstats.get("direct_event_participation", 0)
        )
        if mapped_mentions <= LOW_MENTION_THRESHOLD and strong_pairs == 0 and quote_ev == 0:
            low_mention.append({
                "canonical_id": cid,
                "name": name,
                "mapped_mentions": mapped_mentions,
                "total_pair_participation": pstats.get("total_pair_participation", 0),
            })

    suspicious.sort(key=lambda x: -x["suspicion_score"])
    return suspicious, low_mention


# ---------------------------------------------------------------------------
# Audit status logic
# ---------------------------------------------------------------------------

def _audit_status(
    *,
    has_canonical: bool,
    has_identity_report: bool,
    canonical_per_10k: float,
    suspicious_count: int,
    review_per_canonical: float,
    unresolved_per_canonical: float,
    mapped_speaker_ratio: float | None,
    warnings: list[str],
) -> str:
    if not has_canonical or not has_identity_report:
        return "MISSING_IDENTITY_OUTPUT"

    if (
        canonical_per_10k >= CANONICAL_PER_10K_HIGH_RISK
        or suspicious_count >= SUSPICIOUS_CANONICAL_HIGH_RISK
    ):
        return "HIGH_RISK_IDENTITY_NOISE"

    if (
        review_per_canonical >= REVIEW_PER_CANONICAL_HIGH_RISK
        or unresolved_per_canonical >= UNRESOLVED_PER_CANONICAL_HIGH_RISK
    ):
        return "HIGH_RISK_UNDERMERGE_OR_REVIEW"

    if mapped_speaker_ratio is not None and mapped_speaker_ratio < MAPPED_SPEAKER_RATIO_LOW:
        return "LOW_SPEAKER_MAPPING"

    if warnings:
        return "WATCHLIST"

    return "HEALTHY_OR_ACCEPTABLE"


# ---------------------------------------------------------------------------
# Per-book audit
# ---------------------------------------------------------------------------

def _audit_book(
    book_id: str,
    run_dir: Path,
) -> tuple[dict, list[dict]]:
    """Build per-book audit dict and suspicious character rows for global CSV."""
    booknlp_root = run_dir / "booknlp_chapter_output" / book_id
    reports_root = run_dir / "reports" / book_id
    books_root = run_dir / "books" / book_id
    ml_root = run_dir / "ml" / book_id

    # Discover chapter directories
    chapter_dirs = sorted(
        booknlp_root.glob("booknlp_*_chapter_*"),
        key=lambda d: _chapter_id_from_dir(d),
    )

    # Load main inputs
    canonicals: list[dict] = _load_json(booknlp_root / "canonical_characters.json", default=[])
    coref_map_raw: dict = _load_json(booknlp_root / "local_coref_to_character.json", default={})
    identity_report: dict = _load_json(booknlp_root / "identity_report.json", default={})
    decisions: list[dict] = _load_json(booknlp_root / "identity_decisions.json", default=[])
    unresolved: list[dict] = _load_json(booknlp_root / "unresolved_entities.json", default=[])
    scored: dict = _load_json(booknlp_root / "scored_pair_evidence_by_chapter.json", default={})
    quote_diag: dict = _load_json(booknlp_root / "quote_evidence_diagnostics_by_chapter.json", default={})
    quality: dict = _load_json(reports_root / "quality_report.json", default={})
    bert_summary: dict = _load_json(ml_root / "bert_relation_summary.json", default={})
    chapters_meta: list = _load_json(books_root / "chapters" / "chapters.json", default=[])

    # coref_map: {chapter_id_str: {coref_id_str: canonical_id}}
    coref_map: dict[str, dict[str, str]] = {
        str(ch_id): {str(cid): cname for cid, cname in ch_map.items()}
        for ch_id, ch_map in coref_map_raw.items()
    }

    # Word count
    total_words = sum(int(c.get("num_words", 0)) for c in chapters_meta)

    # Identity report fields
    canonical_count = int(identity_report.get("canonical_characters", len(canonicals)))
    local_clusters_read = int(identity_report.get("local_clusters_read", 0))
    decision_counts: dict = identity_report.get("decision_counts", {})
    review_count = int(decision_counts.get("REVIEW", 0))
    abstain_count = int(decision_counts.get("ABSTAIN", 0))
    reject_count = int(decision_counts.get("REJECT", 0))
    auto_merge_count = int(decision_counts.get("AUTO_MERGE", 0))
    unresolved_count = len(unresolved)

    review_per_canonical = (review_count / canonical_count) if canonical_count else 0.0
    unresolved_per_canonical = (unresolved_count / canonical_count) if canonical_count else 0.0
    canonical_per_10k = (canonical_count / total_words * 10000) if total_words else 0.0

    # Interaction stats
    interactions = quality.get("interactions", {})
    total_pairs = int(interactions.get("total_pairs", 0))
    co_only_pairs = int(interactions.get("co_presence_only_pairs", 0))

    # Strong evidence pairs: prefer quality_report, else compute from scored
    strong_pairs = int(interactions.get("strong_evidence_pairs", 0))
    if strong_pairs == 0 and scored:
        strong_pairs = sum(
            1
            for chap in scored.values()
            for s in chap.values()
            if int(s.get("strong_evidence_count",
                         int(s.get("direct_event_count", 0))
                         + int(s.get("dialogue_turn_count", 0))
                         + int(s.get("quote_about_count", 0))) or 0) > 0
        )

    bert_examples = int(bert_summary.get("example_count", 0))

    # Mention stats (requires reading .entities files)
    mention_stats = _compute_mention_stats(chapter_dirs, coref_map)

    # Evidence participation
    participation = _compute_evidence_participation(scored)

    # Suspicious canonicals
    suspicious, low_mention = _analyse_suspicious_canonicals(
        canonicals, mention_stats, participation
    )

    # High-salience excluded roles
    high_salience_excluded = _compute_high_salience_excluded(decisions, chapter_dirs)

    # Speaker mapping
    speaker_mapping = _aggregate_speaker_mapping(quote_diag)
    mapped_speaker_ratio = speaker_mapping.get("mapped_speaker_ratio")

    # Build warnings list
    audit_warnings: list[str] = []

    if canonical_per_10k >= CANONICAL_PER_10K_HIGH_RISK:
        audit_warnings.append(
            f"High canonical density: {canonical_per_10k:.1f} per 10k words "
            f"(threshold={CANONICAL_PER_10K_HIGH_RISK})"
        )

    if len(suspicious) >= SUSPICIOUS_CANONICAL_HIGH_RISK:
        audit_warnings.append(
            f"{len(suspicious)} suspicious canonical characters detected"
        )

    if review_per_canonical >= REVIEW_PER_CANONICAL_HIGH_RISK:
        audit_warnings.append(
            f"High review ratio: {review_count} REVIEW / {canonical_count} canonical "
            f"= {review_per_canonical:.1f}x (threshold={REVIEW_PER_CANONICAL_HIGH_RISK})"
        )

    if unresolved_per_canonical >= UNRESOLVED_PER_CANONICAL_HIGH_RISK:
        audit_warnings.append(
            f"High unresolved ratio: {unresolved_count} unresolved / {canonical_count} canonical "
            f"= {unresolved_per_canonical:.1f}x"
        )

    if mapped_speaker_ratio is not None and mapped_speaker_ratio < MAPPED_SPEAKER_RATIO_LOW:
        audit_warnings.append(
            f"Low speaker mapping ratio: {mapped_speaker_ratio:.2%} "
            f"(threshold={MAPPED_SPEAKER_RATIO_LOW:.0%})"
        )

    if low_mention:
        audit_warnings.append(
            f"{len(low_mention)} canonical characters with ≤{LOW_MENTION_THRESHOLD} "
            f"mapped mentions and no evidence"
        )

    if high_salience_excluded:
        audit_warnings.append(
            f"{len(high_salience_excluded)} high-salience excluded roles "
            f"(REVIEW/ABSTAIN with high mentions or quote count)"
        )

    status = _audit_status(
        has_canonical=bool(canonicals),
        has_identity_report=bool(identity_report),
        canonical_per_10k=canonical_per_10k,
        suspicious_count=len(suspicious),
        review_per_canonical=review_per_canonical,
        unresolved_per_canonical=unresolved_per_canonical,
        mapped_speaker_ratio=mapped_speaker_ratio,
        warnings=audit_warnings,
    )

    audit = {
        "book_id": book_id,
        "summary": {
            "chapter_count": len(chapters_meta) or len(chapter_dirs),
            "total_words": total_words,
            "canonical_characters": canonical_count,
            "canonical_per_10k_words": round(canonical_per_10k, 2),
            "local_clusters_read": local_clusters_read,
            "unresolved_entities": unresolved_count,
            "review_clusters": review_count,
            "abstain_clusters": abstain_count,
            "reject_clusters": reject_count,
            "auto_merge_clusters": auto_merge_count,
            "review_per_canonical": round(review_per_canonical, 2),
            "unresolved_per_canonical": round(unresolved_per_canonical, 2),
            "suspicious_canonical_count": len(suspicious),
            "low_mention_canonical_count": len(low_mention),
            "high_salience_excluded_role_count": len(high_salience_excluded),
            "total_pairs": total_pairs,
            "strong_evidence_pairs": strong_pairs,
            "co_presence_only_pairs": co_only_pairs,
            "bert_examples": bert_examples,
        },
        "audit_status": status,
        "warnings": audit_warnings,
        "suspicious_canonical_characters": suspicious[:30],
        "low_mention_canonical_characters": low_mention[:30],
        "high_salience_excluded_roles": high_salience_excluded[:20],
        "speaker_mapping": speaker_mapping,
        "evidence_participation": {
            cid: p for cid, p in sorted(
                participation.items(),
                key=lambda x: -x[1]["strong_pair_participation"],
            )[:30]
        },
        "notes": _generate_notes(
            book_id=book_id,
            canonical_per_10k=canonical_per_10k,
            mapped_speaker_ratio=mapped_speaker_ratio,
            suspicious=suspicious,
            high_salience_excluded=high_salience_excluded,
            low_mention=low_mention,
            bert_examples=bert_examples,
        ),
    }

    # Build suspicious character rows for global CSV
    suspicious_rows = [
        {
            "book_id": book_id,
            "canonical_id": s["canonical_id"],
            "name": s["name"],
            "type": s["type"],
            "mapped_mentions": s["mapped_mentions"],
            "proper_mentions": s["proper_mentions"],
            "nominal_mentions": s["nominal_mentions"],
            "pronoun_mentions": s["pronoun_mentions"],
            "strong_pair_participation": s["strong_pair_participation"],
            "suspicion_score": s["suspicion_score"],
            "suspicion_reasons": " | ".join(s["suspicion_reasons"]),
            "aliases": " | ".join(str(a) for a in s["aliases"][:5]),
        }
        for s in suspicious
    ]

    return audit, suspicious_rows


def _generate_notes(
    *,
    book_id: str,
    canonical_per_10k: float,
    mapped_speaker_ratio: float | None,
    suspicious: list[dict],
    high_salience_excluded: list[dict],
    low_mention: list[dict],
    bert_examples: int,
) -> list[str]:
    notes: list[str] = []

    if bert_examples == 0:
        notes.append(
            "No BERT examples generated. "
            "Likely cause: very low mapped speaker ratio and/or few strong evidence pairs."
        )

    if mapped_speaker_ratio is not None and mapped_speaker_ratio < 0.15:
        notes.append(
            f"Extremely low mapped speaker ratio ({mapped_speaker_ratio:.2%}). "
            "Most quotes are attributed to pronoun/anonymous clusters. "
            "Possible fix: promote narrator cluster or improve entity resolution."
        )
    elif mapped_speaker_ratio is not None and mapped_speaker_ratio < MAPPED_SPEAKER_RATIO_LOW:
        notes.append(
            f"Low mapped speaker ratio ({mapped_speaker_ratio:.2%}). "
            "Many quotes cannot be attributed to canonical characters."
        )

    if len(high_salience_excluded) >= 3:
        surfaces = [r["surface"] for r in high_salience_excluded[:3]]
        notes.append(
            f"{len(high_salience_excluded)} high-salience roles excluded from graph: "
            f"{surfaces}. These may be important secondary characters. "
            "Consider reviewing REVIEW decisions or using human_refined mode."
        )

    if len(low_mention) >= 5:
        notes.append(
            f"{len(low_mention)} canonical characters have ≤{LOW_MENTION_THRESHOLD} "
            "mapped mentions and no evidence participation. "
            "Likely spurious promotions from single-mention clusters."
        )

    if canonical_per_10k >= CANONICAL_PER_10K_HIGH_RISK:
        notes.append(
            f"Very high canonical character density ({canonical_per_10k:.1f}/10k words). "
            "Identity layer is likely over-promoting low-frequency clusters. "
            "Raising the minimum-mention threshold would reduce noise."
        )

    return notes


# ---------------------------------------------------------------------------
# Global CSV row builder
# ---------------------------------------------------------------------------

GLOBAL_CSV_COLUMNS = [
    "book_id",
    "chapter_count",
    "total_words",
    "canonical_characters",
    "canonical_per_10k_words",
    "local_clusters_read",
    "unresolved_entities",
    "review_clusters",
    "abstain_clusters",
    "reject_clusters",
    "auto_merge_clusters",
    "unresolved_per_canonical",
    "review_per_canonical",
    "suspicious_canonical_count",
    "low_mention_canonical_count",
    "high_salience_excluded_role_count",
    "mapped_speaker_ratio",
    "total_pairs",
    "strong_evidence_pairs",
    "co_presence_only_pairs",
    "bert_examples",
    "audit_status",
    "top_warnings",
]

SUSPICIOUS_CSV_COLUMNS = [
    "book_id",
    "canonical_id",
    "name",
    "type",
    "mapped_mentions",
    "proper_mentions",
    "nominal_mentions",
    "pronoun_mentions",
    "strong_pair_participation",
    "suspicion_score",
    "suspicion_reasons",
    "aliases",
]


def _audit_to_global_row(audit: dict) -> dict:
    s = audit["summary"]
    sm = audit.get("speaker_mapping", {})
    warnings = audit.get("warnings", [])

    return {
        "book_id": audit["book_id"],
        "chapter_count": s.get("chapter_count", ""),
        "total_words": s.get("total_words", ""),
        "canonical_characters": s.get("canonical_characters", ""),
        "canonical_per_10k_words": s.get("canonical_per_10k_words", ""),
        "local_clusters_read": s.get("local_clusters_read", ""),
        "unresolved_entities": s.get("unresolved_entities", ""),
        "review_clusters": s.get("review_clusters", ""),
        "abstain_clusters": s.get("abstain_clusters", ""),
        "reject_clusters": s.get("reject_clusters", ""),
        "auto_merge_clusters": s.get("auto_merge_clusters", ""),
        "unresolved_per_canonical": s.get("unresolved_per_canonical", ""),
        "review_per_canonical": s.get("review_per_canonical", ""),
        "suspicious_canonical_count": s.get("suspicious_canonical_count", ""),
        "low_mention_canonical_count": s.get("low_mention_canonical_count", ""),
        "high_salience_excluded_role_count": s.get("high_salience_excluded_role_count", ""),
        "mapped_speaker_ratio": sm.get("mapped_speaker_ratio", ""),
        "total_pairs": s.get("total_pairs", ""),
        "strong_evidence_pairs": s.get("strong_evidence_pairs", ""),
        "co_presence_only_pairs": s.get("co_presence_only_pairs", ""),
        "bert_examples": s.get("bert_examples", ""),
        "audit_status": audit.get("audit_status", ""),
        "top_warnings": " | ".join(warnings[:3]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identity audit for an NLP-Graph pipeline run directory."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory, e.g. outputs/runs/baseline_auto_7books_quote_v1",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    reports_root = run_dir / "reports"

    # Discover book IDs
    book_ids: list[str] = []
    if reports_root.exists():
        book_ids = sorted(p.name for p in reports_root.iterdir() if p.is_dir())
    if not book_ids:
        print(f"No book directories found under {reports_root}")
        return

    global_rows: list[dict] = []
    all_suspicious_rows: list[dict] = []

    for book_id in book_ids:
        print(f"Auditing {book_id}…")
        audit, suspicious_rows = _audit_book(book_id, run_dir)

        per_book_path = reports_root / book_id / "identity_audit.json"
        _write_json(per_book_path, audit)
        print(f"  → {per_book_path}  [{audit['audit_status']}]")

        global_rows.append(_audit_to_global_row(audit))
        all_suspicious_rows.extend(suspicious_rows)

    global_dir = reports_root / "_global"
    csv_path = global_dir / "identity_audit.csv"
    susp_path = global_dir / "identity_audit_suspicious_characters.csv"

    _write_csv(csv_path, global_rows, GLOBAL_CSV_COLUMNS)
    _write_csv(susp_path, all_suspicious_rows, SUSPICIOUS_CSV_COLUMNS)

    print(f"\nWrote global summary: {csv_path}")
    print(f"Wrote suspicious characters: {susp_path}")


if __name__ == "__main__":
    main()
