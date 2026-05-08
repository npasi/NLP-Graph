from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


TITLE_PREFIXES = [
    "mr.", "mrs.", "miss", "ms.", "dr.", "prof.", "sir", "lady",
    "lord", "captain", "colonel", "general", "uncle", "aunt",
]

GENERIC_WORDS = {
    "the", "a", "an", "young", "old", "poor", "dear", "little",
    "mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    s = re.sub(r"[’']", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s


def ascii_slug(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = norm(s)
    return s


def strip_title(text: str) -> str:
    s = norm(text)
    for t in sorted(TITLE_PREFIXES, key=len, reverse=True):
        if s == t:
            return s
        if s.startswith(t + " "):
            return s[len(t):].strip()
    return s


def tokens(text: str) -> list[str]:
    s = strip_title(text)
    s = re.sub(r"[^a-z0-9\s]", " ", ascii_slug(s))
    toks = [t for t in s.split() if t and t not in GENERIC_WORDS]
    return toks


def last_token(text: str) -> str:
    toks = tokens(text)
    return toks[-1] if toks else ""


def token_overlap(a: str, b: str) -> float:
    ta = set(tokens(a))
    tb = set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_subname(shorter: str, longer: str) -> bool:
    st = set(tokens(shorter))
    lt = set(tokens(longer))
    return bool(st) and st.issubset(lt) and st != lt


def possible_conflict(a: str, b: str) -> bool:
    na = norm(a)
    nb = norm(b)

    # Avoid obvious Mr/Mrs conflicts.
    if ("mr." in na and "mrs." in nb) or ("mrs." in na and "mr." in nb):
        return True

    if ("miss " in na and "mr." in nb) or ("miss " in nb and "mr." in na):
        return True

    return False


def candidate_surfaces(c: dict[str, Any]) -> list[str]:
    vals: list[str] = []
    for x in [c.get("name")] + list(c.get("aliases", []) or []):
        if isinstance(x, str) and x.strip():
            vals.append(x.strip())

    seen = set()
    out = []
    for v in vals:
        k = norm(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def best_name(c: dict[str, Any]) -> str:
    return str(c.get("name") or c.get("canonical_id"))


def score_pair(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, list[str]]:
    a_name = best_name(a)
    b_name = best_name(b)

    if possible_conflict(a_name, b_name):
        return 0.0, ["blocked: possible gender/title conflict"]

    a_surfaces = candidate_surfaces(a)
    b_surfaces = candidate_surfaces(b)

    best = 0.0
    reasons: list[str] = []

    for sa in a_surfaces:
        for sb in b_surfaces:
            if possible_conflict(sa, sb):
                continue

            nsa = strip_title(sa)
            nsb = strip_title(sb)

            if norm(nsa) == norm(nsb) and norm(nsa):
                best = max(best, 0.95)
                reasons.append(f"same normalized name: {sa!r} ≈ {sb!r}")

            if ascii_slug(nsa) == ascii_slug(nsb) and ascii_slug(nsa):
                best = max(best, 0.92)
                reasons.append(f"same ascii-normalized name: {sa!r} ≈ {sb!r}")

            if is_subname(sa, sb) or is_subname(sb, sa):
                best = max(best, 0.82)
                reasons.append(f"subname relation: {sa!r} / {sb!r}")

            if last_token(sa) and last_token(sa) == last_token(sb):
                best = max(best, 0.74)
                reasons.append(f"shared last token: {last_token(sa)!r}")

            overlap = token_overlap(sa, sb)
            if overlap >= 0.5:
                best = max(best, 0.65)
                reasons.append(f"token overlap {overlap:.2f}: {sa!r} / {sb!r}")

    # Same canonical type helps, but never alone.
    if best > 0 and a.get("type") == b.get("type"):
        best = min(0.99, best + 0.03)
        reasons.append(f"same type: {a.get('type')}")

    # More evidence/source clusters helps slightly.
    source_count = len(a.get("source_clusters", []) or []) + len(b.get("source_clusters", []) or [])
    if best > 0 and source_count >= 3:
        best = min(0.99, best + 0.02)
        reasons.append(f"multiple source clusters: {source_count}")

    # Deduplicate reasons while preserving order.
    seen = set()
    dedup_reasons = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            dedup_reasons.append(r)

    return round(best, 4), dedup_reasons


def build_suggestions(characters: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    chars = sorted(characters, key=lambda c: str(c.get("canonical_id")))

    for i, a in enumerate(chars):
        for b in chars[i + 1:]:
            score, reasons = score_pair(a, b)
            if score < threshold:
                continue

            # Pick target as the more informative / more frequent canonical.
            a_sources = len(a.get("source_clusters", []) or [])
            b_sources = len(b.get("source_clusters", []) or [])

            if b_sources > a_sources or len(best_name(b)) > len(best_name(a)):
                source = a
                target = b
            else:
                source = b
                target = a

            suggestions.append({
                "source_canonical_id": source.get("canonical_id"),
                "source_name": best_name(source),
                "target_canonical_id": target.get("canonical_id"),
                "target_name": best_name(target),
                "confidence": score,
                "reasons": reasons,
                "suggested_alias_entry": {
                    best_name(source): best_name(target)
                },
                "action": "review"
            })

    suggestions.sort(key=lambda x: (-x["confidence"], x["source_name"], x["target_name"]))
    return suggestions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Suggest alias merges for canonical characters.")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--booknlp-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args(argv)

    booknlp_root = Path(args.booknlp_root) if args.booknlp_root else (
        Path("data") / "booknlp_chapter_output" / args.book_id
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("data") / "reports" / args.book_id
    )

    characters = load_json(booknlp_root / "canonical_characters.json", default=[])
    suggestions = build_suggestions(characters, args.threshold)

    out = {
        "book_id": args.book_id,
        "diagnostic_only": True,
        "applied_to_graph": False,
        "note": (
            "These suggestions are diagnostic only. They are never applied automatically. "
            "To use them, copy relevant entries into data/aliases/<book_id>.json and "
            "re-run with --identity-mode human_refined --alias-file <path>."
        ),
        "threshold": args.threshold,
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
    }

    output_path = output_dir / "alias_suggestions.json"
    write_json(output_path, out)

    print(f"Wrote {output_path}")
    print(f"Suggestions: {len(suggestions)}")
    for s in suggestions[:10]:
        print(
            f"- {s['source_name']} -> {s['target_name']} "
            f"(confidence={s['confidence']})"
        )


if __name__ == "__main__":
    main()
