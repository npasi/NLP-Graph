"""Tests for step4b_normalized_evidence.py — chapter-level CO_PRESENCE generation.

Tests A-D cover the four requirements from the task spec:
  A. Co-presence generated without direct events
  B. No co-presence when fewer than two canonical characters
  C. Unmapped entities ignored
  D. Existing direct events preserved
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.step4b_normalized_evidence import build_normalized_pair_evidence


_ENTITIES_HEADER = "COREF\tstart_token\tend_token\tprop\tcat\ttext\n"
_TOKENS_HEADER = (
    "token_ID_within_document\tsentence_ID\tlemma\tdependency_relation\t"
    "syntactic_head_ID\tevent\tbyte_onset\tbyte_offset\n"
)


def _make_booknlp_root(
    tmp: str,
    chapter_id: int,
    coref_to_canonical: Dict[str, str],
    entity_rows: list[str],
    token_rows: list[str],
    event_evidence: Dict[str, Any],
    canonical_characters: list[Dict[str, Any]],
) -> Path:
    """Build a minimal fake booknlp root directory for one chapter."""
    root = Path(tmp)
    run_name = f"booknlp_t_chapter_{chapter_id:04d}"
    rd = root / run_name
    rd.mkdir(parents=True)

    # .entities
    (rd / f"{run_name}.entities").write_text(
        _ENTITIES_HEADER + "".join(entity_rows), encoding="utf-8"
    )
    # .tokens
    (rd / f"{run_name}.tokens").write_text(
        _TOKENS_HEADER + "".join(token_rows), encoding="utf-8"
    )

    # local_coref_to_character.json
    (root / "local_coref_to_character.json").write_text(
        json.dumps({str(chapter_id): coref_to_canonical}), encoding="utf-8"
    )

    # normalized_event_evidence_by_chapter.json
    (root / "normalized_event_evidence_by_chapter.json").write_text(
        json.dumps(event_evidence), encoding="utf-8"
    )

    # canonical_characters.json
    (root / "canonical_characters.json").write_text(
        json.dumps(canonical_characters), encoding="utf-8"
    )

    return root


class TestCoPresenceWithoutDirectEvents(unittest.TestCase):
    """A. Chapter-level co-presence should be generated even with zero DIRECT_EVENT pairs."""

    def test_chapter_co_presence_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_booknlp_root(
                tmp,
                chapter_id=0,
                coref_to_canonical={"1": "char_a", "2": "char_b"},
                entity_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "2\t5\t5\tPROP\tPER\tBob\n",   # different sentence from Alice
                ],
                token_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "5\t1\tBob\tnsubj\t1\tO\t10\t13\n",  # sentence 1, not 0
                ],
                event_evidence={"0": {}},
                canonical_characters=[
                    {"canonical_id": "char_a", "name": "Alice"},
                    {"canonical_id": "char_b", "name": "Bob"},
                ],
            )
            mapping = {"0": {"1": "char_a", "2": "char_b"}}

            result, diag = build_normalized_pair_evidence(
                booknlp_root=root, full_mapping=mapping
            )

            ch0 = result.get("0", {})
            self.assertTrue(
                len(ch0) >= 1,
                "Expected at least one pair in chapter 0"
            )

            # The pair char_a||char_b should exist.
            self.assertIn("char_a||char_b", ch0)
            ev_list = ch0["char_a||char_b"]
            self.assertTrue(len(ev_list) >= 1)
            ev_types = {ev["evidence_type"] for ev in ev_list}
            self.assertIn("CO_PRESENCE", ev_types)

            # Diagnostics.
            d0 = diag.get("0", {})
            self.assertEqual(d0["canonical_characters_observed"], 2)
            self.assertGreaterEqual(d0["pairs_written"], 1)
            self.assertIsNone(d0["reason_if_zero"])

    def test_chapter_scope_field_present(self):
        """Chapter-level items should have scope=chapter."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_booknlp_root(
                tmp,
                chapter_id=0,
                coref_to_canonical={"1": "char_a", "2": "char_b"},
                entity_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "2\t5\t5\tPROP\tPER\tBob\n",
                ],
                token_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "5\t1\tBob\tnsubj\t1\tO\t10\t13\n",
                ],
                event_evidence={"0": {}},
                canonical_characters=[
                    {"canonical_id": "char_a", "name": "Alice"},
                    {"canonical_id": "char_b", "name": "Bob"},
                ],
            )
            mapping = {"0": {"1": "char_a", "2": "char_b"}}
            result, _ = build_normalized_pair_evidence(
                booknlp_root=root, full_mapping=mapping
            )
            ch0 = result.get("0", {})
            ev_list = ch0.get("char_a||char_b", [])
            chap_items = [ev for ev in ev_list if ev.get("scope") == "chapter"]
            self.assertTrue(len(chap_items) >= 1, "Expected at least one chapter-scope item")
            self.assertIsNone(chap_items[0]["sentence_id"])
            self.assertAlmostEqual(chap_items[0]["confidence"], 0.15)


class TestNoCoPresenceFewerThanTwoChars(unittest.TestCase):
    """B. No co-presence when fewer than two canonical characters are observed."""

    def test_single_character_no_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_booknlp_root(
                tmp,
                chapter_id=0,
                coref_to_canonical={"1": "char_a"},
                entity_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "1\t3\t3\tPRON\tPER\tshe\n",
                ],
                token_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "3\t0\tshe\tnsubj\t1\tO\t6\t9\n",
                ],
                event_evidence={"0": {}},
                canonical_characters=[{"canonical_id": "char_a", "name": "Alice"}],
            )
            mapping = {"0": {"1": "char_a"}}

            result, diag = build_normalized_pair_evidence(
                booknlp_root=root, full_mapping=mapping
            )

            ch0 = result.get("0", {})
            self.assertEqual(len(ch0), 0, "Expected zero pairs with only one canonical char")

            d0 = diag.get("0", {})
            self.assertEqual(d0["canonical_characters_observed"], 1)
            self.assertEqual(d0["pairs_written"], 0)
            self.assertEqual(d0["reason_if_zero"], "fewer_than_two_canonical_characters")


class TestUnmappedEntitiesIgnored(unittest.TestCase):
    """C. Entities with no canonical mapping should not generate co-presence."""

    def test_unmapped_entities_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Only coref 1 is mapped; corefs 99 and 100 are not.
            root = _make_booknlp_root(
                tmp,
                chapter_id=0,
                coref_to_canonical={"1": "char_a"},
                entity_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "99\t1\t1\tPROP\tPER\tStranger\n",
                    "100\t2\t2\tPROP\tPER\tNobody\n",
                ],
                token_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "1\t0\tStranger\tnsubj\t1\tO\t6\t14\n",
                    "2\t0\tNobody\tobj\t1\tO\t15\t21\n",
                ],
                event_evidence={"0": {}},
                canonical_characters=[{"canonical_id": "char_a", "name": "Alice"}],
            )
            mapping = {"0": {"1": "char_a"}}

            result, diag = build_normalized_pair_evidence(
                booknlp_root=root, full_mapping=mapping
            )

            ch0 = result.get("0", {})
            self.assertEqual(len(ch0), 0, "Unmapped entities must not generate pairs")

            d0 = diag.get("0", {})
            self.assertEqual(d0["canonical_characters_observed"], 1)
            self.assertEqual(d0["pairs_written"], 0)


class TestExistingDirectEventsPreserved(unittest.TestCase):
    """D. DIRECT_EVENT evidence must be preserved; chapter-level CO_PRESENCE adds to it."""

    def test_direct_and_co_presence_both_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            # char_a and char_b are in different sentences (no sentence-level CO_PRESENCE).
            # But we have a pre-existing DIRECT_EVENT from event evidence.
            # Also char_c is observed in the chapter, paired with char_a and char_b via chapter-level.
            event_ev = {
                "0": {
                    "char_a||char_b": [{
                        "evidence_type": "DIRECT_EVENT",
                        "sentence_id": 0,
                        "text": "Alice helped Bob",
                        "involved_characters": ["char_a", "char_b"],
                        "predicate": "helped",
                        "confidence": 0.9,
                    }]
                }
            }
            root = _make_booknlp_root(
                tmp,
                chapter_id=0,
                coref_to_canonical={"1": "char_a", "2": "char_b", "3": "char_c"},
                entity_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "2\t5\t5\tPROP\tPER\tBob\n",
                    "3\t10\t10\tPROP\tPER\tCarol\n",
                ],
                token_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "5\t1\tBob\tnsubj\t1\tO\t10\t13\n",
                    "10\t2\tCarol\tnsubj\t1\tO\t20\t25\n",
                ],
                event_evidence=event_ev,
                canonical_characters=[
                    {"canonical_id": "char_a", "name": "Alice"},
                    {"canonical_id": "char_b", "name": "Bob"},
                    {"canonical_id": "char_c", "name": "Carol"},
                ],
            )
            mapping = {"0": {"1": "char_a", "2": "char_b", "3": "char_c"}}

            result, diag = build_normalized_pair_evidence(
                booknlp_root=root, full_mapping=mapping
            )

            ch0 = result.get("0", {})

            # The DIRECT_EVENT pair must still be there.
            self.assertIn("char_a||char_b", ch0)
            ab = ch0["char_a||char_b"]
            types = {ev["evidence_type"] for ev in ab}
            self.assertIn("DIRECT_EVENT", types, "DIRECT_EVENT evidence must be preserved")

            # char_a||char_c and char_b||char_c should have chapter-level CO_PRESENCE.
            self.assertIn("char_a||char_c", ch0)
            self.assertIn("char_b||char_c", ch0)

            for pk in ("char_a||char_c", "char_b||char_c"):
                ev_list = ch0[pk]
                self.assertTrue(
                    any(ev["evidence_type"] == "CO_PRESENCE" for ev in ev_list),
                    f"{pk} should have CO_PRESENCE evidence"
                )

            # Diagnostics.
            d0 = diag.get("0", {})
            self.assertEqual(d0["canonical_characters_observed"], 3)
            self.assertEqual(d0["direct_event_pairs"], 1)
            # char_a||char_b has DIRECT_EVENT and is excluded from chapter-level
            self.assertEqual(d0["chapter_co_presence_pairs"], 2)  # a-c and b-c


class TestDiagnosticsFileWritten(unittest.TestCase):
    """Diagnostics JSON file is written to the booknlp root."""

    def test_diagnostics_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_booknlp_root(
                tmp,
                chapter_id=0,
                coref_to_canonical={"1": "char_a", "2": "char_b"},
                entity_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "2\t5\t5\tPROP\tPER\tBob\n",
                ],
                token_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "5\t1\tBob\tnsubj\t1\tO\t10\t13\n",
                ],
                event_evidence={"0": {}},
                canonical_characters=[
                    {"canonical_id": "char_a", "name": "Alice"},
                    {"canonical_id": "char_b", "name": "Bob"},
                ],
            )
            mapping = {"0": {"1": "char_a", "2": "char_b"}}

            # Call via main() to check the file is written.
            from src.step4b_normalized_evidence import main as step4b_main
            import io, contextlib
            step4b_main([
                "--book-id", "fake",
                "--booknlp-root", str(root),
                "--log-level", "WARNING",
            ])

            diag_path = root / "co_presence_diagnostics_by_chapter.json"
            self.assertTrue(diag_path.exists(), "Diagnostics file must be written")
            diag = json.loads(diag_path.read_text())
            self.assertIn("0", diag)
            d0 = diag["0"]
            self.assertIn("canonical_characters_observed", d0)
            self.assertIn("characters", d0)
            self.assertIn("pairs_written", d0)


if __name__ == "__main__":
    unittest.main()
