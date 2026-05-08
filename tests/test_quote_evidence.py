"""Tests for DIALOGUE_TURN and QUOTE_ABOUT evidence extraction in step4b.

Tests 1-7 per task spec:
  1. test_dialogue_turn_generated
  2. test_dialogue_turn_not_generated_for_same_speaker
  3. test_dialogue_turn_respects_gap
  4. test_quote_about_generated
  5. test_unmapped_speaker_ignored
  6. test_quote_diagnostics_written
  7. test_score_interactions_counts_quote_evidence
  + extra: dedup tests, score weighting checks
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.step4b_normalized_evidence import (
    _dialogue_turn_evidence,
    _quote_about_evidence,
    build_normalized_pair_evidence,
)
from src.score_interactions import score_pair


_ENT_HDR  = "COREF\tstart_token\tend_token\tprop\tcat\ttext\n"
_TOK_HDR  = (
    "token_ID_within_document\tsentence_ID\tlemma\tdependency_relation\t"
    "syntactic_head_ID\tevent\tbyte_onset\tbyte_offset\n"
)
_QT_HDR   = "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n"


def _make_qt_row(qs: int, qe: int, char_id: int, quote: str, ment_start: int = 0) -> str:
    return f"{qs}\t{qe}\t{ment_start}\t{ment_start}\tphrase\t{char_id}\t{quote}\n"


def _make_root(
    tmp: str,
    chapter_id: int = 0,
    coref_map: Dict[str, str] | None = None,
    ent_rows: List[str] | None = None,
    tok_rows: List[str] | None = None,
    qt_rows: List[str] | None = None,
    event_ev: Dict[str, Any] | None = None,
    canon_chars: List[Dict[str, Any]] | None = None,
) -> Path:
    root = Path(tmp)
    run_name = f"booknlp_t_chapter_{chapter_id:04d}"
    rd = root / run_name
    rd.mkdir(parents=True)

    (rd / f"{run_name}.entities").write_text(
        _ENT_HDR + "".join(ent_rows or []), encoding="utf-8"
    )
    (rd / f"{run_name}.tokens").write_text(
        _TOK_HDR + "".join(tok_rows or []), encoding="utf-8"
    )
    if qt_rows is not None:
        (rd / f"{run_name}.quotes").write_text(
            _QT_HDR + "".join(qt_rows), encoding="utf-8"
        )

    (root / "local_coref_to_character.json").write_text(
        json.dumps({str(chapter_id): coref_map or {}}), encoding="utf-8"
    )
    (root / "normalized_event_evidence_by_chapter.json").write_text(
        json.dumps(event_ev or {str(chapter_id): {}}), encoding="utf-8"
    )
    (root / "canonical_characters.json").write_text(
        json.dumps(canon_chars or []), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# Test 1: DIALOGUE_TURN generated for adjacent different-speaker quotes
# ---------------------------------------------------------------------------

class TestDialogueTurnGenerated(unittest.TestCase):
    def test_dialogue_turn_generated(self):
        """Two adjacent quotes by different mapped speakers within gap → DIALOGUE_TURN."""
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            _make_qt_row(10, 20, char_id=1, quote='"Hello!"').strip().split("\t"),
            _make_qt_row(25, 35, char_id=2, quote='"Hi there!"').strip().split("\t"),
        ]
        coref_map = {"1": "char_alice", "2": "char_bob"}

        ev = _dialogue_turn_evidence(qt_hdr, qt_rows, coref_map, max_gap=80)

        self.assertEqual(len(ev), 1)
        pk = "char_alice||char_bob"
        self.assertIn(pk, ev)
        items = ev[pk]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["evidence_type"], "DIALOGUE_TURN")
        self.assertAlmostEqual(items[0]["confidence"], 0.85)
        self.assertEqual(items[0]["token_gap"], 5)   # 25 - 20

    def test_dialogue_turn_full_pipeline(self):
        """End-to-end: DIALOGUE_TURN appears in build_normalized_pair_evidence."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_root(
                tmp,
                coref_map={"1": "char_alice", "2": "char_bob"},
                ent_rows=[
                    "1\t10\t10\tPROP\tPER\tAlice\n",
                    "2\t25\t25\tPROP\tPER\tBob\n",
                ],
                tok_rows=[
                    "10\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "25\t1\tBob\tnsubj\t1\tO\t6\t9\n",
                ],
                qt_rows=[
                    _make_qt_row(10, 20, 1, '"Hello!"'),
                    _make_qt_row(25, 35, 2, '"Hi there!"'),
                ],
                canon_chars=[
                    {"canonical_id": "char_alice", "name": "Alice"},
                    {"canonical_id": "char_bob", "name": "Bob"},
                ],
            )
            mapping = {"0": {"1": "char_alice", "2": "char_bob"}}
            result, _, _ = build_normalized_pair_evidence(booknlp_root=root, full_mapping=mapping)

            ch0 = result.get("0", {})
            self.assertIn("char_alice||char_bob", ch0)
            types = {ev["evidence_type"] for ev in ch0["char_alice||char_bob"]}
            self.assertIn("DIALOGUE_TURN", types)


# ---------------------------------------------------------------------------
# Test 2: No DIALOGUE_TURN for same speaker
# ---------------------------------------------------------------------------

class TestDialogueTurnSameSpeaker(unittest.TestCase):
    def test_same_speaker_no_dialogue_turn(self):
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            _make_qt_row(10, 20, char_id=1, quote='"First."').strip().split("\t"),
            _make_qt_row(22, 35, char_id=1, quote='"Second."').strip().split("\t"),
        ]
        coref_map = {"1": "char_alice"}

        ev = _dialogue_turn_evidence(qt_hdr, qt_rows, coref_map, max_gap=80)
        self.assertEqual(len(ev), 0, "No DIALOGUE_TURN for same speaker")


# ---------------------------------------------------------------------------
# Test 3: DIALOGUE_TURN respects gap limit
# ---------------------------------------------------------------------------

class TestDialogueTurnGap(unittest.TestCase):
    def test_gap_too_large_no_turn(self):
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            _make_qt_row(10, 20, char_id=1, quote='"Alice says."').strip().split("\t"),
            _make_qt_row(200, 210, char_id=2, quote='"Bob says."').strip().split("\t"),
        ]
        coref_map = {"1": "char_alice", "2": "char_bob"}

        ev = _dialogue_turn_evidence(qt_hdr, qt_rows, coref_map, max_gap=80)
        self.assertEqual(len(ev), 0, "Gap 180 > 80 should not produce DIALOGUE_TURN")

    def test_gap_exactly_at_limit(self):
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            _make_qt_row(10, 20, char_id=1, quote='"A."').strip().split("\t"),
            _make_qt_row(100, 110, char_id=2, quote='"B."').strip().split("\t"),
        ]
        coref_map = {"1": "char_alice", "2": "char_bob"}

        ev = _dialogue_turn_evidence(qt_hdr, qt_rows, coref_map, max_gap=80)
        self.assertEqual(len(ev), 1, "Gap 80 == limit should produce DIALOGUE_TURN")


# ---------------------------------------------------------------------------
# Test 4: QUOTE_ABOUT generated
# ---------------------------------------------------------------------------

class TestQuoteAboutGenerated(unittest.TestCase):
    def test_quote_about_generated(self):
        """Entity mention inside quote span → QUOTE_ABOUT between speaker and mentioned."""
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            # Speaker char_id=1, quote spans tokens 10-30
            _make_qt_row(10, 30, char_id=1, quote='"I think Bob is great."').strip().split("\t"),
        ]
        # Entity coref 2 (Bob) starts at token 15 (inside 10-30)
        ent_rows_parsed = [
            ["2", "15", "15", "PROP", "PER", "Bob"],
        ]
        coref_map = {"1": "char_alice", "2": "char_bob"}

        ev = _quote_about_evidence(qt_hdr, qt_rows, ent_rows_parsed, coref_map)

        self.assertEqual(len(ev), 1)
        pk = "char_alice||char_bob"
        self.assertIn(pk, ev)
        items = ev[pk]
        self.assertEqual(items[0]["evidence_type"], "QUOTE_ABOUT")
        self.assertEqual(items[0]["speaker"], "char_alice")
        self.assertEqual(items[0]["mentioned_character"], "char_bob")
        self.assertAlmostEqual(items[0]["confidence"], 0.75)

    def test_quote_about_full_pipeline(self):
        """End-to-end: QUOTE_ABOUT appears in build_normalized_pair_evidence."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_root(
                tmp,
                coref_map={"1": "char_alice", "2": "char_bob"},
                ent_rows=[
                    "1\t5\t5\tPROP\tPER\tAlice\n",
                    "2\t20\t20\tPROP\tPER\tBob\n",
                ],
                tok_rows=[
                    "5\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "20\t1\tBob\tnsubj\t1\tO\t50\t53\n",
                ],
                qt_rows=[
                    # Alice (coref 1) speaks; Bob (coref 2) at token 20 is inside quote 10-30
                    _make_qt_row(10, 30, 1, '"I think Bob is great."'),
                ],
                canon_chars=[
                    {"canonical_id": "char_alice", "name": "Alice"},
                    {"canonical_id": "char_bob", "name": "Bob"},
                ],
            )
            mapping = {"0": {"1": "char_alice", "2": "char_bob"}}
            result, _, _ = build_normalized_pair_evidence(booknlp_root=root, full_mapping=mapping)

            ch0 = result.get("0", {})
            self.assertIn("char_alice||char_bob", ch0)
            types = {ev["evidence_type"] for ev in ch0["char_alice||char_bob"]}
            self.assertIn("QUOTE_ABOUT", types)

    def test_speaker_not_mentioned_as_mentioned_character(self):
        """Speaker's own coref inside quote span should not create QUOTE_ABOUT pair."""
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            _make_qt_row(10, 30, char_id=1, quote='"I am Alice."').strip().split("\t"),
        ]
        ent_rows_parsed = [
            ["1", "15", "15", "PRON", "PER", "I"],
        ]
        coref_map = {"1": "char_alice"}

        ev = _quote_about_evidence(qt_hdr, qt_rows, ent_rows_parsed, coref_map)
        self.assertEqual(len(ev), 0, "Self-reference should not create QUOTE_ABOUT")


# ---------------------------------------------------------------------------
# Test 5: Unmapped speaker ignored
# ---------------------------------------------------------------------------

class TestUnmappedSpeakerIgnored(unittest.TestCase):
    def test_unmapped_speaker_no_dialogue_turn(self):
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            # char_id=99 not in coref_map
            _make_qt_row(10, 20, char_id=99, quote='"Hello!"').strip().split("\t"),
            _make_qt_row(25, 35, char_id=2,  quote='"Hi!"').strip().split("\t"),
        ]
        coref_map = {"2": "char_bob"}  # only bob is mapped

        ev = _dialogue_turn_evidence(qt_hdr, qt_rows, coref_map, max_gap=80)
        self.assertEqual(len(ev), 0, "Unmapped speaker should not generate DIALOGUE_TURN")

    def test_unmapped_speaker_no_quote_about(self):
        qt_hdr = _QT_HDR.strip().split("\t")
        qt_rows = [
            _make_qt_row(10, 30, char_id=99, quote='"I see Bob."').strip().split("\t"),
        ]
        ent_rows_parsed = [["2", "20", "20", "PROP", "PER", "Bob"]]
        coref_map = {"2": "char_bob"}  # speaker 99 is not mapped

        ev = _quote_about_evidence(qt_hdr, qt_rows, ent_rows_parsed, coref_map)
        self.assertEqual(len(ev), 0, "Unmapped speaker should not generate QUOTE_ABOUT")


# ---------------------------------------------------------------------------
# Test 6: Diagnostics file written by main()
# ---------------------------------------------------------------------------

class TestQuoteDiagnosticsWritten(unittest.TestCase):
    def test_quote_diagnostics_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_root(
                tmp,
                coref_map={"1": "char_alice", "2": "char_bob"},
                ent_rows=[
                    "1\t0\t0\tPROP\tPER\tAlice\n",
                    "2\t5\t5\tPROP\tPER\tBob\n",
                ],
                tok_rows=[
                    "0\t0\tAlice\tnsubj\t1\tO\t0\t5\n",
                    "5\t1\tBob\tnsubj\t1\tO\t6\t9\n",
                ],
                qt_rows=[
                    _make_qt_row(10, 20, 1, '"Hello!"'),
                    _make_qt_row(25, 35, 2, '"Hi!"'),
                ],
                canon_chars=[
                    {"canonical_id": "char_alice", "name": "Alice"},
                    {"canonical_id": "char_bob", "name": "Bob"},
                ],
            )

            from src.step4b_normalized_evidence import main as step4b_main
            step4b_main([
                "--book-id", "fake",
                "--booknlp-root", str(root),
                "--log-level", "WARNING",
            ])

            diag_path = root / "quote_evidence_diagnostics_by_chapter.json"
            self.assertTrue(diag_path.exists(), "Quote diagnostics file must be written")

            diag = json.loads(diag_path.read_text())
            self.assertIn("0", diag)
            d0 = diag["0"]
            self.assertIn("quotes_total", d0)
            self.assertIn("mapped_speaker_quotes", d0)
            self.assertIn("dialogue_turn_pairs", d0)
            self.assertIn("quote_about_pairs", d0)
            self.assertEqual(d0["quotes_total"], 2)
            self.assertEqual(d0["mapped_speaker_quotes"], 2)
            self.assertEqual(d0["dialogue_turn_pairs"], 1)

    def test_missing_quotes_file_produces_zero_diagnostics(self):
        """Books without .quotes files get zero-valued diagnostics."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_root(
                tmp,
                coref_map={"1": "char_alice"},
                ent_rows=["1\t0\t0\tPROP\tPER\tAlice\n"],
                tok_rows=["0\t0\tAlice\tnsubj\t1\tO\t0\t5\n"],
                qt_rows=None,  # no quotes file
                canon_chars=[{"canonical_id": "char_alice", "name": "Alice"}],
            )

            from src.step4b_normalized_evidence import main as step4b_main
            step4b_main([
                "--book-id", "fake",
                "--booknlp-root", str(root),
                "--log-level", "WARNING",
            ])

            diag_path = root / "quote_evidence_diagnostics_by_chapter.json"
            self.assertTrue(diag_path.exists())
            diag = json.loads(diag_path.read_text())
            d0 = diag.get("0", {})
            self.assertEqual(d0.get("quotes_total", 0), 0)


# ---------------------------------------------------------------------------
# Test 7: score_interactions uses new weights for DIALOGUE_TURN and QUOTE_ABOUT
# ---------------------------------------------------------------------------

class TestScoreInteractionsQuoteEvidence(unittest.TestCase):
    def test_dialogue_turn_weight(self):
        """DIALOGUE_TURN contributes 1.5 per item to interaction_score."""
        evidences = [
            {"evidence_type": "DIALOGUE_TURN", "sentence_id": None,
             "text": '"A"', "predicate": None, "confidence": 0.85},
        ]
        scored = score_pair(evidences)
        self.assertEqual(scored["dialogue_turn_count"], 1)
        self.assertAlmostEqual(scored["interaction_score"], 1.5)
        self.assertEqual(scored["relation_type"], "dialogue_interaction")

    def test_quote_about_weight(self):
        """QUOTE_ABOUT contributes 1.0 per item to interaction_score."""
        evidences = [
            {"evidence_type": "QUOTE_ABOUT", "sentence_id": None,
             "text": '"B"', "predicate": None, "confidence": 0.75},
        ]
        scored = score_pair(evidences)
        self.assertEqual(scored["quote_about_count"], 1)
        self.assertAlmostEqual(scored["interaction_score"], 1.0)
        self.assertEqual(scored["relation_type"], "quote_about_interaction")

    def test_strong_evidence_count_field(self):
        evidences = [
            {"evidence_type": "DIRECT_EVENT", "sentence_id": 0,
             "text": "act", "predicate": "helped", "confidence": 0.9},
            {"evidence_type": "DIALOGUE_TURN", "sentence_id": None,
             "text": '"hi"', "predicate": None, "confidence": 0.85},
            {"evidence_type": "QUOTE_ABOUT", "sentence_id": None,
             "text": '"mentioned"', "predicate": None, "confidence": 0.75},
        ]
        scored = score_pair(evidences)
        self.assertEqual(scored["strong_evidence_count"], 3)
        self.assertAlmostEqual(scored["interaction_score"], 2.0 + 1.5 + 1.0)
        # strong_evidence_count=3 < 4, direct_event_count=1 → medium
        self.assertEqual(scored["confidence"], "medium")

    def test_combined_score_formula(self):
        """Full formula: 2*DE + 1.5*DT + 1.0*QA + 0.25*CP."""
        evidences = [
            {"evidence_type": "DIRECT_EVENT",  "sentence_id": 1,  "text": "x", "predicate": "hit", "confidence": 0.9},
            {"evidence_type": "DIALOGUE_TURN", "sentence_id": None, "text": '"a"', "predicate": None, "confidence": 0.85},
            {"evidence_type": "DIALOGUE_TURN", "sentence_id": None, "text": '"b"', "predicate": None, "confidence": 0.85},
            {"evidence_type": "QUOTE_ABOUT",   "sentence_id": None, "text": '"c"', "predicate": None, "confidence": 0.75},
            {"evidence_type": "CO_PRESENCE",   "sentence_id": 2,  "text": "y", "predicate": None, "confidence": 0.25},
        ]
        scored = score_pair(evidences)
        expected = 2.0 * 1 + 1.5 * 2 + 1.0 * 1 + 0.25 * 1
        self.assertAlmostEqual(scored["interaction_score"], expected)
        self.assertEqual(scored["quote_evidence_count"], 3)

    def test_confidence_high_via_dialogue(self):
        """3+ DIALOGUE_TURN items alone → confidence = high."""
        evidences = [
            {"evidence_type": "DIALOGUE_TURN", "sentence_id": None, "text": f'"{i}"',
             "predicate": None, "confidence": 0.85}
            for i in range(3)
        ]
        scored = score_pair(evidences)
        self.assertEqual(scored["confidence"], "high")

    def test_confidence_low_medium_via_quote_about(self):
        """1 QUOTE_ABOUT, no direct → confidence = low_medium."""
        evidences = [
            {"evidence_type": "QUOTE_ABOUT", "sentence_id": None,
             "text": '"x"', "predicate": None, "confidence": 0.75},
        ]
        scored = score_pair(evidences)
        self.assertEqual(scored["confidence"], "low_medium")

    def test_relation_type_priority(self):
        """DIRECT_EVENT takes precedence over DIALOGUE_TURN in relation_type."""
        evidences = [
            {"evidence_type": "DIRECT_EVENT",  "sentence_id": 1, "text": "a", "predicate": "hurt", "confidence": 0.9},
            {"evidence_type": "DIALOGUE_TURN", "sentence_id": None, "text": '"b"', "predicate": None, "confidence": 0.85},
        ]
        scored = score_pair(evidences)
        self.assertEqual(scored["relation_type"], "direct_interaction")


if __name__ == "__main__":
    unittest.main()
