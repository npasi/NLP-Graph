"""Tests for scripts/make_identity_audit.py helper functions."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import make_identity_audit as audit_mod


class TestSuspiciousNameHeuristic(unittest.TestCase):

    def test_abstract_name_ivory(self):
        reasons = audit_mod._is_suspicious_name("Ivory", ["Ivory"])
        self.assertTrue(any("abstract" in r or "Ivory" in r for r in reasons),
                        f"Expected suspicion for 'Ivory', got: {reasons}")

    def test_abstract_name_good_god(self):
        reasons = audit_mod._is_suspicious_name("Good God", ["Good God"])
        self.assertTrue(reasons, f"Expected suspicion for 'Good God', got: {reasons}")

    def test_abstract_name_jove(self):
        reasons = audit_mod._is_suspicious_name("Jove", ["Jove"])
        self.assertTrue(reasons, f"Expected suspicion for 'Jove', got: {reasons}")

    def test_group_pattern_the_french(self):
        reasons = audit_mod._is_suspicious_name("the French", ["the French"])
        self.assertTrue(any("group" in r.lower() or "nationality" in r.lower() for r in reasons),
                        f"Expected group flag for 'the French', got: {reasons}")

    def test_group_pattern_the_company(self):
        reasons = audit_mod._is_suspicious_name("the Company", ["the Company"])
        self.assertTrue(reasons, f"Expected suspicion for 'the Company', got: {reasons}")

    def test_long_alias_flagged(self):
        long_alias = "a Mrs. Younge, who was some time ago governess to Miss Darcy and was dismissed from her charge on account of her conduct"
        reasons = audit_mod._is_suspicious_name("Mrs. Younge", [long_alias])
        self.assertTrue(any("long alias" in r.lower() or "alias" in r.lower() for r in reasons),
                        f"Expected long alias flag, got: {reasons}")

    def test_normal_name_not_flagged(self):
        reasons = audit_mod._is_suspicious_name("Elizabeth Bennet", ["Elizabeth", "Miss Bennet"])
        self.assertFalse(reasons, f"Normal name should not be flagged, got: {reasons}")

    def test_normal_name_kurtz(self):
        reasons = audit_mod._is_suspicious_name("Kurtz", ["Kurtz", "Mr. Kurtz"])
        self.assertFalse(reasons, f"Kurtz should not be flagged, got: {reasons}")


class TestLowMentionDetection(unittest.TestCase):

    def test_low_mention_no_evidence(self):
        canonicals = [
            {"canonical_id": "char_ghost", "name": "Ghost",
             "type": "individual_candidate", "aliases": ["Ghost"]},
        ]
        mention_stats = {"char_ghost": {"mapped_mentions": 1, "proper_mentions": 1,
                                         "nominal_mentions": 0, "pronoun_mentions": 0}}
        participation = {"char_ghost": {"total_pair_participation": 0,
                                         "strong_pair_participation": 0,
                                         "direct_event_participation": 0,
                                         "dialogue_turn_participation": 0,
                                         "quote_about_participation": 0,
                                         "co_presence_only_participation": 0}}
        suspicious, low_mention = audit_mod._analyse_suspicious_canonicals(
            canonicals, mention_stats, participation
        )
        self.assertEqual(len(low_mention), 1)
        self.assertEqual(low_mention[0]["canonical_id"], "char_ghost")

    def test_not_low_mention_when_strong_evidence(self):
        canonicals = [
            {"canonical_id": "char_alice", "name": "Alice",
             "type": "individual_candidate", "aliases": ["Alice"]},
        ]
        mention_stats = {"char_alice": {"mapped_mentions": 1, "proper_mentions": 1,
                                         "nominal_mentions": 0, "pronoun_mentions": 0}}
        participation = {"char_alice": {"total_pair_participation": 2,
                                         "strong_pair_participation": 1,
                                         "direct_event_participation": 1,
                                         "dialogue_turn_participation": 0,
                                         "quote_about_participation": 0,
                                         "co_presence_only_participation": 0}}
        _suspicious, low_mention = audit_mod._analyse_suspicious_canonicals(
            canonicals, mention_stats, participation
        )
        # 1 mention but has strong evidence → not low-mention
        self.assertEqual(len(low_mention), 0)


class TestHighSalienceExcluded(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_chapter_dir(self, chapter_id: int, coref_id: str, mention_count: int) -> Path:
        chapter_dir = self.tmp_path / f"booknlp_test_chapter_{chapter_id:04d}"
        chapter_dir.mkdir()

        # Write .entities file
        entities_lines = ["COREF\tstart_token\tend_token\tprop\tcat\ttext"]
        for i in range(mention_count):
            entities_lines.append(f"{coref_id}\t{i*3}\t{i*3}\tNOM\tPER\tthe manager")
        (chapter_dir / f"test_chapter_{chapter_id:04d}.entities").write_text(
            "\n".join(entities_lines), encoding="utf-8"
        )

        # Empty quotes file
        (chapter_dir / f"test_chapter_{chapter_id:04d}.quotes").write_text(
            "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n",
            encoding="utf-8",
        )
        return chapter_dir

    def test_high_salience_excluded_detected(self):
        chapter_dir = self._make_chapter_dir(0, "42", 15)
        decisions = [
            {
                "chapter_id": 0,
                "coref_id": "42",
                "surface": "the manager",
                "decision": "REVIEW",
                "confidence": 0.0,
                "reasons": ["review_role_candidate"],
            }
        ]
        result = audit_mod._compute_high_salience_excluded(decisions, [chapter_dir])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["coref_id"], "42")
        self.assertEqual(result[0]["mention_count"], 15)

    def test_low_mention_not_flagged(self):
        chapter_dir = self._make_chapter_dir(0, "7", 3)
        decisions = [
            {
                "chapter_id": 0,
                "coref_id": "7",
                "surface": "the old man",
                "decision": "REVIEW",
                "confidence": 0.0,
                "reasons": ["review_role_candidate"],
            }
        ]
        result = audit_mod._compute_high_salience_excluded(decisions, [chapter_dir])
        self.assertEqual(len(result), 0, "3 mentions is below threshold, should not be flagged")


class TestMissingFilesRobustness(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_audit_book_missing_all_files(self):
        """Script should not crash when all optional files are absent."""
        run_dir = self.tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "booknlp_chapter_output" / "test_book").mkdir(parents=True)
        (run_dir / "reports" / "test_book").mkdir(parents=True)
        (run_dir / "books" / "test_book" / "chapters").mkdir(parents=True)
        (run_dir / "ml" / "test_book").mkdir(parents=True)

        # Should not raise
        result_audit, susp_rows = audit_mod._audit_book("test_book", run_dir)

        self.assertEqual(result_audit["book_id"], "test_book")
        self.assertIn("audit_status", result_audit)
        self.assertIsInstance(susp_rows, list)

    def test_load_json_missing_file(self):
        result = audit_mod._load_json(self.tmp_path / "nonexistent.json", default=[])
        self.assertEqual(result, [])

    def test_load_json_corrupt_file(self):
        p = self.tmp_path / "bad.json"
        p.write_text("not valid json{{{", encoding="utf-8")
        result = audit_mod._load_json(p, default={"fallback": True})
        self.assertEqual(result, {"fallback": True})


class TestAuditStatus(unittest.TestCase):

    def test_missing_output_status(self):
        status = audit_mod._audit_status(
            has_canonical=False,
            has_identity_report=True,
            canonical_per_10k=5.0,
            suspicious_count=0,
            review_per_canonical=1.0,
            unresolved_per_canonical=2.0,
            mapped_speaker_ratio=0.5,
            warnings=[],
        )
        self.assertEqual(status, "MISSING_IDENTITY_OUTPUT")

    def test_high_risk_noise_status(self):
        status = audit_mod._audit_status(
            has_canonical=True,
            has_identity_report=True,
            canonical_per_10k=25.0,  # above threshold
            suspicious_count=1,
            review_per_canonical=1.0,
            unresolved_per_canonical=2.0,
            mapped_speaker_ratio=0.5,
            warnings=["some warning"],
        )
        self.assertEqual(status, "HIGH_RISK_IDENTITY_NOISE")

    def test_low_speaker_mapping_status(self):
        status = audit_mod._audit_status(
            has_canonical=True,
            has_identity_report=True,
            canonical_per_10k=5.0,
            suspicious_count=1,
            review_per_canonical=1.0,
            unresolved_per_canonical=2.0,
            mapped_speaker_ratio=0.10,  # below threshold
            warnings=[],
        )
        self.assertEqual(status, "LOW_SPEAKER_MAPPING")

    def test_healthy_status(self):
        status = audit_mod._audit_status(
            has_canonical=True,
            has_identity_report=True,
            canonical_per_10k=3.0,
            suspicious_count=0,
            review_per_canonical=1.0,
            unresolved_per_canonical=2.0,
            mapped_speaker_ratio=0.6,
            warnings=[],
        )
        self.assertEqual(status, "HEALTHY_OR_ACCEPTABLE")


if __name__ == "__main__":
    unittest.main()
