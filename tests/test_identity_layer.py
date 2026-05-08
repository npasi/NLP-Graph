"""Unit tests for the Character Identity Layer.

All tests use synthetic data in temporary directories.
No real BookNLP model is loaded or executed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.character_identity_layer import (
    CharacterIdentityLayer,
    LocalCluster,
    _norm,
    _qualifies_for_promotion,
    _strip_title,
    classify_cluster,
    load_canonical_characters,
    load_coref_mapping,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _char(
    cid: int,
    proper: List[str],
    common: Optional[List[str]] = None,
    pronoun: Optional[List[str]] = None,
    count: int = 5,
    gender: str = "",
) -> Dict[str, Any]:
    def _m(names: List[str]) -> List[Dict[str, Any]]:
        return [{"n": n, "c": max(1, count - i)} for i, n in enumerate(names)]
    return {
        "id": cid, "count": count,
        "mentions": {
            "proper":  _m(proper),
            "common":  _m(common or []),
            "pronoun": _m(pronoun or []),
        },
        "g": {"inference": gender},
    }


def _make_run_dir(
    root: Path,
    book_id: str,
    chapter_id: int,
    characters: List[Dict[str, Any]],
) -> Path:
    run_name = f"booknlp_{book_id}_chapter_{chapter_id:04d}"
    run_dir  = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{run_name}.book").write_text(
        json.dumps({"characters": characters}), encoding="utf-8"
    )
    (run_dir / f"{run_name}.entities").write_text(
        "COREF\tstart_token\tend_token\tprop\tcat\ttext\n", encoding="utf-8"
    )
    (run_dir / f"{run_name}.tokens").write_text(
        "token_ID_within_document\tsentence_ID\tlemma\tdependency_relation\t"
        "syntactic_head_ID\tevent\tbyte_onset\tbyte_offset\n",
        encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------
# 1. Name helpers
# ---------------------------------------------------------------------------

class TestNormAndStrip(unittest.TestCase):
    def test_norm_collapses_whitespace(self):
        self.assertEqual(_norm("  Mr.  Scrooge "), "mr. scrooge")

    def test_strip_mr(self):
        base, title = _strip_title("Mr. Scrooge")
        self.assertEqual(title, "mr.")
        self.assertEqual(base, "Scrooge")

    def test_strip_mrs(self):
        base, title = _strip_title("Mrs. Cratchit")
        self.assertEqual(title, "mrs.")
        self.assertEqual(base, "Cratchit")

    def test_strip_uncle(self):
        base, title = _strip_title("Uncle Karl")
        self.assertEqual(title, "uncle")
        self.assertEqual(base, "Karl")

    def test_no_title(self):
        base, title = _strip_title("Scrooge")
        self.assertIsNone(title)
        self.assertEqual(base, "Scrooge")


# ---------------------------------------------------------------------------
# 2. Cluster classification
# ---------------------------------------------------------------------------

class TestClassification(unittest.TestCase):
    def _cl(self, proper, common=None, pronoun=None) -> LocalCluster:
        cl = LocalCluster(
            chapter_id=0, coref_id="1",
            display_name=(proper or common or pronoun or ["?"])[0],
            proper_names=proper or [],
            common_names=common or [],
            pronoun_forms=pronoun or [],
            mention_count=5,
        )
        return cl

    def test_individual_scrooge(self):
        self.assertEqual(classify_cluster(self._cl(["Scrooge"])), "individual_candidate")

    def test_group_children(self):
        cl = self._cl([], ["the children"])
        self.assertEqual(classify_cluster(cl), "group_candidate")

    def test_group_policemen(self):
        cl = self._cl([], ["the policemen"])
        self.assertEqual(classify_cluster(cl), "group_candidate")

    def test_generic_noise_everybody(self):
        cl = self._cl([], [], ["everybody"])
        cl.display_name = "everybody"
        cl.common_names = ["everybody"]
        cl.proper_names = []
        self.assertEqual(classify_cluster(cl), "generic_noise")

    def test_generic_noise_dear(self):
        cl = LocalCluster(
            chapter_id=0, coref_id="2", display_name="dear",
            proper_names=[], common_names=[], pronoun_forms=["dear"],
            mention_count=2,
        )
        cl.cluster_type = classify_cluster(cl)
        self.assertEqual(cl.cluster_type, "generic_noise")

    def test_abstract_noise_christmas(self):
        self.assertEqual(classify_cluster(self._cl(["Christmas"])), "abstract_noise")

    def test_narrator_candidate(self):
        cl = LocalCluster(
            chapter_id=0, coref_id="3", display_name="i",
            proper_names=[], common_names=[],
            pronoun_forms=["I", "I", "I", "me", "my", "I", "I"],
            mention_count=7,
        )
        self.assertEqual(classify_cluster(cl), "narrator_candidate")

    def test_review_role_supervisor(self):
        cl = self._cl([], ["the supervisor"])
        self.assertEqual(classify_cluster(cl), "review_role_candidate")

    def test_review_role_a_businessman(self):
        cl = self._cl([], ["a businessman"])
        self.assertEqual(classify_cluster(cl), "review_role_candidate")

    def test_review_role_a_client(self):
        cl = self._cl([], ["a client"])
        self.assertEqual(classify_cluster(cl), "review_role_candidate")

    def test_individual_bob_cratchit(self):
        self.assertEqual(
            classify_cluster(self._cl(["Bob Cratchit", "Bob"])),
            "individual_candidate",
        )


# ---------------------------------------------------------------------------
# 3. Promotion policy
# ---------------------------------------------------------------------------

class TestPromotionPolicy(unittest.TestCase):
    def _cl(self, cluster_type, proper=None) -> LocalCluster:
        cl = LocalCluster(
            chapter_id=0, coref_id="1", display_name="x",
            proper_names=proper or [],
            common_names=[], pronoun_forms=[], mention_count=3,
        )
        cl.cluster_type = cluster_type
        return cl

    def test_individual_with_proper_name_promoted(self):
        self.assertTrue(_qualifies_for_promotion(self._cl("individual_candidate", ["K."]), False))

    def test_individual_without_proper_name_not_promoted(self):
        self.assertFalse(_qualifies_for_promotion(self._cl("individual_candidate"), False))

    def test_review_role_not_promoted_without_alias(self):
        self.assertFalse(_qualifies_for_promotion(self._cl("review_role_candidate"), False))

    def test_review_role_promoted_when_alias_resolved(self):
        self.assertTrue(_qualifies_for_promotion(self._cl("review_role_candidate"), True))

    def test_narrator_never_promoted(self):
        self.assertFalse(_qualifies_for_promotion(self._cl("narrator_candidate"), False))
        self.assertFalse(_qualifies_for_promotion(self._cl("narrator_candidate"), True))

    def test_ambiguous_never_promoted(self):
        self.assertFalse(_qualifies_for_promotion(self._cl("ambiguous"), False))

    def test_group_not_promoted_without_alias(self):
        self.assertFalse(_qualifies_for_promotion(self._cl("group_candidate"), False))

    def test_agent_nonhuman_with_proper_name_promoted(self):
        self.assertTrue(
            _qualifies_for_promotion(self._cl("agent_nonhuman", ["Ghost of Christmas Past"]), False)
        )

    def test_agent_nonhuman_without_proper_not_promoted(self):
        self.assertFalse(_qualifies_for_promotion(self._cl("agent_nonhuman"), False))


# ---------------------------------------------------------------------------
# 4. Regression – noisy roles must NOT become canonical (The Trial scenario)
# ---------------------------------------------------------------------------

class TestNoisyRolesExcluded(unittest.TestCase):
    """
    Regression: 'a businessman', 'a child', 'a client', 'a defendant',
    'a doorkeeper' must NOT appear in canonical_characters.json unless
    alias-resolved.
    """

    _NOISY_ROLES = [
        "a businessman",
        "a child",
        "a client",
        "a defendant",
        "a doorkeeper",
    ]

    def _run_layer(self, root: Path, alias_path: Optional[Path] = None) -> List[str]:
        layer = CharacterIdentityLayer(booknlp_root=root, alias_file=alias_path)
        layer.run()
        chars = load_canonical_characters(root)
        return [c["name"] for c in chars]

    def _make_book(self, root: Path, roles: List[str]) -> None:
        characters = []
        for i, role in enumerate(roles, start=1):
            characters.append(_char(i, [], [role]))
        _make_run_dir(root, "test", 0, characters)

    def test_noisy_roles_not_canonical_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_book(root, self._NOISY_ROLES)
            names = self._run_layer(root)
            for role in self._NOISY_ROLES:
                self.assertNotIn(
                    role, names,
                    f"Noisy role {role!r} must NOT become a canonical character",
                )

    def test_noisy_roles_in_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_book(root, self._NOISY_ROLES)
            layer = CharacterIdentityLayer(booknlp_root=root)
            layer.run()
            unresolved = json.loads((root / "unresolved_entities.json").read_text())
            surfaces = {u["surface"] for u in unresolved}
            for role in self._NOISY_ROLES:
                self.assertIn(
                    role, surfaces,
                    f"Noisy role {role!r} must appear in unresolved_entities.json",
                )

    def test_noisy_role_promoted_via_alias(self):
        """A role that is alias-resolved SHOULD become canonical."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_book(root, ["the doorkeeper"])
            alias_path = root / "aliases.json"
            alias_path.write_text(
                json.dumps({"aliases": {"the doorkeeper": "Doorkeeper"}, "chapter_aliases": {}}),
                encoding="utf-8",
            )
            names = self._run_layer(root, alias_path)
            self.assertIn(
                "Doorkeeper", names,
                "Alias-resolved role SHOULD become a canonical character",
            )

    def test_mapping_excludes_noisy_roles(self):
        """local_coref_to_character.json must not contain entries for noisy roles."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Mix: one real character + noisy roles
            chars = [_char(1, ["Josef K.", "K."])]
            for i, role in enumerate(self._NOISY_ROLES, start=2):
                chars.append(_char(i, [], [role]))
            _make_run_dir(root, "test", 0, chars)
            CharacterIdentityLayer(booknlp_root=root).run()

            mapping = load_coref_mapping(root)
            ch0 = mapping.get("0", {})
            # Coref 1 (Josef K.) should be mapped
            self.assertIn("1", ch0)
            # Corefs 2–6 (noisy roles) must NOT be mapped
            for i in range(2, 2 + len(self._NOISY_ROLES)):
                self.assertNotIn(
                    str(i), ch0,
                    f"Noisy role coref {i} must NOT appear in local_coref_to_character.json",
                )


# ---------------------------------------------------------------------------
# 5. Core identity-layer integration tests
# ---------------------------------------------------------------------------

class TestSameCorefDifferentChapters(unittest.TestCase):
    """coref id=5 in ch0 ≠ coref id=5 in ch1 unless same name."""

    def test_distinct_names_not_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(5, ["Scrooge"])])
            _make_run_dir(root, "t", 1, [_char(5, ["Tiny Tim"])])
            CharacterIdentityLayer(booknlp_root=root).run()
            m = load_coref_mapping(root)
            self.assertNotEqual(m.get("0", {}).get("5"), m.get("1", {}).get("5"))


class TestNoiseFiltration(unittest.TestCase):
    def test_christmas_not_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(1, ["Scrooge"]), _char(2, ["Christmas"])])
            CharacterIdentityLayer(booknlp_root=root).run()
            names = {c["name"] for c in load_canonical_characters(root)}
            self.assertIn("Scrooge", names)
            self.assertNotIn("Christmas", names)

    def test_dear_not_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(1, ["Scrooge"]),
                                          _char(2, [], [], ["dear", "dear", "dear"])])
            CharacterIdentityLayer(booknlp_root=root).run()
            self.assertNotIn("2", load_coref_mapping(root).get("0", {}))


class TestMrScrooge(unittest.TestCase):
    """Mr. Scrooge, Scrooge, Uncle Scrooge → one canonical."""

    def test_three_forms_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(1, ["Scrooge", "Mr. Scrooge"])])
            _make_run_dir(root, "t", 1, [_char(3, ["Uncle Scrooge", "Scrooge"])])
            CharacterIdentityLayer(booknlp_root=root).run()
            m = load_coref_mapping(root)
            self.assertEqual(m.get("0", {}).get("1"), m.get("1", {}).get("3"))


class TestMrMrsNoMerge(unittest.TestCase):
    def test_mr_and_mrs_cratchit_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(1, ["Bob Cratchit"]),
                                          _char(2, ["Mrs. Cratchit"])])
            CharacterIdentityLayer(booknlp_root=root).run()
            m = load_coref_mapping(root).get("0", {})
            self.assertNotEqual(m.get("1"), m.get("2"))

    def test_mr_and_mrs_bennet_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(1, ["Mr. Bennet"]),
                                          _char(2, ["Mrs. Bennet"])])
            CharacterIdentityLayer(booknlp_root=root).run()
            m = load_coref_mapping(root).get("0", {})
            self.assertNotEqual(m.get("1"), m.get("2"))


class TestChapterAliasGhost(unittest.TestCase):
    def test_ghost_different_per_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [_char(10, [], ["the Ghost"])])
            _make_run_dir(root, "t", 1, [_char(10, [], ["the Ghost"])])
            alias_path = root / "a.json"
            alias_path.write_text(json.dumps({
                "aliases": {},
                "chapter_aliases": {
                    "0": {"the Ghost": "Jacob Marley"},
                    "1": {"the Ghost": "Ghost of Christmas Past"},
                },
            }), encoding="utf-8")
            CharacterIdentityLayer(booknlp_root=root, alias_file=alias_path).run()
            m = load_coref_mapping(root)
            self.assertNotEqual(m.get("0", {}).get("10"), m.get("1", {}).get("10"))
            names = {c["canonical_id"]: c["name"]
                     for c in load_canonical_characters(root)}
            self.assertIn("Jacob Marley",          names.get(m["0"]["10"], ""))
            self.assertIn("Ghost of Christmas Past", names.get(m["1"]["10"], ""))


class TestNarratorAbstain(unittest.TestCase):
    def test_narrator_not_in_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run_dir(root, "t", 0, [
                _char(1, ["Scrooge"]),
                _char(2, [], [], ["I", "I", "I", "me", "my", "I", "I", "I"], count=8),
            ])
            CharacterIdentityLayer(booknlp_root=root).run()
            self.assertIsNone(load_coref_mapping(root).get("0", {}).get("2"))
            unresolved = json.loads((root / "unresolved_entities.json").read_text())
            self.assertIn("2", {str(u["local_coref_id"]) for u in unresolved})


# ---------------------------------------------------------------------------
# 6. step3b – DIRECT_EVENT extraction
# ---------------------------------------------------------------------------

class TestDirectEventExtraction(unittest.TestCase):
    """Sentence 'Bob took Tim' → DIRECT_EVENT between char_bob_cratchit and char_tiny_tim."""

    def _write_tok_ent(self, run_dir: Path, run_id: str) -> None:
        (run_dir / f"{run_id}.tokens").write_text(
            "token_ID_within_document\tsentence_ID\tlemma\tdependency_relation\t"
            "syntactic_head_ID\tevent\tbyte_onset\tbyte_offset\n"
            "0\t0\tBob\tnsubj\t1\tO\t0\t3\n"
            "1\t0\ttake\troot\t1\tEVENT\t4\t8\n"
            "2\t0\tTim\tobj\t1\tO\t9\t12\n",
            encoding="utf-8",
        )
        (run_dir / f"{run_id}.entities").write_text(
            "COREF\tstart_token\tend_token\tprop\tcat\ttext\n"
            "1\t0\t0\tPROP\tPER\tBob\n"
            "2\t2\t2\tPROP\tPER\tTim\n",
            encoding="utf-8",
        )

    def test_direct_event(self):
        from src.step3b_normalized_predicates import extract_normalized_events

        with tempfile.TemporaryDirectory() as tmp:
            root   = Path(tmp)
            run_id = "booknlp_46_chapter_0000"
            rd     = root / run_id
            rd.mkdir()
            (rd / f"{run_id}.book").write_text(
                json.dumps({"characters": [
                    _char(1, ["Bob Cratchit", "Bob"]),
                    _char(2, ["Tiny Tim", "Tim"]),
                ]}),
                encoding="utf-8",
            )
            self._write_tok_ent(rd, run_id)

            mapping = {"0": {"1": "char_bob_cratchit", "2": "char_tiny_tim"}}
            result  = extract_normalized_events(root, mapping)
            ch0     = result.get("0", {})
            pk      = "char_bob_cratchit||char_tiny_tim"
            self.assertIn(pk, ch0)
            preds = [e["predicate"] for e in ch0[pk]
                     if e["evidence_type"] == "DIRECT_EVENT"]
            self.assertIn("take", preds)


# ---------------------------------------------------------------------------
# 7. step4b – no empty pairs by default
# ---------------------------------------------------------------------------

class TestStep4bNoEmptyPairs(unittest.TestCase):
    def test_no_not_observed_by_default(self):
        from src.step4b_normalized_evidence import build_normalized_pair_evidence

        with tempfile.TemporaryDirectory() as tmp:
            root   = Path(tmp)
            run_id = "booknlp_t_chapter_0000"
            rd     = root / run_id
            rd.mkdir()
            (rd / f"{run_id}.book").write_text(
                json.dumps({"characters": [
                    _char(1, ["Scrooge"]),
                    _char(2, ["Fred"]),
                    _char(3, ["Bob Cratchit"]),
                ]}),
                encoding="utf-8",
            )
            # Scrooge + Fred in sentence 0; Bob never appears
            (rd / f"{run_id}.entities").write_text(
                "COREF\tstart_token\tend_token\tprop\tcat\ttext\n"
                "1\t0\t0\tPROP\tPER\tScrooge\n"
                "2\t1\t1\tPROP\tPER\tFred\n",
                encoding="utf-8",
            )
            (rd / f"{run_id}.tokens").write_text(
                "token_ID_within_document\tsentence_ID\tlemma\tdependency_relation\t"
                "syntactic_head_ID\tevent\tbyte_onset\tbyte_offset\n"
                "0\t0\tScrooge\tnsubj\t2\tO\t0\t7\n"
                "1\t0\tFred\troot\t2\tO\t8\t12\n",
                encoding="utf-8",
            )
            (root / "normalized_event_evidence_by_chapter.json").write_text(
                json.dumps({"0": {}}), encoding="utf-8"
            )

            mapping = {"0": {"1": "char_scrooge", "2": "char_fred",
                             "3": "char_bob_cratchit"}}
            result  = build_normalized_pair_evidence(
                booknlp_root=root, full_mapping=mapping, include_empty=False,
            )
            ch0 = result.get("0", {})
            # Bob has no co-occurrences → no pairs for him
            bob_pairs = [k for k in ch0 if "char_bob_cratchit" in k]
            self.assertEqual(bob_pairs, [])
            # No NOT_OBSERVED entries at all
            for items in ch0.values():
                for ev in items:
                    self.assertNotEqual(ev.get("evidence_type"), "NOT_OBSERVED")


# ---------------------------------------------------------------------------
# 8. CLI --help smoke tests
# ---------------------------------------------------------------------------

class TestCLIHelp(unittest.TestCase):
    def _help(self, module: str) -> None:
        import importlib, io
        from contextlib import redirect_stderr, redirect_stdout
        mod = importlib.import_module(module)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                mod.main(["--help"])
            except SystemExit as e:
                self.assertEqual(e.code, 0)

    def test_step2b_help(self):     self._help("src.step2b_character_identity")
    def test_step3b_help(self):     self._help("src.step3b_normalized_predicates")
    def test_step4b_help(self):     self._help("src.step4b_normalized_evidence")
    def test_graphs_help(self):     self._help("src.build_normalized_graphs")


# ---------------------------------------------------------------------------
# Identity mode tests
# ---------------------------------------------------------------------------

class TestIdentityMode(unittest.TestCase):
    """Verify identity mode metadata is recorded and alias file rules are enforced."""

    def _make_booknlp_root(self, tmp: Path, book_id: str = "testbook") -> Path:
        """Create a minimal booknlp root with one chapter."""
        root = tmp / "booknlp_chapter_output" / book_id
        _make_run_dir(root, book_id, 0, [_char(1, ["Alice"], count=10)])
        return root

    def test_auto_conservative_no_alias_identity_report(self):
        """auto_conservative with no alias file records correct metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_booknlp_root(Path(tmp))
            layer = CharacterIdentityLayer(
                booknlp_root=root, alias_file=None, identity_mode="auto_conservative"
            )
            report = layer.run()

        self.assertEqual(report["identity_mode"], "auto_conservative")
        self.assertFalse(report["manual_aliases_used"])
        self.assertIsNone(report["manual_alias_file"])
        self.assertIn("review_policy", report)

    def test_human_refined_with_alias_identity_report(self):
        """human_refined with an alias file records correct metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_booknlp_root(Path(tmp))
            alias_path = Path(tmp) / "aliases.json"
            alias_path.write_text(
                '{"aliases": {}, "chapter_aliases": {}}', encoding="utf-8"
            )
            layer = CharacterIdentityLayer(
                booknlp_root=root,
                alias_file=alias_path,
                identity_mode="human_refined",
            )
            report = layer.run()

        self.assertEqual(report["identity_mode"], "human_refined")
        self.assertTrue(report["manual_aliases_used"])
        self.assertIsNotNone(report["manual_alias_file"])

    def test_auto_conservative_ignores_existing_alias_file(self):
        """auto_conservative must not load aliases even if an alias file exists
        on disk — the caller is responsible for not passing alias_file=None."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_booknlp_root(Path(tmp))
            # Simulate a data/aliases/<book>.json that exists but should not be used.
            alias_path = Path(tmp) / "data" / "aliases" / "testbook.json"
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            alias_path.write_text(
                '{"aliases": {"alice": "Alice"}, "chapter_aliases": {}}',
                encoding="utf-8",
            )
            # auto_conservative: alias_file=None → no aliases loaded.
            layer = CharacterIdentityLayer(
                booknlp_root=root, alias_file=None, identity_mode="auto_conservative"
            )
            report = layer.run()

        self.assertFalse(report["manual_aliases_used"])
        self.assertIsNone(report["manual_alias_file"])

    def test_step2b_auto_conservative_rejects_alias_file_arg(self):
        """step2b CLI: auto_conservative + --alias-file raises ValueError."""
        import src.step2b_character_identity as s2b
        with self.assertRaises((ValueError, SystemExit)):
            s2b.main([
                "--book-id", "x",
                "--identity-mode", "auto_conservative",
                "--alias-file", "some/path.json",
            ])

    def test_step2b_human_refined_requires_alias_file(self):
        """step2b CLI: human_refined without --alias-file raises ValueError."""
        import src.step2b_character_identity as s2b
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_booknlp_root(Path(tmp))
            with self.assertRaises((ValueError, SystemExit)):
                s2b.main([
                    "--book-id", "testbook",
                    "--booknlp-root", str(root),
                    "--identity-mode", "human_refined",
                ])

    def test_step2b_human_refined_with_alias_file_accepted(self):
        """step2b CLI: human_refined + valid --alias-file completes without error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_booknlp_root(Path(tmp))
            alias_path = Path(tmp) / "aliases.json"
            alias_path.write_text(
                '{"aliases": {}, "chapter_aliases": {}}', encoding="utf-8"
            )
            import src.step2b_character_identity as s2b
            # Should not raise.
            s2b.main([
                "--book-id", "testbook",
                "--booknlp-root", str(root),
                "--identity-mode", "human_refined",
                "--alias-file", str(alias_path),
            ])

    def test_identity_report_written_to_disk(self):
        """identity_report.json on disk includes the new metadata fields."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_booknlp_root(Path(tmp))
            layer = CharacterIdentityLayer(
                booknlp_root=root, alias_file=None, identity_mode="auto_conservative"
            )
            layer.run()
            report = json.loads((root / "identity_report.json").read_text())

        self.assertEqual(report["identity_mode"], "auto_conservative")
        self.assertFalse(report["manual_aliases_used"])
        self.assertIsNone(report["manual_alias_file"])
        self.assertIn("review_policy", report)

    def test_manifest_identity_fields(self):
        """make_book_entry includes identity_mode, manual_aliases_used,
        manual_alias_file."""
        from src.utils.manifest import make_book_entry
        from src.utils.output_paths import OutputPaths

        with tempfile.TemporaryDirectory() as tmp:
            paths = OutputPaths("mybook", output_root=tmp, run_id="r1")
            entry = make_book_entry(
                book_id="mybook",
                input_path="data/raw/x.txt",
                paths=paths,
                identity_mode="auto_conservative",
                manual_alias_file=None,
            )

        self.assertEqual(entry["identity_mode"], "auto_conservative")
        self.assertFalse(entry["manual_aliases_used"])
        self.assertIsNone(entry["manual_alias_file"])

    def test_manifest_human_refined_fields(self):
        """make_book_entry records human_refined metadata correctly."""
        from src.utils.manifest import make_book_entry
        from src.utils.output_paths import OutputPaths

        with tempfile.TemporaryDirectory() as tmp:
            paths = OutputPaths("mybook", output_root=tmp, run_id="r1")
            entry = make_book_entry(
                book_id="mybook",
                input_path="data/raw/x.txt",
                paths=paths,
                identity_mode="human_refined",
                manual_alias_file="data/aliases/mybook.json",
            )

        self.assertEqual(entry["identity_mode"], "human_refined")
        self.assertTrue(entry["manual_aliases_used"])
        self.assertEqual(entry["manual_alias_file"], "data/aliases/mybook.json")

    def test_alias_suggestions_diagnostic_fields(self):
        """alias_suggestions.json contains diagnostic_only=true and applied_to_graph=false."""
        import src.suggest_aliases as sa
        from src.utils.io import write_json

        with tempfile.TemporaryDirectory() as tmp:
            # Minimal booknlp root with canonical_characters.json
            booknlp_root = Path(tmp) / "booknlp"
            booknlp_root.mkdir()
            write_json(booknlp_root / "canonical_characters.json", [])
            out_dir = Path(tmp) / "reports"
            out_dir.mkdir()

            sa.main([
                "--book-id", "testbook",
                "--booknlp-root", str(booknlp_root),
                "--output-dir", str(out_dir),
            ])

            result = json.loads((out_dir / "alias_suggestions.json").read_text())

        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["applied_to_graph"])


if __name__ == "__main__":
    unittest.main()
