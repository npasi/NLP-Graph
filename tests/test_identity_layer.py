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


if __name__ == "__main__":
    unittest.main()
