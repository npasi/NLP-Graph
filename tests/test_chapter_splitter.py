"""Tests for src/chapter_splitter.py using synthetic texts."""

import unittest

from src.chapter_splitter import split_into_chapters


# ── helpers ──────────────────────────────────────────────────────────────────

_BODY = ("word " * 600).strip()   # 600 words — above TOC_BODY_MAX_TOKENS=500
_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK TESTBOOK ***\n"
_GUTENBERG_END = "\n*** END OF THE PROJECT GUTENBERG EBOOK TESTBOOK ***\n"
_LICENSE_SECTIONS = """
SECTION 1. GENERAL TERMS OF USE

You may use this eBook for any purpose.

SECTION 2. INFORMATION ABOUT THE MISSION OF PROJECT GUTENBERG

The mission of Project Gutenberg is to encourage creation of ebooks.

SECTION 3. INFORMATION ABOUT THE PROJECT GUTENBERG LITERARY ARCHIVE

Project Gutenberg Literary Archive Foundation.
"""


def _wrap_gutenberg(body: str) -> str:
    return _GUTENBERG_START + body + _GUTENBERG_END + _LICENSE_SECTIONS


# ── 1. Standard word headings ─────────────────────────────────────────────────

class TestWordHeadings(unittest.TestCase):

    def _check(self, text: str, expected_titles: list[str]) -> None:
        chapters = split_into_chapters(text)
        titles = [ch["title"] for ch in chapters]
        self.assertEqual(titles, expected_titles, f"Got: {titles}")

    def test_chapter_one_style(self):
        text = _wrap_gutenberg(
            "\nChapter One\n\n" + _BODY +
            "\n\nChapter Two\n\n" + _BODY +
            "\n\nChapter Three\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]["title"], "Chapter One")
        self.assertEqual(chapters[1]["title"], "Chapter Two")
        self.assertEqual(chapters[2]["title"], "Chapter Three")

    def test_chapter_arabic_style(self):
        text = _wrap_gutenberg(
            "\nChapter 1\n\n" + _BODY +
            "\n\nChapter 2\n\n" + _BODY +
            "\n\nChapter 3\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]["title"], "Chapter 1")

    def test_chapter_all_caps_arabic(self):
        text = _wrap_gutenberg(
            "\nCHAPTER 1\n\n" + _BODY +
            "\n\nCHAPTER 2\n\n" + _BODY +
            "\n\nCHAPTER 3\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertIn("1", chapters[0]["title"])

    def test_chapter_roman_word(self):
        text = _wrap_gutenberg(
            "\nCHAPTER I\n\n" + _BODY +
            "\n\nCHAPTER II\n\n" + _BODY +
            "\n\nCHAPTER III\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]["title"], "Chapter I")

    def test_chapter_with_inline_title(self):
        text = _wrap_gutenberg(
            "\nChapter 1: The Beginning\n\n" + _BODY +
            "\n\nChapter 2: The Middle\n\n" + _BODY +
            "\n\nChapter 3: The End\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]["title"], "Chapter 1: The Beginning")


# ── 2. Roman numeral only headings ───────────────────────────────────────────

class TestRomanNumeralOnlyHeadings(unittest.TestCase):

    def test_bare_roman_numerals(self):
        text = (
            "\nI\n\n" + _BODY +
            "\n\nII\n\n" + _BODY +
            "\n\nIII\n\n" + _BODY +
            "\n\nIV\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertGreaterEqual(len(chapters), 3)
        self.assertIn("I", chapters[0]["raw_heading"])


# ── 3. ALL CAPS literary headings (Jekyll-style) ─────────────────────────────

class TestAllCapsLiteraryHeadings(unittest.TestCase):

    def _jekyll_text(self) -> str:
        return _wrap_gutenberg(
            "\n\nSTORY OF THE DOOR\n\n" + _BODY +
            "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY +
            "\n\nDR. JEKYLL WAS QUITE AT EASE\n\n" + _BODY +
            "\n\nTHE CAREW MURDER CASE\n\n" + _BODY +
            "\n\nINCIDENT OF THE LETTER\n\n" + _BODY
        )

    def test_jekyll_chapters_detected(self):
        chapters = split_into_chapters(self._jekyll_text())
        titles = [ch["title"] for ch in chapters]
        self.assertEqual(len(chapters), 5, f"Got {len(chapters)} chapters: {titles}")

    def test_jekyll_titles_preserved(self):
        chapters = split_into_chapters(self._jekyll_text())
        self.assertEqual(chapters[0]["title"], "STORY OF THE DOOR")
        self.assertEqual(chapters[1]["title"], "SEARCH FOR MR. HYDE")
        self.assertEqual(chapters[2]["title"], "DR. JEKYLL WAS QUITE AT EASE")

    def test_jekyll_raw_heading_preserved(self):
        chapters = split_into_chapters(self._jekyll_text())
        self.assertEqual(chapters[0]["raw_heading"], "STORY OF THE DOOR")

    def test_jekyll_bodies_not_empty(self):
        chapters = split_into_chapters(self._jekyll_text())
        for ch in chapters:
            self.assertGreater(ch["num_tokens"], 0, f"Empty body for {ch['title']}")

    def test_too_few_allcaps_headings_falls_back(self):
        # Only 2 headings — below MIN_HEADINGS_FOR_VALID_SCHEME=3 → full text fallback
        text = _wrap_gutenberg(
            "\n\nSTORY OF THE DOOR\n\n" + _BODY +
            "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Full text")


# ── 4. Gutenberg footer sections must NOT be detected as chapters ─────────────

class TestGutenbergFooterIgnored(unittest.TestCase):

    def test_section_headings_not_chapters(self):
        # Text has only Gutenberg license sections, no real chapter headings.
        text = _GUTENBERG_START + "\n\nSome introductory prose.\n\n" + _GUTENBERG_END + _LICENSE_SECTIONS
        chapters = split_into_chapters(text)
        # Should fall back to full text — not split on SECTION 1, SECTION 2
        titles = [ch["title"] for ch in chapters]
        for t in titles:
            self.assertNotIn("Section", t, f"License section leaked into chapters: {titles}")
            self.assertNotIn("SECTION", t, f"License section leaked into chapters: {titles}")

    def test_general_terms_of_use_not_chapter(self):
        text = _wrap_gutenberg("\n\nSome story prose without headings.\n\n" + _BODY)
        chapters = split_into_chapters(text)
        for ch in chapters:
            self.assertNotIn("GENERAL TERMS", ch["title"])

    def test_real_chapters_plus_footer(self):
        # Real book chapters should be detected; footer sections ignored.
        text = _wrap_gutenberg(
            "\nChapter 1\n\n" + _BODY +
            "\n\nChapter 2\n\n" + _BODY +
            "\n\nChapter 3\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 3)
        for ch in chapters:
            self.assertNotIn("Section", ch["title"])

    def test_jekyll_chapters_plus_footer(self):
        # Jekyll-style ALL CAPS headings; license sections must not appear.
        text = _wrap_gutenberg(
            "\n\nSTORY OF THE DOOR\n\n" + _BODY +
            "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY +
            "\n\nDR. JEKYLL WAS QUITE AT EASE\n\n" + _BODY +
            "\n\nTHE CAREW MURDER CASE\n\n" + _BODY +
            "\n\nINCIDENT OF THE LETTER\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        titles = [ch["title"] for ch in chapters]
        self.assertEqual(len(chapters), 5, f"Titles: {titles}")
        for t in titles:
            self.assertNotIn("SECTION", t)
            self.assertNotIn("GENERAL TERMS", t)
            self.assertNotIn("PROJECT GUTENBERG", t)


# ── 5. Title-page all-caps lines must NOT be detected as chapters ─────────────

class TestTitlePageIgnored(unittest.TestCase):

    def test_title_line_not_chapter(self):
        # The title appears before any chapters and has a zero-length body
        # (followed immediately by the author line), so TOC-zone filtering
        # should remove it.
        text = _wrap_gutenberg(
            "THE STRANGE CASE OF DR. JEKYLL AND MR. HYDE\n\n"
            "BY ROBERT LOUIS STEVENSON\n\n"
            + "\n\nSTORY OF THE DOOR\n\n" + _BODY
            + "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY
            + "\n\nDR. JEKYLL WAS QUITE AT EASE\n\n" + _BODY
            + "\n\nTHE CAREW MURDER CASE\n\n" + _BODY
            + "\n\nINCIDENT OF THE LETTER\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        titles = [ch["title"] for ch in chapters]
        self.assertNotIn("THE STRANGE CASE OF DR. JEKYLL AND MR. HYDE", titles,
                         f"Title page leaked into chapters: {titles}")

    def test_author_line_not_chapter(self):
        text = _wrap_gutenberg(
            "GREAT EXPECTATIONS\n\n"
            "BY CHARLES DICKENS\n\n"
            + "\n\nChapter 1\n\n" + _BODY
            + "\n\nChapter 2\n\n" + _BODY
            + "\n\nChapter 3\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        titles = [ch["title"] for ch in chapters]
        for t in titles:
            self.assertNotIn("BY CHARLES DICKENS", t)

    def test_metadata_by_line_filtered(self):
        # "BY ROBERT LOUIS STEVENSON" starts with "BY" → metadata filter
        text = _wrap_gutenberg(
            "\n\nBY ROBERT LOUIS STEVENSON\n\n"
            + "\n\nSTORY OF THE DOOR\n\n" + _BODY
            + "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY
            + "\n\nDR. JEKYLL WAS QUITE AT EASE\n\n" + _BODY
            + "\n\nTHE CAREW MURDER CASE\n\n" + _BODY
            + "\n\nINCIDENT OF THE LETTER\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        titles = [ch["title"] for ch in chapters]
        self.assertNotIn("BY ROBERT LOUIS STEVENSON", titles,
                         f"Metadata line leaked into chapters: {titles}")


# ── 6. Fallback to full text ──────────────────────────────────────────────────

class TestFallback(unittest.TestCase):

    def test_no_headings_returns_full_text(self):
        text = "This is just a plain text with no chapter headings at all. " * 50
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Full text")
        self.assertEqual(chapters[0]["chapter_id"], 0)

    def test_single_heading_falls_back(self):
        # Only one heading — below MIN_HEADINGS_FOR_VALID_SCHEME
        text = _wrap_gutenberg("\nChapter 1\n\n" + _BODY)
        chapters = split_into_chapters(text)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Full text")

    def test_empty_text_falls_back(self):
        chapters = split_into_chapters("")
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Full text")


# ── 7. Nested PART + Chapter (Madame Bovary style) ───────────────────────────

class TestNestedPartChapter(unittest.TestCase):
    """Test A – nested PART containers with resetting chapter numbers."""

    def _nested_text(self) -> str:
        return _wrap_gutenberg(
            "\nPART I\n\nChapter One\n\n" + _BODY +
            "\n\nChapter Two\n\n" + _BODY +
            "\n\nPART II\n\nChapter One\n\n" + _BODY +
            "\n\nChapter Two\n\n" + _BODY
        )

    def test_nested_produces_four_sections(self):
        chapters = split_into_chapters(self._nested_text())
        self.assertEqual(len(chapters), 4, f"Titles: {[ch['title'] for ch in chapters]}")

    def test_nested_titles_include_parent_context(self):
        chapters = split_into_chapters(self._nested_text())
        titles = [ch["title"] for ch in chapters]
        self.assertIn("Part I", titles[0])
        self.assertIn("Chapter One", titles[0])
        self.assertIn("Part II", titles[2])
        self.assertIn("Chapter One", titles[2])

    def test_nested_no_zero_token_sections(self):
        chapters = split_into_chapters(self._nested_text())
        for ch in chapters:
            self.assertGreater(ch["num_tokens"], 0, f"Zero-token section: {ch['title']}")

    def test_nested_part_headings_not_standalone_sections(self):
        chapters = split_into_chapters(self._nested_text())
        titles = [ch["title"] for ch in chapters]
        # "Part I" alone should not appear as a section title.
        self.assertNotIn("Part I", titles)
        self.assertNotIn("Part II", titles)


# ── 8. Nested VOLUME + BOOK + CHAPTER (3-level) ──────────────────────────────

class TestNestedVolumeBookChapter(unittest.TestCase):
    """Test B – three-level VOLUME → BOOK → CHAPTER hierarchy."""

    def _text(self) -> str:
        return _wrap_gutenberg(
            "\nVOLUME I\n\nBOOK FIRST\n\nCHAPTER I\n\n" + _BODY +
            "\n\nCHAPTER II\n\n" + _BODY +
            "\n\nBOOK SECOND\n\nCHAPTER I\n\n" + _BODY +
            "\n\nVOLUME II\n\nBOOK FIRST\n\nCHAPTER I\n\n" + _BODY
        )

    def test_four_chapter_sections(self):
        chapters = split_into_chapters(self._text())
        self.assertEqual(len(chapters), 4, f"Titles: {[ch['title'] for ch in chapters]}")

    def test_no_standalone_volume_or_book(self):
        chapters = split_into_chapters(self._text())
        titles = [ch["title"] for ch in chapters]
        self.assertNotIn("Volume I", titles)
        self.assertNotIn("Book First", titles)
        self.assertNotIn("Book Second", titles)

    def test_all_sections_nonempty(self):
        chapters = split_into_chapters(self._text())
        for ch in chapters:
            self.assertGreater(ch["num_tokens"], 0)


# ── 9. Empty heading suppression (Jekyll HASTIE LANYON style) ────────────────

class TestEmptyHeadingSuppression(unittest.TestCase):
    """Test D – zero-token sections are suppressed."""

    def test_zero_token_signature_dropped(self):
        text = _wrap_gutenberg(
            "\n\nSTORY OF THE DOOR\n\n" + _BODY +
            "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY +
            "\n\nDR. JEKYLL WAS QUITE AT EASE\n\n" + _BODY +
            "\n\nTHE CAREW MURDER CASE\n\n" + _BODY +
            "\n\nINCIDENT OF THE LETTER\n\n" + _BODY +
            # Signature line immediately followed by next real heading (0-token body).
            "\n\nHASTIE LANYON.\n\n"
            "\n\nHENRY JEKYLL'S FULL STATEMENT\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        titles = [ch["title"] for ch in chapters]
        # HASTIE LANYON. has a zero-body and must be suppressed.
        self.assertNotIn("HASTIE LANYON.", titles)
        # All remaining sections must have content.
        for ch in chapters:
            self.assertGreater(ch["num_tokens"], 0, f"Zero-token: {ch['title']}")

    def test_nonempty_sections_preserved(self):
        text = _wrap_gutenberg(
            "\n\nSTORY OF THE DOOR\n\n" + _BODY +
            "\n\nSEARCH FOR MR. HYDE\n\n" + _BODY +
            "\n\nDR. JEKYLL WAS QUITE AT EASE\n\n" + _BODY +
            "\n\nTHE CAREW MURDER CASE\n\n" + _BODY +
            "\n\nINCIDENT OF THE LETTER\n\n" + _BODY +
            "\n\nHASTIE LANYON.\n\n"
            "\n\nHENRY JEKYLL'S FULL STATEMENT\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertIn("STORY OF THE DOOR", [ch["title"] for ch in chapters])
        self.assertIn("HENRY JEKYLL'S FULL STATEMENT", [ch["title"] for ch in chapters])


# ── 10. Dramatic text: ACT + SCENE ───────────────────────────────────────────

class TestDramaticActScene(unittest.TestCase):
    """Test E – ACT is container, SCENE is leaf; speaker names ignored."""

    def _drama_text(self) -> str:
        return _wrap_gutenberg(
            "\nACT I\n\nSCENE I\n\n" + _BODY +
            "\n\nSCENE II\n\n" + _BODY +
            "\n\nACT II\n\nSCENE I\n\n" + _BODY
        )

    def test_three_scene_sections(self):
        chapters = split_into_chapters(self._drama_text())
        self.assertEqual(len(chapters), 3, f"Titles: {[ch['title'] for ch in chapters]}")

    def test_scene_titles_have_act_context(self):
        chapters = split_into_chapters(self._drama_text())
        self.assertIn("Act I", chapters[0]["title"])
        self.assertIn("Scene I", chapters[0]["title"])
        self.assertIn("Act II", chapters[2]["title"])

    def test_no_standalone_act(self):
        chapters = split_into_chapters(self._drama_text())
        for ch in chapters:
            self.assertNotEqual(ch["title"].strip(), "Act I")
            self.assertNotEqual(ch["title"].strip(), "Act II")


# ── 11. Mixed section types: LETTER + CHAPTER ────────────────────────────────

class TestMixedLetterChapter(unittest.TestCase):
    """Test F – LETTER sections followed by CHAPTER sections."""

    def _text(self) -> str:
        return _wrap_gutenberg(
            "\nLetter I\n\n" + _BODY +
            "\n\nLetter II\n\n" + _BODY +
            "\n\nChapter I\n\n" + _BODY
        )

    def test_three_sections(self):
        chapters = split_into_chapters(self._text())
        self.assertEqual(len(chapters), 3, f"Titles: {[ch['title'] for ch in chapters]}")

    def test_all_sections_nonempty(self):
        chapters = split_into_chapters(self._text())
        for ch in chapters:
            self.assertGreater(ch["num_tokens"], 0)


# ── 12. TOC duplicate suppression ────────────────────────────────────────────

class TestTOCDuplicateSuppression(unittest.TestCase):
    """Test G – TOC headings are filtered, real headings kept."""

    def test_toc_entries_not_duplicated(self):
        # TOC entries have tiny bodies (immediately followed by next heading).
        # Real entries have _BODY (~600 words) as body.
        toc_section = (
            "\nPART I\nChapter One\nChapter Two\n"
            "PART II\nChapter One\nChapter Two\n\n"
        )
        real_section = (
            "PART I\n\nChapter One\n\n" + _BODY +
            "\n\nChapter Two\n\n" + _BODY +
            "\n\nPART II\n\nChapter One\n\n" + _BODY +
            "\n\nChapter Two\n\n" + _BODY
        )
        text = _wrap_gutenberg(toc_section + real_section)
        chapters = split_into_chapters(text)
        # Should have 4 real chapters, not 8 (TOC + real).
        self.assertEqual(len(chapters), 4, f"Got {len(chapters)}: {[ch['title'] for ch in chapters]}")


# ── 13. Front matter not included as chapters ─────────────────────────────────

class TestFrontMatterExcluded(unittest.TestCase):
    """Test H – PREFACE and ETYMOLOGY (single-word, non-section-word headings)
    do not leak into the chapter list when real CHAPTER headings follow."""

    def test_front_matter_titles_not_chapters(self):
        # PREFACE and ETYMOLOGY are 1-word all-caps → not detected by either
        # the word or allcaps detector.  Three CHAPTER headings → word scheme wins.
        text = _wrap_gutenberg(
            "\nPREFACE\n\n" + "preface text " * 30 +
            "\n\nETYMOLOGY\n\n" + "etymology text " * 30 +
            "\n\nCHAPTER 1\n\n" + _BODY +
            "\n\nCHAPTER 2\n\n" + _BODY +
            "\n\nCHAPTER 3\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        titles = [ch["title"].upper() for ch in chapters]
        for t in titles:
            self.assertNotIn("PREFACE", t)
            self.assertNotIn("ETYMOLOGY", t)
        # Should detect the 3 real chapters.
        self.assertEqual(len(chapters), 3)


# ── 14. Bare roman numerals still work ───────────────────────────────────────

class TestBareRomanNumerals(unittest.TestCase):
    """Test I – bare roman numeral headings continue to work after changes."""

    def test_bare_roman_three_sections(self):
        text = (
            "\nI\n\n" + _BODY +
            "\n\nII\n\n" + _BODY +
            "\n\nIII\n\n" + _BODY +
            "\n\nIV\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertGreaterEqual(len(chapters), 3)

    def test_bare_roman_not_confused_with_nested(self):
        # Roman-only text should NOT trigger the nested detector (no container words).
        text = (
            "\nI\n\n" + _BODY +
            "\n\nII\n\n" + _BODY +
            "\n\nIII\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        self.assertGreaterEqual(len(chapters), 3)
        # Titles should just be the numeral, not "Part I - Chapter I" style.
        self.assertNotIn(" - ", chapters[0]["title"])


# ── 15. Repeated chapter numbering inside different containers ────────────────

class TestRepeatedChapterNumbering(unittest.TestCase):
    """Test C – Chapter I can appear under multiple PART containers."""

    def test_chapter_i_repeats_across_parts(self):
        text = _wrap_gutenberg(
            "\nPART I\n\nChapter I\n\n" + _BODY +
            "\n\nChapter II\n\n" + _BODY +
            "\n\nPART II\n\nChapter I\n\n" + _BODY +
            "\n\nChapter II\n\n" + _BODY +
            "\n\nPART III\n\nChapter I\n\n" + _BODY
        )
        chapters = split_into_chapters(text)
        # 2 + 2 + 1 = 5 sections.
        self.assertEqual(len(chapters), 5, f"Titles: {[ch['title'] for ch in chapters]}")
        # Titles must distinguish the repeated Chapter I.
        titles = [ch["title"] for ch in chapters]
        # There should be three sections whose last segment is "Chapter I"
        # (use word-boundary match to exclude "Chapter II", "Chapter III"…).
        import re as _re
        ch_ones = [t for t in titles if _re.search(r'\bChapter I\b', t)]
        self.assertEqual(len(ch_ones), 3, f"Chapter I occurrences: {ch_ones}")
        # Each must have a different parent prefix.
        self.assertEqual(len(set(ch_ones)), 3)


if __name__ == "__main__":
    unittest.main()
