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


if __name__ == "__main__":
    unittest.main()
