"""
Split full-book character pair interactions into per-chapter JSON files.

This script takes the full-book sentences_by_pair JSON (generated from full-book BookNLP)
and splits it into per-chapter JSONs, where each chapter's JSON contains only the
interactions (sentences) that occur within that chapter.

Steps:
1. Load the full book text to reference byte positions.
2. Load per-chapter BookNLP outputs to determine byte ranges for each chapter.
3. Load the full-book pair JSON.
4. For each sentence in the full-book JSON, find its byte range from full-book tokens.
5. Match the sentence byte range to a chapter based on chapter byte ranges.
6. Group sentences by chapter and character pair.
7. Output per-chapter JSON files in the results folder.

Assumptions:
- Per-chapter BookNLP outputs exist in data/<book_id>/booknlp_<book_id>_chapter_<ch_id>/
- Full-book BookNLP output is in data/booknlp_full/full_<book_id>/
- Chapter IDs are 0-based (chapter_0000, etc.), but output files use 1-based names (chapter_1_interactions.json).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a TSV file into header and rows."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    return header, rows


def _parse_chapter_id(run_dir_name: str) -> Optional[int]:
    """Parse chapter ID from run directory name like 'booknlp_46_chapter_0000'."""
    parts = run_dir_name.split("_")
    if len(parts) < 4 or parts[-2] != "chapter":
        return None
    last = parts[-1]
    if not last.isdigit():
        return None
    return int(last)


def _get_chapter_starts(full_text_path: Path) -> Dict[int, int]:
    """
    Get the starting byte positions of each chapter in the full text.

    Searches for STAVE headers to find chapter boundaries.
    """
    content = full_text_path.read_text(encoding="utf-8", errors="replace")
    chapter_starts: Dict[int, int] = {}
    import re
    # Match STAVE I: etc.
    pattern = re.compile(r'STAVE\s+(I|II|III|IV|V):', re.IGNORECASE)
    for match in pattern.finditer(content):
        stave = match.group(1).upper()
        if stave == 'I':
            ch_id = 0
        elif stave == 'II':
            ch_id = 1
        elif stave == 'III':
            ch_id = 2
        elif stave == 'IV':
            ch_id = 3
        elif stave == 'V':
            ch_id = 4
        else:
            continue
        chapter_starts[ch_id] = match.start()
    # Sort and add end
    sorted_starts = sorted(chapter_starts.items())
    chapter_ranges = {}
    for i, (ch_id, start) in enumerate(sorted_starts):
        if i + 1 < len(sorted_starts):
            end = sorted_starts[i+1][1] - 1
        else:
            end = len(content) - 1
        chapter_ranges[ch_id] = (start, end)
    return chapter_ranges


def _get_sentence_byte_ranges(full_tokens_path: Path) -> Dict[int, Tuple[int, int]]:
    """
    Get byte ranges for each sentence from the full-book tokens.

    Returns a dict: sentence_ID -> (min_byte_onset, max_byte_offset)
    """
    header, rows = _read_tsv(full_tokens_path)
    if not rows:
        return {}
    sent_idx = header.index("sentence_ID")
    onset_idx = header.index("byte_onset")
    offset_idx = header.index("byte_offset")
    sent_ranges: Dict[int, Tuple[int, int]] = {}
    for row in rows:
        sid = int(row[sent_idx])
        onset = int(row[onset_idx])
        offset = int(row[offset_idx])
        if sid not in sent_ranges:
            sent_ranges[sid] = (onset, offset)
        else:
            min_o, max_o = sent_ranges[sid]
            sent_ranges[sid] = (min(min_o, onset), max(max_o, offset))
    return sent_ranges


def _find_chapter_for_sentence(sent_onset: int, sent_offset: int, chapter_ranges: Dict[int, Tuple[int, int]]) -> Optional[int]:
    """
    Find which chapter a sentence belongs to based on its byte range.

    The sentence is assigned to the chapter whose byte range contains the sentence's range.
    If no chapter contains it, return None.
    """
    for ch_id, (ch_start, ch_end) in chapter_ranges.items():
        if sent_onset >= ch_start and sent_offset <= ch_end:
            return ch_id
    return None


def _format_chapter_text(chapter_num: int, data: Dict[str, List[str]]) -> str:
    """Format the chapter data into the specified text format."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"CHAPTER {chapter_num} - PROPER NAME INTERACTIONS")
    lines.append(f"Total unique proper name pairs: {len(data)}")
    lines.append("=" * 70)
    lines.append("")

    # Sort pairs by the pair key (which includes names)
    sorted_pairs = sorted(data.items(), key=lambda x: x[0].lower())

    for idx, (pair_key, sentences) in enumerate(sorted_pairs, 1):
        # Parse pair_key: "name1.id1,name2,id2"
        parts = pair_key.split(",")
        if len(parts) == 3:
            name1 = parts[0].split(".")[0]
            name2 = parts[1]
        else:
            name1, name2 = "Unknown", "Unknown"
        lines.append(f"[PAIR {idx}] {name1} ◄──► {name2}")
        lines.append("-" * 70)
        for sent in sentences:
            # Indent each line of the sentence
            sent_lines = sent.split("\n")
            for s_line in sent_lines:
                lines.append(f"  • {s_line}")
        lines.append("")

    return "\n".join(lines)


def split_full_book_to_chapters(
    *,
    full_book_json: Path,
    full_tokens_path: Path,
    full_text_path: Path,
    output_root: Path,
) -> None:
    """
    Main function to split the full-book JSON into per-chapter text files.

    - Load chapter byte ranges from full text by finding STAVE headers.
    - Load sentence byte ranges from full-book tokens.
    - Load full-book pair JSON.
    - For each pair and each sentence, find the chapter using sentence_id.
    - Group into per-chapter dicts.
    - Write per-chapter text files in the specified format.
    """
    # Step 1: Get chapter byte ranges from full text
    chapter_ranges = _get_chapter_starts(full_text_path)
    if not chapter_ranges:
        raise ValueError("No chapter ranges found. Check STAVE headers in full text.")

    # Step 2: Get sentence byte ranges from full-book tokens
    sent_ranges = _get_sentence_byte_ranges(full_tokens_path)
    if not sent_ranges:
        raise ValueError("No sentence ranges found in full-book tokens.")

    # Step 3: Load full-book pair JSON
    full_data: Dict[str, List[Dict[str, Any]]] = json.loads(full_book_json.read_text(encoding="utf-8"))

    # Step 4: Group by chapter
    chapter_data: Dict[int, Dict[str, List[str]]] = {}
    for pair_key, sentences in full_data.items():
        for sent_info in sentences:
            sent_text = sent_info["text"]
            sid = sent_info["sentence_id"]
            sent_range = sent_ranges.get(sid)
            if not sent_range:
                logger.warning(f"No byte range found for sentence ID {sid}")
                continue
            sent_onset, sent_offset = sent_range
            # Find chapter
            ch_id = _find_chapter_for_sentence(sent_onset, sent_offset, chapter_ranges)
            if ch_id is None:
                logger.warning(f"No chapter found for sentence ID {sid} at {sent_onset}-{sent_offset}")
                continue
            if ch_id not in chapter_data:
                chapter_data[ch_id] = {}
            if pair_key not in chapter_data[ch_id]:
                chapter_data[ch_id][pair_key] = []
            chapter_data[ch_id][pair_key].append(sent_text)

    # Step 5: Write per-chapter text files
    output_root.mkdir(parents=True, exist_ok=True)
    for ch_id, data in chapter_data.items():
        out_path = output_root / f"chapter_{ch_id + 1}_interactions.txt"  # 1-based
        content = _format_chapter_text(ch_id + 1, data)
        out_path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote chapter {ch_id + 1} to {out_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Split full-book pair JSON into per-chapter JSONs.")
    parser.add_argument("--full-book-json", required=True, help="Path to full-book sentences_by_pair JSON.")
    parser.add_argument("--full-tokens", required=True, help="Path to full-book tokens file.")
    parser.add_argument("--full-text", required=True, help="Path to full book text file.")
    parser.add_argument("--output-root", required=True, help="Output directory for per-chapter JSONs.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    split_full_book_to_chapters(
        full_book_json=Path(args.full_book_json),
        full_tokens_path=Path(args.full_tokens),
        full_text_path=Path(args.full_text),
        output_root=Path(args.output_root),
    )


if __name__ == "__main__":
    main()