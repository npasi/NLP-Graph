from pathlib import Path


def clean_gutenberg_text(input_path: Path, output_path: Path) -> Path:
    text = input_path.read_text(encoding="utf-8", errors="replace")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start = text.find(start_marker)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0

    end = text.find(end_marker)
    if end == -1:
        end = len(text)

    cleaned = text[start:end].strip()

    output_path.write_text(cleaned, encoding="utf-8")
    return output_path
