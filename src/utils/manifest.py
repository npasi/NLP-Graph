from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.output_paths import OutputPaths


def _git_info() -> tuple[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"


def _new_manifest(run_id: str) -> dict:
    branch, commit = _git_info()
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_commit": commit,
        "pipeline": "NLP-Graph",
        "books": [],
    }


def make_book_entry(
    book_id: str,
    input_path: str,
    paths: OutputPaths,
    status: str = "completed",
    error: str | None = None,
) -> dict:
    entry: dict = {
        "book_id": book_id,
        "input": input_path,
        "status": status,
        "outputs": {
            "chapters": str(paths.book_chapters_dir),
            "booknlp": str(paths.booknlp_book_dir),
            "reports": str(paths.reports_book_dir),
            "graphs": str(paths.graphs_chapters_dir),
            "ml": str(paths.ml_book_dir),
        },
    }
    if error:
        entry["error"] = error
    return entry


def write_manifest(run_dir: Path, run_id: str, book_entry: dict) -> Path:
    """Write or update manifest.json inside run_dir with the given book entry."""
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = _new_manifest(run_id)
    else:
        manifest = _new_manifest(run_id)

    books: list[dict] = manifest.setdefault("books", [])
    for i, b in enumerate(books):
        if b.get("book_id") == book_entry.get("book_id"):
            books[i] = book_entry
            break
    else:
        books.append(book_entry)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path
