from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def project_root() -> Path:
    # src/utils/io.py -> src/utils -> src -> project root
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding=encoding)


def read_text(path: Path, *, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_slug(s: str) -> str:
    # Minimal slug for filenames.
    import re

    out = re.sub(r"[^A-Za-z0-9]+", "_", (s or "").strip().lower()).strip("_")
    return out or "book"

