import hashlib
import re
from pathlib import Path


def doc_id_from_path(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def parse_markdown(path: Path) -> tuple[str, str]:
    """Return (title, body) from a markdown file."""
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").replace("-", " ").title()
    match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        title = match.group(1).strip()
    return title, text.strip()


def parse_text_file(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return title, text.strip()


def load_document(path: Path) -> tuple[str, str, str]:
    """Return (doc_id, title, content)."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        title, content = parse_markdown(path)
    elif suffix in {".txt", ".text"}:
        title, content = parse_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return doc_id_from_path(path), title, content
