import hashlib
import re

from ask_my_docs.models import Chunk, ChunkMetadata


def _tokenize_approx(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def chunk_text(
    text: str,
    *,
    doc_id: str,
    title: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    section: str | None = None,
    page: int | None = None,
    url: str | None = None,
) -> list[Chunk]:
    """Structure-aware-ish chunking: split on headings first, then token windows."""
    sections = _split_by_headings(text)
    chunks: list[Chunk] = []

    for sec_title, sec_body in sections:
        tokens = _tokenize_approx(sec_body)
        if not tokens:
            continue

        step = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + chunk_size]
            if not window:
                continue
            chunk_text_str = _detokenize(window)
            chunk_id = hashlib.sha256(
                f"{doc_id}:{sec_title}:{start}:{chunk_text_str[:80]}".encode()
            ).hexdigest()[:20]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    metadata=ChunkMetadata(
                        doc_id=doc_id,
                        title=title,
                        source=source,
                        section=sec_title or section,
                        page=page,
                        url=url,
                    ),
                )
            )
            if start + chunk_size >= len(tokens):
                break

    return chunks


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    heading_re = re.compile(r"^#{1,3}\s+(.+)$")

    for line in lines:
        m = heading_re.match(line)
        if m:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    if not sections:
        return [(None, text)]
    return sections
