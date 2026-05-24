from pathlib import Path

from ask_my_docs.config import settings
from ask_my_docs.index.store import DocumentIndex
from ask_my_docs.ingest.chunker import chunk_text
from ask_my_docs.ingest.parser import load_document


def ingest_directory(docs_dir: Path, index: DocumentIndex | None = None) -> tuple[int, int]:
    index = index or DocumentIndex.load()
    doc_count = 0
    chunk_count = 0

    patterns = ("*.md", "*.markdown", "*.txt")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(docs_dir.glob(pattern)))

    for path in files:
        doc_id, title, content = load_document(path)
        chunks = chunk_text(
            content,
            doc_id=doc_id,
            title=title,
            source=str(path.name),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            url=str(path),
        )
        index.add_document_chunks(doc_id, chunks)
        doc_count += 1
        chunk_count += len(chunks)

    index.save()
    return doc_count, chunk_count
