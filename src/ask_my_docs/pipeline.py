from ask_my_docs.config import settings
from ask_my_docs.generation.answer import generate_answer
from ask_my_docs.index.embeddings import build_embeddings_for_index
from ask_my_docs.index.store import DocumentIndex
from ask_my_docs.ingest.pipeline import ingest_directory
from ask_my_docs.models import AskResponse
from ask_my_docs.retrieval.hybrid import hybrid_retrieve


def ensure_index_built(index: DocumentIndex | None = None) -> DocumentIndex:
    index = index or DocumentIndex.load()
    if index.is_empty():
        raise RuntimeError("Index is empty. Run ingest first.")
    if index.embeddings is None:
        build_embeddings_for_index(index)
        index.save()
    return index


def ask(
    question: str,
    *,
    include_debug: bool = False,
    index: DocumentIndex | None = None,
) -> AskResponse:
    index = ensure_index_built(index)
    chunks, debug = hybrid_retrieve(index, question)
    response = generate_answer(question, chunks)
    if include_debug:
        response.retrieval_debug = debug
    return response


def ingest(docs_dir: str) -> tuple[int, int]:
    from pathlib import Path

    index = DocumentIndex.load()
    doc_count, chunk_count = ingest_directory(Path(docs_dir), index)
    build_embeddings_for_index(index)
    index.save()
    return doc_count, chunk_count
