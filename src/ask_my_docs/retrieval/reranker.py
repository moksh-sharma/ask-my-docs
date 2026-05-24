from sentence_transformers import CrossEncoder

from ask_my_docs.config import settings
from ask_my_docs.models import Chunk

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank_chunks(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    pairs = [(query, c.text) for c in chunks]
    model = get_reranker()
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
