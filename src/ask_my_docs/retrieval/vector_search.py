import numpy as np

from ask_my_docs.index.embeddings import embed_texts
from ask_my_docs.index.store import DocumentIndex


def vector_search(index: DocumentIndex, query: str, top_k: int) -> list[tuple[str, float]]:
    if index.embeddings is None or not index.chunks:
        return []

    q_vec = embed_texts([query])[0]
    scores = index.embeddings @ q_vec
    ranked = np.argsort(-scores)[:top_k]
    return [(index.chunks[i].chunk_id, float(scores[i])) for i in ranked]
