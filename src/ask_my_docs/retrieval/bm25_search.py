from ask_my_docs.index.store import DocumentIndex, _tokenize_for_bm25


def bm25_search(index: DocumentIndex, query: str, top_k: int) -> list[tuple[str, float]]:
    if index.bm25 is None or not index.chunks:
        return []

    tokens = _tokenize_for_bm25(query)
    scores = index.bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(index.chunks[i].chunk_id, float(s)) for i, s in ranked if s > 0]
