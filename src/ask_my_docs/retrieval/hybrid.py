from ask_my_docs.config import settings
from ask_my_docs.index.store import DocumentIndex
from ask_my_docs.models import Chunk
from ask_my_docs.retrieval.bm25_search import bm25_search
from ask_my_docs.retrieval.reranker import rerank_chunks
from ask_my_docs.retrieval.vector_search import vector_search


def rrf_score(rank: int, k: int | None = None) -> float:
    k = k or settings.rrf_k
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    top_k: int,
    k: int | None = None,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score(rank, k)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def hybrid_retrieve(
    index: DocumentIndex,
    query: str,
    *,
    top_k_per_channel: int | None = None,
    fused_top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> tuple[list[Chunk], dict]:
    top_k_per_channel = top_k_per_channel or settings.hybrid_top_k_per_channel
    fused_top_k = fused_top_k or settings.hybrid_fused_top_k
    rerank_top_k = rerank_top_k or settings.rerank_top_k

    bm25_hits = bm25_search(index, query, top_k_per_channel)
    vec_hits = vector_search(index, query, top_k_per_channel)

    bm25_ids = [cid for cid, _ in bm25_hits]
    vec_ids = [cid for cid, _ in vec_hits]

    fused = reciprocal_rank_fusion([bm25_ids, vec_ids], fused_top_k)
    fused_ids = [cid for cid, _ in fused]

    chunk_by_id = {c.chunk_id: c for c in index.chunks}
    candidates = [chunk_by_id[cid] for cid in fused_ids if cid in chunk_by_id]

    reranked = rerank_chunks(query, candidates, rerank_top_k)

    debug = {
        "bm25_top": bm25_hits[:10],
        "vector_top": vec_hits[:10],
        "fused_top": fused[:10],
        "reranked_ids": [c.chunk_id for c in reranked],
    }
    return reranked, debug
