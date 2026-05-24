from ask_my_docs.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_prefers_items_in_both_lists():
    list_a = ["a", "b", "c"]
    list_b = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([list_a, list_b], top_k=3)
    ids = [cid for cid, _ in fused]
    assert ids[0] in {"a", "b"}
    assert "b" in ids
    assert "a" in ids
