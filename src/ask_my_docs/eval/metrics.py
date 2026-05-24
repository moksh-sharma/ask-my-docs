import re

from ask_my_docs.models import AskResponse, Chunk


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    if not expected_doc_ids:
        return 1.0
    top = set(retrieved_doc_ids[:k])
    expected = set(expected_doc_ids)
    return 1.0 if expected & top else 0.0


def faithfulness_score(answer: str, context_chunks: list[Chunk]) -> float:
    """Heuristic: fraction of answer tokens found in retrieved context."""
    if not answer or not context_chunks:
        return 0.0

    norm_answer = re.sub(r"\[\d+\]", "", answer.lower())
    answer_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", norm_answer)}
    if not answer_tokens:
        return 1.0

    context = " ".join(c.text.lower() for c in context_chunks)
    supported = sum(1 for t in answer_tokens if t in context)
    return supported / len(answer_tokens)


def citation_accuracy(response: AskResponse, num_chunks: int) -> float:
    if num_chunks == 0:
        return 0.0
    if not response.citations:
        return 0.0

    valid = 0
    for c in response.citations:
        if 1 <= c.id <= num_chunks:
            valid += 1
    return valid / len(response.citations)


def answer_overlap_with_reference(answer: str, reference: str | None) -> float:
    if not reference:
        return 1.0
    ref_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", reference.lower())}
    ans_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", answer.lower())}
    if not ref_tokens:
        return 1.0
    return len(ref_tokens & ans_tokens) / len(ref_tokens)
