import json
import re

import httpx

from ask_my_docs.config import settings
from ask_my_docs.generation.prompts import SYSTEM_PROMPT, build_user_prompt, format_context_blocks
from ask_my_docs.models import AskResponse, Chunk, CitationRef


class CitationValidationError(ValueError):
    pass


def validate_citations(
    answer: str,
    citations: list[CitationRef],
    num_context_blocks: int,
) -> None:
    if num_context_blocks == 0:
        raise CitationValidationError("No context blocks available")

    if not citations:
        raise CitationValidationError("Response must include at least one citation")

    for ref in citations:
        if ref.id < 1 or ref.id > num_context_blocks:
            raise CitationValidationError(
                f"Citation id {ref.id} out of range 1..{num_context_blocks}"
            )

    inline_ids = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    cited_ids = {c.id for c in citations}
    if inline_ids and not inline_ids.issubset(cited_ids | set(range(1, num_context_blocks + 1))):
        # Allow inline refs that map to valid context indices
        for iid in inline_ids:
            if iid < 1 or iid > num_context_blocks:
                raise CitationValidationError(f"Inline citation [{iid}] out of range")


def _parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _call_ollama(system: str, user: str) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            url,
            json={
                "model": settings.ollama_model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Ollama returned empty response: {data}")
        return content


def _mock_generate(question: str, chunks: list[Chunk]) -> dict:
    """Deterministic extractive fallback for CI / offline use."""
    if not chunks:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "citations": [],
        }

    best = chunks[0]
    sentence = best.text.split(".")[0].strip()
    if not sentence:
        sentence = best.text[:160].strip()
    if sentence and not sentence.endswith("."):
        sentence += "."
    answer = f"{sentence} [1]"
    return {
        "answer": answer,
        "citations": [{"id": 1, "quote": sentence[:120]}],
    }


def generate_answer(
    question: str,
    chunks: list[Chunk],
    *,
    max_retries: int = 2,
) -> AskResponse:
    context = format_context_blocks(chunks)
    user_prompt = build_user_prompt(question, context)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if settings.mock_llm:
                payload = _mock_generate(question, chunks)
            else:
                raw = _call_ollama(SYSTEM_PROMPT, user_prompt)
                payload = _parse_llm_json(raw)

            answer = payload.get("answer", "").strip()
            raw_citations = payload.get("citations", [])
            citations = [CitationRef.model_validate(c) for c in raw_citations]

            validate_citations(answer, citations, len(chunks))

            sources = []
            for i, chunk in enumerate(chunks, start=1):
                sources.append(
                    {
                        "index": i,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.metadata.doc_id,
                        "title": chunk.metadata.title,
                        "source": chunk.metadata.source,
                        "section": chunk.metadata.section,
                    }
                )

            return AskResponse(answer=answer, citations=citations, sources=sources)
        except (CitationValidationError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            user_prompt = (
                user_prompt
                + f"\n\nPrevious response failed validation ({exc}). "
                "Fix citations: use only [1]..[n] and include citations array."
            )

    raise CitationValidationError(f"Failed to produce valid cited answer: {last_error}")
