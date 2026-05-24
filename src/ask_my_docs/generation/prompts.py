SYSTEM_PROMPT = """You are a precise documentation assistant. Answer ONLY using the provided context.
Rules:
1. Every factual claim must include an inline citation like [1] or [2] matching context block numbers.
2. If the answer is not in the context, respond: "I don't have enough information in the provided documents."
3. Do not invent facts, URLs, or policy details.
4. Return valid JSON with keys: answer (string), citations (array of {id: int, quote: string|null}).
"""


def format_context_blocks(chunks: list) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = f"[{i}] source={meta.source}"
        if meta.section:
            header += f" section={meta.section}"
        if meta.page is not None:
            header += f" page={meta.page}"
        if meta.url:
            header += f" url={meta.url}"
        lines.append(header)
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def build_user_prompt(question: str, context: str) -> str:
    return f"""Context:
{context}

Question: {question}

Respond with JSON only."""
