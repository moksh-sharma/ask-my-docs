# Ask My Docs

Production-style **Ask My Docs** RAG with hybrid retrieval (BM25 + vectors), cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline.

## Architecture

```
Documents → chunk + embed → [BM25 index | Vector index]
                                    ↓
User question → hybrid retrieve (RRF) → cross-encoder rerank → LLM + citations
                                    ↓
                         Golden-set eval gates deploy (GitHub Actions)
```

## Quick start

```bash
cd ask-my-docs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ingest sample docs and build index
ask-docs-ingest --docs-dir data/sample_docs

# Ask a question (uses MOCK_LLM without Ollama)
export MOCK_LLM=true
ask-docs-ask "When are invoices sent?"

# Run evaluation suite (same gate as CI)
ask-docs-eval
```

### API server

```bash
export MOCK_LLM=true
ask-docs-ingest
uvicorn apps.api.main:app --reload --app-dir .
```

Endpoints:

- `GET /health`
- `POST /ingest?docs_dir=data/sample_docs`
- `POST /ask` — body: `{"question": "...", "include_debug": false}`

### Ollama generation

```bash
# Pull a model (once)
ollama pull llama3.2

cp .env.example .env
export MOCK_LLM=false
ask-docs-ask "How far in advance must renewals be requested?"
```

Configure via env: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.2`).

## Project layout

| Path | Purpose |
|------|---------|
| `src/ask_my_docs/ingest/` | Parsing, heading-aware chunking |
| `src/ask_my_docs/index/` | Chunk store, BM25, embeddings |
| `src/ask_my_docs/retrieval/` | Hybrid search, RRF, reranker |
| `src/ask_my_docs/generation/` | Prompts, citation validation, LLM |
| `src/ask_my_docs/eval/` | Golden-set metrics + CI thresholds |
| `data/sample_docs/` | Example knowledge base |
| `data/golden_set.jsonl` | Eval cases |
| `.github/workflows/rag-eval.yml` | CI eval gate |

## Evaluation metrics

CI fails if any threshold is missed:

| Metric | Default threshold |
|--------|-------------------|
| Recall@k | ≥ 0.80 |
| Faithfulness | ≥ 0.85 |
| Citation accuracy | ≥ 0.90 |

Configure via env: `EVAL_RECALL_AT_K`, `EVAL_FAITHFULNESS_MIN`, `EVAL_CITATION_ACCURACY_MIN`.

## Add your documents

1. Place `.md` or `.txt` files in a directory.
2. Run `ask-docs-ingest --docs-dir /path/to/docs`.
3. Add cases to `data/golden_set.jsonl` with `expected_sources` filenames.
4. Run `ask-docs-eval` before merging.

## License

MIT
