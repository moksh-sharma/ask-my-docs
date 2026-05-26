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

## Quick start (web UI)

```bash
cd ask-my-docs
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # MOCK_LLM=true works without Ollama
ask-docs-serve
```

Open **http://127.0.0.1:8000** in your browser. The UI will ingest `data/sample_docs` on first load; ask questions in the chat panel.

If you moved the project folder, delete and recreate `.venv` so CLI entrypoints point at the correct Python path.

## CLI (optional)

```bash
ask-docs-ingest --docs-dir data/sample_docs
ask-docs-ask "When are invoices sent?"
ask-docs-eval
```

## API

Endpoints (also used by the web UI):

- `GET /` — web frontend
- `GET /health`
- `POST /ingest?docs_dir=data/sample_docs`
- `POST /ask` — body: `{"question": "...", "include_debug": false}`

### Ollama generation

```bash
ollama pull llama3.2
# In .env set MOCK_LLM=false
ask-docs-serve
```

Configure via env: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.2`).

## Project layout

| Path | Purpose |
|------|---------|
| `apps/web/` | Browser UI (HTML/CSS/JS) |
| `apps/api/` | FastAPI server |
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
2. Click **Rebuild index** in the UI (or run `ask-docs-ingest --docs-dir /path/to/docs`).
3. Add cases to `data/golden_set.jsonl` with `expected_sources` filenames.
4. Run `ask-docs-eval` before merging.

## License

MIT
