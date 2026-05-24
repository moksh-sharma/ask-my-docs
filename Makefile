.PHONY: install ingest ask eval test serve

install:
	pip install -e ".[dev]"

ingest:
	MOCK_LLM=true ask-docs-ingest --docs-dir data/sample_docs

ask:
	MOCK_LLM=true ask-docs-ask $(Q)

eval:
	MOCK_LLM=true ask-docs-eval

test:
	MOCK_LLM=true pytest -q

serve:
	MOCK_LLM=true uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
