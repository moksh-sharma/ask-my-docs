import tempfile
from pathlib import Path

import pytest

from ask_my_docs.config import settings
from ask_my_docs.pipeline import ask, ingest


@pytest.fixture
def isolated_index(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp) / "index"
        monkeypatch.setattr(settings, "index_dir", index_dir)
        monkeypatch.setattr(settings, "mock_llm", True)
        yield index_dir


def test_ingest_and_ask(isolated_index):
    docs = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    doc_count, chunk_count = ingest(str(docs))
    assert doc_count == 3
    assert chunk_count > 0

    response = ask("When are invoices sent?")
    assert "[1]" in response.answer or response.citations
    assert response.citations
    assert response.sources
