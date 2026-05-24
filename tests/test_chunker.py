from ask_my_docs.ingest.chunker import chunk_text


def test_chunker_produces_metadata():
    chunks = chunk_text(
        "# Intro\n\nHello world " * 200,
        doc_id="doc1",
        title="T",
        source="file.md",
        chunk_size=50,
        chunk_overlap=10,
    )
    assert len(chunks) >= 1
    assert chunks[0].metadata.doc_id == "doc1"
    assert chunks[0].metadata.source == "file.md"
