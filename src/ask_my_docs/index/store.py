import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ask_my_docs.config import settings
from ask_my_docs.models import Chunk


def _tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


class DocumentIndex:
    """Persists chunks, BM25 corpus, and dense embeddings."""

    CHUNKS_FILE = "chunks.json"
    EMBEDDINGS_FILE = "embeddings.npy"
    META_FILE = "meta.json"

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self._chunk_id_to_idx: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "DocumentIndex":
        inst = cls(index_dir)
        inst._load_from_disk()
        return inst

    def _load_from_disk(self) -> None:
        chunks_path = self.index_dir / self.CHUNKS_FILE
        if not chunks_path.exists():
            return

        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.model_validate(c) for c in raw]
        self._rebuild_chunk_map()
        self._rebuild_bm25()

        emb_path = self.index_dir / self.EMBEDDINGS_FILE
        if emb_path.exists() and self.chunks:
            self._embeddings = np.load(emb_path)

    def _rebuild_chunk_map(self) -> None:
        self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def _rebuild_bm25(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        corpus = [_tokenize_for_bm25(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def add_document_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        # Replace existing chunks for this document
        self.chunks = [c for c in self.chunks if c.metadata.doc_id != doc_id]
        self.chunks.extend(chunks)
        self._rebuild_chunk_map()
        self._rebuild_bm25()
        self._embeddings = None  # rebuilt on save

    def set_embeddings(self, matrix: np.ndarray) -> None:
        if matrix.shape[0] != len(self.chunks):
            raise ValueError("Embedding matrix row count must match chunk count")
        self._embeddings = matrix

    @property
    def embeddings(self) -> np.ndarray | None:
        return self._embeddings

    @property
    def bm25(self) -> BM25Okapi | None:
        return self._bm25

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        idx = self._chunk_id_to_idx.get(chunk_id)
        if idx is None:
            return None
        return self.chunks[idx]

    def doc_ids(self) -> set[str]:
        return {c.metadata.doc_id for c in self.chunks}

    def save(self) -> None:
        chunks_path = self.index_dir / self.CHUNKS_FILE
        chunks_path.write_text(
            json.dumps([c.model_dump() for c in self.chunks], indent=2),
            encoding="utf-8",
        )
        meta = {
            "chunk_count": len(self.chunks),
            "embedding_model": settings.embedding_model,
        }
        (self.index_dir / self.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self._embeddings is not None:
            np.save(self.index_dir / self.EMBEDDINGS_FILE, self._embeddings)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0
