import numpy as np
from sentence_transformers import SentenceTransformer

from ask_my_docs.config import settings
from ask_my_docs.index.store import DocumentIndex
from ask_my_docs.models import Chunk

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def build_embeddings_for_index(index: DocumentIndex) -> np.ndarray:
    texts = [c.text for c in index.chunks]
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    matrix = embed_texts(texts)
    index.set_embeddings(matrix)
    return matrix
