from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 512
    chunk_overlap: int = 64

    hybrid_top_k_per_channel: int = 50
    hybrid_fused_top_k: int = 20
    rerank_top_k: int = 5

    rrf_k: int = 60

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    mock_llm: bool = False

    eval_recall_at_k: float = 0.80
    eval_faithfulness_min: float = 0.85
    eval_citation_accuracy_min: float = 0.90


settings = Settings()
