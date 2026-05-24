from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    doc_id: str
    title: str
    source: str
    section: str | None = None
    page: int | None = None
    url: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class CitationRef(BaseModel):
    id: int = Field(description="1-based citation index into retrieved context")
    quote: str | None = Field(default=None, description="Supporting quote from the source")


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationRef]
    sources: list[dict]
    retrieval_debug: dict | None = None


class AskRequest(BaseModel):
    question: str
    include_debug: bool = False


class IngestResult(BaseModel):
    documents: int
    chunks: int


class EvalCase(BaseModel):
    id: str
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(
        default_factory=list,
        description="Source filenames expected in top-k (resolved to doc_ids at eval time)",
    )
    reference_answer: str | None = None
    must_cite: bool = True


class EvalReport(BaseModel):
    total: int
    recall_at_k: float
    faithfulness: float
    citation_accuracy: float
    passed: bool
    thresholds: dict
    failures: list[dict]
