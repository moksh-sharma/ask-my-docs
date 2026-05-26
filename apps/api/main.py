from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ask_my_docs.config import settings
from ask_my_docs.models import AskRequest, AskResponse, IngestResult
from ask_my_docs.pipeline import ask, ensure_index_built, ingest

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_index_built()
    except RuntimeError:
        pass
    yield


app = FastAPI(
    title="Ask My Docs",
    description="Hybrid RAG API with BM25 + vector search, reranking, and citations",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResult)
def ingest_docs(docs_dir: str | None = None):
    path = Path(docs_dir) if docs_dir else settings.data_dir / "sample_docs"
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Docs directory not found: {path}")
    doc_count, chunk_count = ingest(str(path))
    return IngestResult(documents=doc_count, chunks=chunk_count)


@app.post("/ask", response_model=AskResponse)
def ask_docs(body: AskRequest):
    try:
        return ask(body.question, include_debug=body.include_debug)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
