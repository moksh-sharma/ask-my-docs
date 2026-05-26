import argparse
import json
import sys
from pathlib import Path

from ask_my_docs.config import settings
from ask_my_docs.eval.runner import run_eval
from ask_my_docs.pipeline import ask, ingest


def ingest_main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=settings.data_dir / "sample_docs",
        help="Directory containing .md/.txt files",
    )
    args = parser.parse_args()
    doc_count, chunk_count = ingest(str(args.docs_dir))
    print(f"Ingested {doc_count} documents, {chunk_count} chunks → {settings.index_dir}")


def ask_main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against indexed docs")
    parser.add_argument("question", nargs="+", help="Question text")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    question = " ".join(args.question)
    response = ask(question, include_debug=args.debug)
    print(json.dumps(response.model_dump(), indent=2))


def serve_main() -> None:
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=".",
    )


def eval_main() -> None:
    parser = argparse.ArgumentParser(description="Run golden-set RAG evaluation")
    parser.add_argument(
        "--golden",
        type=Path,
        default=settings.data_dir / "golden_set.jsonl",
    )
    parser.add_argument("--docs-dir", type=Path, default=settings.data_dir / "sample_docs")
    parser.add_argument("--fail-on-threshold", action="store_true", default=True)
    args = parser.parse_args()

    report = run_eval(args.golden, args.docs_dir)
    print(json.dumps(report.model_dump(), indent=2))
    if args.fail_on_threshold and not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    eval_main()
