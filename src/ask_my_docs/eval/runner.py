import json
import tempfile
from pathlib import Path

from ask_my_docs.config import settings
from ask_my_docs.eval.metrics import (
    answer_overlap_with_reference,
    citation_accuracy,
    faithfulness_score,
    recall_at_k,
)
from ask_my_docs.index.store import DocumentIndex
from ask_my_docs.models import EvalCase, EvalReport
from ask_my_docs.pipeline import ask, ingest
from ask_my_docs.retrieval.hybrid import hybrid_retrieve


def load_golden(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(EvalCase.model_validate(json.loads(line)))
    return cases


def run_eval(golden_path: Path, docs_dir: Path) -> EvalReport:
    settings.mock_llm = True

    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp) / "index"
        settings.index_dir = index_dir

        ingest(str(docs_dir))
        index = DocumentIndex.load(index_dir)

        cases = load_golden(golden_path)
        if not cases:
            raise ValueError(f"No eval cases in {golden_path}")

        recall_scores: list[float] = []
        faith_scores: list[float] = []
        cite_scores: list[float] = []
        failures: list[dict] = []

        k = settings.rerank_top_k

        source_to_doc_id = {c.metadata.source: c.metadata.doc_id for c in index.chunks}

        for case in cases:
            chunks, _ = hybrid_retrieve(index, case.question)
            retrieved_doc_ids = [c.metadata.doc_id for c in chunks]
            expected_ids = list(case.expected_doc_ids)
            for src in case.expected_sources:
                doc_id = source_to_doc_id.get(src)
                if doc_id and doc_id not in expected_ids:
                    expected_ids.append(doc_id)
            r = recall_at_k(retrieved_doc_ids, expected_ids, k)
            recall_scores.append(r)

            try:
                response = ask(case.question, index=index)
            except Exception as exc:
                failures.append({"case_id": case.id, "error": str(exc)})
                faith_scores.append(0.0)
                cite_scores.append(0.0)
                continue

            f = faithfulness_score(response.answer, chunks)
            c = citation_accuracy(response, len(chunks))
            faith_scores.append(f)
            cite_scores.append(c)

            ref_overlap = answer_overlap_with_reference(
                response.answer, case.reference_answer
            )

            passed_case = r >= 1.0 and f >= settings.eval_faithfulness_min and c >= 0.99
            if case.reference_answer:
                passed_case = passed_case and ref_overlap >= 0.3

            if not passed_case:
                failures.append(
                    {
                        "case_id": case.id,
                        "recall": r,
                        "faithfulness": f,
                        "citation_accuracy": c,
                        "ref_overlap": ref_overlap,
                    }
                )

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_faith = sum(faith_scores) / len(faith_scores)
    avg_cite = sum(cite_scores) / len(cite_scores)

    thresholds = {
        "recall_at_k": settings.eval_recall_at_k,
        "faithfulness": settings.eval_faithfulness_min,
        "citation_accuracy": settings.eval_citation_accuracy_min,
    }

    passed = (
        avg_recall >= settings.eval_recall_at_k
        and avg_faith >= settings.eval_faithfulness_min
        and avg_cite >= settings.eval_citation_accuracy_min
    )

    return EvalReport(
        total=len(cases),
        recall_at_k=round(avg_recall, 4),
        faithfulness=round(avg_faith, 4),
        citation_accuracy=round(avg_cite, 4),
        passed=passed,
        thresholds=thresholds,
        failures=failures,
    )
