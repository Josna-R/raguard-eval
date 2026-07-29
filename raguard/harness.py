"""Evaluation harness: runs metrics + judge over a batch of RAG eval cases
and produces a per-case scorecard plus aggregate stats."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .judge import Judge, get_judge
from .metrics import groundedness, relevance, retrieval_precision_at_k, retrieval_recall

GROUNDEDNESS_FLAG_THRESHOLD = 0.15  # below this, flag as likely hallucination


@dataclass
class EvalCase:
    id: str
    question: str
    answer: str
    contexts: List[str]
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    id: str
    question: str
    groundedness: float
    relevance: float
    retrieval_precision: Optional[float]
    retrieval_recall: Optional[float]
    faithfulness_score: float
    faithfulness_rationale: str
    flagged_hallucination: bool


class EvalHarness:
    def __init__(self, judge: Optional[Judge] = None):
        self.judge = judge or get_judge()

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        g = groundedness(case.answer, case.contexts)
        r = relevance(case.answer, case.question)
        precision = (
            retrieval_precision_at_k(case.retrieved_ids, case.relevant_ids) if case.relevant_ids else None
        )
        recall = retrieval_recall(case.retrieved_ids, case.relevant_ids) if case.relevant_ids else None
        score, rationale = self.judge.score_faithfulness(case.question, case.contexts, case.answer)

        return EvalResult(
            id=case.id,
            question=case.question,
            groundedness=round(g, 4),
            relevance=round(r, 4),
            retrieval_precision=round(precision, 4) if precision is not None else None,
            retrieval_recall=round(recall, 4) if recall is not None else None,
            faithfulness_score=score,
            faithfulness_rationale=rationale,
            flagged_hallucination=g < GROUNDEDNESS_FLAG_THRESHOLD,
        )

    def evaluate_batch(self, cases: List[EvalCase]) -> List[EvalResult]:
        return [self.evaluate_case(c) for c in cases]

    @staticmethod
    def aggregate(results: List[EvalResult]) -> dict:
        if not results:
            return {}
        n = len(results)
        precisions = [r.retrieval_precision for r in results if r.retrieval_precision is not None]
        recalls = [r.retrieval_recall for r in results if r.retrieval_recall is not None]
        return {
            "n_cases": n,
            "avg_groundedness": round(sum(r.groundedness for r in results) / n, 4),
            "avg_relevance": round(sum(r.relevance for r in results) / n, 4),
            "avg_faithfulness_score": round(sum(r.faithfulness_score for r in results) / n, 4),
            "avg_retrieval_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "avg_retrieval_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "hallucination_flag_rate": round(sum(r.flagged_hallucination for r in results) / n, 4),
        }
