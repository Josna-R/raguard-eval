"""The evaluation harness: wires metrics + judge together per-case and aggregates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from raguard.judge import HeuristicJudge, Judge
from raguard.metrics import (
    citation_accuracy,
    groundedness,
    pii_fabrication_check,
    relevance,
    retrieval_precision_at_k,
    retrieval_recall,
)

# An answer whose groundedness falls below this threshold is flagged as a
# likely hallucination.
GROUNDEDNESS_FLAG_THRESHOLD = 0.15

# Default k for retrieval_precision_at_k when evaluating a case.
DEFAULT_PRECISION_K = 3

# Matches inline citation markers like "[1]", "[2]" as a DocuMind-style RAG
# answerer would emit them (1-indexed position into the retrieved chunk list).
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_citation_ids(answer: str, retrieved_ids: Sequence[str]) -> List[str]:
    """Derive cited chunk ids from [n] markers in `answer`.

    Marker [n] is treated as a 1-indexed pointer into `retrieved_ids`. If
    the marker's index falls outside the range of what was actually
    retrieved, it is mapped to a synthetic "phantom_ref_n" id that will
    never match a real retrieved id -- this is how a model citing a
    source number that doesn't correspond to anything actually retrieved
    (a "phantom citation") shows up for citation_accuracy to catch.
    """
    cited: List[str] = []
    for marker in _CITATION_MARKER_RE.findall(answer or ""):
        idx = int(marker) - 1
        if 0 <= idx < len(retrieved_ids):
            cited.append(retrieved_ids[idx])
        else:
            cited.append(f"phantom_ref_{marker}")
    return cited


@dataclass
class EvalCase:
    """A single input to the evaluation harness."""

    id: str
    question: str
    answer: str
    contexts: List[str]
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: Optional[List[str]] = None
    cited_ids: Optional[List[str]] = None


@dataclass
class EvalResult:
    """The scored output for a single EvalCase."""

    case_id: str
    groundedness: float
    relevance: float
    retrieval_precision: float
    retrieval_recall: float
    citation_accuracy: float
    faithfulness_score: float
    faithfulness_rationale: str
    hallucination_flag: bool
    has_fabricated_pii: bool
    fabricated_pii: List[str]


class EvalHarness:
    """Runs EvalCases through the metrics + judge pipeline."""

    def __init__(self, judge: Optional[Judge] = None, precision_k: int = DEFAULT_PRECISION_K):
        self.judge = judge or HeuristicJudge()
        self.precision_k = precision_k

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        g = groundedness(case.answer, case.contexts)
        r = relevance(case.answer, case.question)

        relevant_ids = case.relevant_ids if case.relevant_ids is not None else []
        precision = retrieval_precision_at_k(
            case.retrieved_ids, relevant_ids, self.precision_k
        )
        recall = retrieval_recall(case.retrieved_ids, relevant_ids)

        cited_ids = case.cited_ids if case.cited_ids is not None else []
        c_acc = citation_accuracy(cited_ids, case.retrieved_ids)

        faithfulness_score, rationale = self.judge.score_faithfulness(
            case.answer, case.contexts
        )

        fabricated = pii_fabrication_check(case.answer, case.contexts)

        return EvalResult(
            case_id=case.id,
            groundedness=g,
            relevance=r,
            retrieval_precision=precision,
            retrieval_recall=recall,
            citation_accuracy=c_acc,
            faithfulness_score=faithfulness_score,
            faithfulness_rationale=rationale,
            hallucination_flag=g < GROUNDEDNESS_FLAG_THRESHOLD,
            has_fabricated_pii=len(fabricated) > 0,
            fabricated_pii=fabricated,
        )

    def evaluate_batch(self, cases: Sequence[EvalCase]) -> List[EvalResult]:
        return [self.evaluate_case(case) for case in cases]


def aggregate(results: Sequence[EvalResult]) -> Dict[str, float]:
    """Roll a list of EvalResults up into scorecard-level averages/rates."""
    n = len(results)
    if n == 0:
        return {
            "n_cases": 0,
            "avg_groundedness": 0.0,
            "avg_relevance": 0.0,
            "avg_faithfulness_score": 0.0,
            "avg_retrieval_precision": 0.0,
            "avg_retrieval_recall": 0.0,
            "avg_citation_accuracy": 0.0,
            "hallucination_flag_rate": 0.0,
            "pii_fabrication_rate": 0.0,
        }

    return {
        "n_cases": n,
        "avg_groundedness": sum(r.groundedness for r in results) / n,
        "avg_relevance": sum(r.relevance for r in results) / n,
        "avg_faithfulness_score": sum(r.faithfulness_score for r in results) / n,
        "avg_retrieval_precision": sum(r.retrieval_precision for r in results) / n,
        "avg_retrieval_recall": sum(r.retrieval_recall for r in results) / n,
        "avg_citation_accuracy": sum(r.citation_accuracy for r in results) / n,
        "hallucination_flag_rate": sum(1 for r in results if r.hallucination_flag) / n,
        "pii_fabrication_rate": sum(1 for r in results if r.has_fabricated_pii) / n,
    }
