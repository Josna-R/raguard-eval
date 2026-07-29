import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raguard.cli import load_cases
from raguard.harness import EvalHarness
from raguard.metrics import groundedness, relevance, retrieval_precision_at_k, retrieval_recall

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "eval_cases.json")


def test_grounded_answer_scores_high():
    cases = load_cases(DATA_PATH)
    case = next(c for c in cases if c.id == "case-1-grounded")
    g = groundedness(case.answer, case.contexts)
    assert g > 0.5, f"expected high groundedness for a faithful answer, got {g}"


def test_hallucinated_answer_scores_low_and_is_flagged():
    cases = load_cases(DATA_PATH)
    case = next(c for c in cases if c.id == "case-2-hallucinated")
    harness = EvalHarness()
    result = harness.evaluate_case(case)
    assert result.flagged_hallucination is True
    assert result.groundedness < 0.15


def test_grounded_case_is_not_flagged():
    cases = load_cases(DATA_PATH)
    case = next(c for c in cases if c.id == "case-1-grounded")
    harness = EvalHarness()
    result = harness.evaluate_case(case)
    assert result.flagged_hallucination is False


def test_retrieval_precision_and_recall():
    assert retrieval_precision_at_k(["a", "b", "c"], ["a", "c"]) == 2 / 3
    assert retrieval_recall(["a", "b"], ["a", "c"]) == 0.5
    assert retrieval_precision_at_k([], ["a"]) == 0.0
    assert retrieval_recall(["a"], []) == 1.0


def test_aggregate_reports_hallucination_rate():
    cases = load_cases(DATA_PATH)
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = EvalHarness.aggregate(results)
    assert agg["n_cases"] == 4
    assert 0.0 <= agg["hallucination_flag_rate"] <= 1.0
    # We know case-2 is a deliberate hallucination fixture, so the rate must be > 0
    assert agg["hallucination_flag_rate"] > 0
