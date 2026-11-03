"""Tests for raguard.harness -- per-case scoring and aggregation."""

from raguard.harness import (
    GROUNDEDNESS_FLAG_THRESHOLD,
    EvalCase,
    EvalHarness,
    aggregate,
    extract_citation_ids,
)


def _make_case(**kwargs):
    defaults = dict(
        id="case-x",
        question="What is the warranty period?",
        answer="The warranty period is 2 years.",
        contexts=["Our warranty lasts 2 years from the date of purchase."],
        retrieved_ids=["d1"],
        relevant_ids=["d1"],
        cited_ids=None,
    )
    defaults.update(kwargs)
    return EvalCase(**defaults)


def test_evaluate_case_well_grounded_is_not_flagged():
    harness = EvalHarness()
    result = harness.evaluate_case(_make_case())
    assert result.case_id == "case-x"
    assert result.groundedness > GROUNDEDNESS_FLAG_THRESHOLD
    assert result.hallucination_flag is False
    assert result.has_fabricated_pii is False
    assert result.fabricated_pii == []


def test_evaluate_case_hallucination_flag_triggers_below_threshold():
    harness = EvalHarness()
    case = _make_case(
        answer="Our office has a rooftop garden and a ping pong table.",
        contexts=["Our warranty lasts 2 years from the date of purchase."],
    )
    result = harness.evaluate_case(case)
    assert result.groundedness < GROUNDEDNESS_FLAG_THRESHOLD
    assert result.hallucination_flag is True


def test_evaluate_case_detects_fabricated_pii():
    harness = EvalHarness()
    case = _make_case(
        answer="Call us at (415) 555-0199 for help with your warranty.",
        contexts=["Our warranty lasts 2 years from the date of purchase."],
    )
    result = harness.evaluate_case(case)
    assert result.has_fabricated_pii is True
    assert len(result.fabricated_pii) == 1


def test_evaluate_batch_returns_one_result_per_case():
    harness = EvalHarness()
    cases = [_make_case(id=f"case-{i}") for i in range(3)]
    results = harness.evaluate_batch(cases)
    assert [r.case_id for r in results] == ["case-0", "case-1", "case-2"]


def test_aggregate_computes_averages_rates_and_handles_empty():
    harness = EvalHarness()
    good_case = _make_case(id="good")
    bad_case = _make_case(
        id="bad", answer="Our office has a rooftop garden and a ping pong table."
    )
    agg = aggregate(harness.evaluate_batch([good_case, bad_case]))

    assert agg["n_cases"] == 2
    assert agg["hallucination_flag_rate"] == 0.5
    assert agg["pii_fabrication_rate"] == 0.0
    assert 0.0 <= agg["avg_groundedness"] <= 1.0
    assert 0.0 <= agg["avg_faithfulness_score"] <= 5.0

    empty_agg = aggregate([])
    assert empty_agg["n_cases"] == 0
    assert empty_agg["avg_groundedness"] == 0.0
    assert empty_agg["hallucination_flag_rate"] == 0.0


def test_extract_citation_ids_maps_markers_and_flags_phantom_refs():
    assert extract_citation_ids(
        "Supported by sources [1] and [2].", ["doc_a", "doc_b"]
    ) == ["doc_a", "doc_b"]

    cited = extract_citation_ids("Supported by source [5].", ["doc_a", "doc_b"])
    assert cited == ["phantom_ref_5"]
    assert cited[0] not in ["doc_a", "doc_b"]
