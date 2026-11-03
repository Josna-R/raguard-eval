"""Tests for raguard.regression -- direction-aware baseline comparison."""

from raguard.regression import (
    DEFAULT_TOLERANCE,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)


def test_save_and_load_baseline_roundtrip_and_creates_parent_dirs(tmp_path):
    agg = {"n_cases": 5, "avg_groundedness": 0.5, "hallucination_flag_rate": 0.2}
    path = tmp_path / "nested" / "dir" / "baseline.json"

    save_baseline(path, agg)

    assert path.exists()
    assert load_baseline(path) == agg


def test_higher_is_better_regression_and_improvement():
    baseline = {"avg_groundedness": 0.6}
    # Dropped well beyond tolerance -> a regression.
    regressed = compare_to_baseline({"avg_groundedness": 0.4}, baseline)
    assert len(regressed) == 1
    assert regressed[0].metric == "avg_groundedness"
    assert regressed[0].direction == "higher_is_better"

    # Improved -> never a regression.
    improved = compare_to_baseline({"avg_groundedness": 0.8}, baseline)
    assert improved == []


def test_lower_is_better_regression_and_improvement():
    baseline = {"hallucination_flag_rate": 0.1}

    # hallucination_flag_rate going UP is a regression, even though the
    # raw delta (current - baseline) is positive -- direction-awareness
    # is the whole point here.
    regressed = compare_to_baseline({"hallucination_flag_rate": 0.3}, baseline)
    assert len(regressed) == 1
    assert regressed[0].direction == "lower_is_better"
    assert regressed[0].delta > 0  # positive delta, but still a regression

    # hallucination_flag_rate going DOWN is an improvement and must never
    # be flagged, even though a naive "did it move" check might be
    # tempted to look only at the sign of the delta.
    improved = compare_to_baseline({"hallucination_flag_rate": 0.02}, baseline)
    assert improved == []


def test_tolerance_boundary_at_and_just_past_tolerance():
    baseline = {"avg_groundedness": 0.5}

    at_boundary = compare_to_baseline(
        {"avg_groundedness": 0.5 - DEFAULT_TOLERANCE}, baseline, tolerance=DEFAULT_TOLERANCE
    )
    assert at_boundary == []

    past_boundary = compare_to_baseline(
        {"avg_groundedness": 0.5 - DEFAULT_TOLERANCE - 0.001},
        baseline,
        tolerance=DEFAULT_TOLERANCE,
    )
    assert len(past_boundary) == 1


def test_n_cases_ignored_and_multiple_regressions_all_reported():
    baseline = {
        "n_cases": 10,
        "avg_groundedness": 0.6,
        "avg_citation_accuracy": 0.8,
        "hallucination_flag_rate": 0.1,
    }
    current = {
        "n_cases": 3,  # metadata, not a quality metric -- must be ignored
        "avg_groundedness": 0.3,
        "avg_citation_accuracy": 0.8,  # unchanged
        "hallucination_flag_rate": 0.5,
    }
    regressions = compare_to_baseline(current, baseline)
    metrics_flagged = {r.metric for r in regressions}
    assert metrics_flagged == {"avg_groundedness", "hallucination_flag_rate"}
