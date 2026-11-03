"""Tests for raguard.cli -- run / baseline / check subcommands."""

import json

import pytest

from raguard.cli import main

SAMPLE_CASES = "sample_data/eval_cases.json"


def _write_cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    with open(path, "w") as f:
        json.dump(cases, f)
    return str(path)


SIMPLE_CASES = [
    {
        "id": "case-1",
        "question": "What is the warranty period?",
        "answer": "The warranty period is 2 years.",
        "contexts": ["Our warranty lasts 2 years from the date of purchase."],
        "retrieved_ids": ["d1"],
        "relevant_ids": ["d1"],
    },
    {
        "id": "case-2",
        "question": "What is the refund policy?",
        "answer": "Refunds are issued within 30 days of purchase, no questions asked.",
        "contexts": ["Refunds are issued within 30 days of purchase."],
        "retrieved_ids": ["d2"],
        "relevant_ids": ["d2"],
    },
]


def test_cli_run_prints_scorecard_and_executes_on_real_sample_data(tmp_path, capsys):
    cases_path = _write_cases(tmp_path, SIMPLE_CASES)
    exit_code = main(["run", cases_path])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "RAGuard Scorecard" in captured.out
    assert "avg_groundedness" in captured.out

    # Also confirm it runs cleanly end-to-end on the real committed fixtures.
    assert main(["run", SAMPLE_CASES]) == 0


def test_cli_run_writes_html_and_json(tmp_path):
    cases_path = _write_cases(tmp_path, SIMPLE_CASES)
    html_path = tmp_path / "out.html"
    json_path = tmp_path / "out.json"

    exit_code = main(["run", cases_path, "--html", str(html_path), "--json", str(json_path)])

    assert exit_code == 0
    assert html_path.exists()
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert data["aggregate"]["n_cases"] == 2
    assert len(data["results"]) == 2


def test_cli_baseline_saves_aggregate(tmp_path):
    cases_path = _write_cases(tmp_path, SIMPLE_CASES)
    baseline_path = tmp_path / "baseline.json"

    exit_code = main(["baseline", cases_path, "--save", str(baseline_path)])

    assert exit_code == 0
    assert baseline_path.exists()
    with open(baseline_path) as f:
        baseline = json.load(f)
    assert baseline["n_cases"] == 2
    assert "avg_groundedness" in baseline


def test_cli_check_passes_with_no_regression_message(tmp_path, capsys):
    cases_path = _write_cases(tmp_path, SIMPLE_CASES)
    baseline_path = tmp_path / "baseline.json"
    main(["baseline", cases_path, "--save", str(baseline_path)])

    exit_code = main(["check", cases_path, "--baseline", str(baseline_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No regressions detected" in captured.out


def test_cli_check_exits_1_on_inflated_fake_baseline(tmp_path, capsys):
    """A deliberately-inflated fake baseline (numbers no real run could
    match) must make `check` detect a regression and exit 1 -- this is
    the actual CI gate behavior."""
    cases_path = _write_cases(tmp_path, SIMPLE_CASES)
    baseline_path = tmp_path / "fake_baseline.json"

    fake_baseline = {
        "n_cases": 2,
        "avg_groundedness": 0.99,
        "avg_relevance": 0.99,
        "avg_faithfulness_score": 5.0,
        "avg_retrieval_precision": 1.0,
        "avg_retrieval_recall": 1.0,
        "avg_citation_accuracy": 1.0,
        "hallucination_flag_rate": 0.0,
        "pii_fabrication_rate": 0.0,
    }
    with open(baseline_path, "w") as f:
        json.dump(fake_baseline, f)

    exit_code = main(["check", cases_path, "--baseline", str(baseline_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "REGRESSION DETECTED" in captured.out


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])
