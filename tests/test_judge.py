"""Tests for raguard.judge -- the HeuristicJudge must work fully offline."""

import pytest

from raguard.judge import HeuristicJudge, OpenAIJudge, get_default_judge


def test_heuristic_judge_requires_no_api_key_and_scales_with_groundedness(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = HeuristicJudge()

    high_score, rationale = judge.score_faithfulness(
        "The warranty lasts 2 years from purchase.",
        ["Our warranty lasts 2 years from the date of purchase."],
    )
    low_score, _ = judge.score_faithfulness(
        "Our mascot is a dinosaur.",
        ["Our warranty lasts 2 years from the date of purchase."],
    )

    assert 1.0 <= high_score <= 5.0
    assert isinstance(rationale, str) and rationale
    assert high_score > low_score


def test_default_judge_selection_respects_raguard_judge_env_var(monkeypatch):
    monkeypatch.delenv("RAGUARD_JUDGE", raising=False)
    assert isinstance(get_default_judge(), HeuristicJudge)

    monkeypatch.setenv("RAGUARD_JUDGE", "openai")
    assert isinstance(get_default_judge(), OpenAIJudge)


def test_openai_judge_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = OpenAIJudge(api_key=None)
    with pytest.raises(RuntimeError):
        judge.score_faithfulness("answer", ["context"])
