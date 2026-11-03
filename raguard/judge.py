"""LLM-as-judge faithfulness scoring, with a pluggable backend.

RAGuard runs fully offline by default: `HeuristicJudge` derives a 1-5
faithfulness score from the lexical `groundedness` metric, so no API key
is ever required to run the eval suite or the CI gate. `OpenAIJudge` is
an opt-in upgrade for teams that want a real model-as-judge rationale --
it is selected via the `RAGUARD_JUDGE=openai` environment variable and is
never required for tests or CI.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from raguard.metrics import groundedness


class Judge(ABC):
    """Abstract base class for a faithfulness judge."""

    @abstractmethod
    def score_faithfulness(
        self, answer: str, contexts: Sequence[str]
    ) -> Tuple[float, str]:
        """Return (score, rationale). Score is on a 1-5 scale."""
        raise NotImplementedError


class HeuristicJudge(Judge):
    """Default, fully-offline judge.

    Maps the TF-IDF `groundedness` score onto a 1-5 faithfulness scale.
    Deterministic, free, and requires no network access or API key --
    this is what makes RAGuard runnable in CI with zero configuration.
    """

    # groundedness in [0, 1] -> faithfulness in [1, 5]
    MIN_SCORE = 1.0
    MAX_SCORE = 5.0

    def score_faithfulness(
        self, answer: str, contexts: Sequence[str]
    ) -> Tuple[float, str]:
        g = groundedness(answer, contexts)
        score = self.MIN_SCORE + g * (self.MAX_SCORE - self.MIN_SCORE)
        score = round(score, 2)

        if g >= 0.6:
            rationale = (
                f"High lexical overlap with retrieved context (groundedness="
                f"{g:.2f}); answer appears well-supported."
            )
        elif g >= 0.3:
            rationale = (
                f"Moderate lexical overlap with retrieved context (groundedness="
                f"{g:.2f}); answer is partially supported."
            )
        else:
            rationale = (
                f"Low lexical overlap with retrieved context (groundedness="
                f"{g:.2f}); answer may be unsupported or hallucinated."
            )
        return score, rationale


class OpenAIJudge(Judge):
    """Opt-in LLM-as-judge backed by a real OpenAI API call.

    Only instantiated/used when RAGUARD_JUDGE=openai. Requires the
    `openai` package and an `OPENAI_API_KEY` in the environment at call
    time -- never at import time, so the rest of RAGuard stays importable
    without the dependency installed.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    PROMPT_TEMPLATE = (
        "You are a strict evaluator of RAG system outputs. Rate how "
        "faithful the ANSWER is to the CONTEXT on a 1-5 scale, where 5 "
        "means every claim in the answer is directly supported by the "
        "context, and 1 means the answer is largely unsupported or "
        "contradicts the context.\n\n"
        "CONTEXT:\n{context}\n\n"
        "ANSWER:\n{answer}\n\n"
        "Respond in exactly this format:\n"
        "SCORE: <integer 1-5>\n"
        "RATIONALE: <one sentence>"
    )

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("RAGUARD_OPENAI_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def score_faithfulness(
        self, answer: str, contexts: Sequence[str]
    ) -> Tuple[float, str]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenAIJudge requires the 'openai' package. Install it with "
                "`pip install openai` to use RAGUARD_JUDGE=openai."
            ) from exc

        if not self.api_key:
            raise RuntimeError(
                "OpenAIJudge requires OPENAI_API_KEY to be set in the "
                "environment."
            )

        client = OpenAI(api_key=self.api_key)
        prompt = self.PROMPT_TEMPLATE.format(
            context=" ".join(contexts), answer=answer
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return self._parse_response(content)

    @staticmethod
    def _parse_response(content: str) -> Tuple[float, str]:
        score = 3.0
        rationale = content.strip()
        for line in content.splitlines():
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()
        return score, rationale


def get_default_judge() -> Judge:
    """Return the judge selected by the RAGUARD_JUDGE env var.

    Defaults to HeuristicJudge (offline, no API key) unless
    RAGUARD_JUDGE=openai is explicitly set.
    """
    if os.environ.get("RAGUARD_JUDGE", "").lower() == "openai":
        return OpenAIJudge()
    return HeuristicJudge()
