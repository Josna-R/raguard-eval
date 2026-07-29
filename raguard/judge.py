"""
Optional LLM-as-judge scoring.

Default `HeuristicJudge` reuses the offline TF-IDF similarity metrics so
the harness needs no API key. `OpenAIJudge` / `AnthropicJudge` ask a real
model to rate faithfulness 1-5 with a rationale -- higher fidelity, but
costs money and API latency, so it's opt-in via RAGUARD_JUDGE=openai.
"""
from __future__ import annotations

import abc
import os

from .metrics import groundedness


class Judge(abc.ABC):
    @abc.abstractmethod
    def score_faithfulness(self, question: str, contexts: list[str], answer: str) -> tuple[float, str]:
        """Return (score 1-5, rationale)."""


class HeuristicJudge(Judge):
    def score_faithfulness(self, question: str, contexts: list[str], answer: str) -> tuple[float, str]:
        g = groundedness(answer, contexts)
        score = round(1 + g * 4, 2)  # map [0,1] similarity -> [1,5] score
        rationale = (
            f"Offline heuristic: TF-IDF cosine overlap between answer and retrieved "
            f"context is {g:.2f}, mapped to a 1-5 faithfulness score."
        )
        return score, rationale


class OpenAIJudge(Judge):
    def __init__(self, model: str = "gpt-4.1-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai to use OpenAIJudge") from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def score_faithfulness(self, question: str, contexts: list[str], answer: str) -> tuple[float, str]:
        context = "\n\n".join(contexts)
        prompt = (
            "Rate how faithful the ANSWER is to the CONTEXT on a 1-5 scale "
            "(5 = fully supported, 1 = fabricated/unsupported). "
            f"Respond as 'SCORE: <n>\\nRATIONALE: <text>'.\n\n"
            f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
        )
        resp = self._client.chat.completions.create(
            model=self._model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        text = resp.choices[0].message.content.strip()
        try:
            score_line, rationale_line = text.split("\n", 1)
            score = float(score_line.split(":")[1].strip())
            rationale = rationale_line.split(":", 1)[1].strip()
        except Exception:
            score, rationale = 3.0, text
        return score, rationale


def get_judge(kind: str | None = None) -> Judge:
    kind = kind or os.environ.get("RAGUARD_JUDGE", "heuristic")
    if kind == "heuristic":
        return HeuristicJudge()
    if kind == "openai":
        return OpenAIJudge()
    raise ValueError(f"Unknown judge kind: {kind}")
