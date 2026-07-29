"""Run: python -m raguard.cli sample_data/eval_cases.json"""
from __future__ import annotations

import json
import sys

from .harness import EvalCase, EvalHarness


def load_cases(path: str) -> list[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalCase(**item) for item in raw]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/eval_cases.json"
    cases = load_cases(path)
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)

    header = f"{'ID':<22} {'Grounded':>9} {'Relevance':>10} {'Faithful':>9} {'Precision':>10} {'Hallucination?':>15}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.id:<22} {r.groundedness:>9.3f} {r.relevance:>10.3f} {r.faithfulness_score:>9.2f} "
            f"{'' if r.retrieval_precision is None else format(r.retrieval_precision, '.3f'):>10} "
            f"{'YES' if r.flagged_hallucination else 'no':>15}"
        )

    print()
    print("Aggregate:", json.dumps(EvalHarness.aggregate(results), indent=2))


if __name__ == "__main__":
    main()
