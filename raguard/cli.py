"""RAGuard command-line interface.

Subcommands:
  run      <cases.json> [--html out.html] [--json out.json]
  baseline <cases.json> --save <path>
  check    <cases.json> --baseline <path>   (the CI regression gate; exits 1 on regression)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from raguard.harness import EvalCase, EvalHarness, aggregate, extract_citation_ids
from raguard.regression import compare_to_baseline, load_baseline, save_baseline
from raguard.report import save_html_report


def load_cases(path: str) -> List[EvalCase]:
    """Load EvalCases from a JSON file (a list of case dicts).

    If a case dict has no explicit "cited_ids", they are derived from
    [n] citation markers in the answer text via extract_citation_ids.
    """
    with open(path, "r") as f:
        raw_cases = json.load(f)

    cases: List[EvalCase] = []
    for raw in raw_cases:
        retrieved_ids = raw.get("retrieved_ids", [])
        cited_ids = raw.get("cited_ids")
        if cited_ids is None:
            cited_ids = extract_citation_ids(raw.get("answer", ""), retrieved_ids)

        cases.append(
            EvalCase(
                id=raw["id"],
                question=raw["question"],
                answer=raw["answer"],
                contexts=raw.get("contexts", []),
                retrieved_ids=retrieved_ids,
                relevant_ids=raw.get("relevant_ids"),
                cited_ids=cited_ids,
            )
        )
    return cases


def _print_scorecard(agg: dict) -> None:
    print("\nRAGuard Scorecard")
    print("=" * 40)
    for key, value in agg.items():
        if isinstance(value, float):
            print(f"  {key:<28} {value:.3f}")
        else:
            print(f"  {key:<28} {value}")
    print("=" * 40)


def _results_to_json(results, agg) -> dict:
    return {
        "aggregate": agg,
        "results": [asdict(r) for r in results],
    }


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = aggregate(results)
    _print_scorecard(agg)

    if args.html:
        save_html_report(args.html, results, agg, cases=cases)
        print(f"\nHTML report written to {args.html}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(_results_to_json(results, agg), f, indent=2)
        print(f"JSON results written to {args.json}")

    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = aggregate(results)
    _print_scorecard(agg)

    save_baseline(args.save, agg)
    print(f"\nBaseline saved to {args.save}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = aggregate(results)
    _print_scorecard(agg)

    baseline = load_baseline(args.baseline)
    regressions = compare_to_baseline(agg, baseline, tolerance=args.tolerance)

    if regressions:
        print(f"\nREGRESSION DETECTED: {len(regressions)} metric(s) moved the wrong way")
        print("-" * 40)
        for reg in regressions:
            print(
                f"  {reg.metric:<28} baseline={reg.baseline_value:.3f} "
                f"current={reg.current_value:.3f} delta={reg.delta:+.3f} "
                f"({reg.direction})"
            )
        print("-" * 40)
        return 1

    print("\nNo regressions detected. All metrics within tolerance.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raguard", description="Reference-free evaluation harness for RAG/LLM pipelines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the eval harness against a cases file.")
    run_parser.add_argument("cases", help="Path to a cases JSON file.")
    run_parser.add_argument("--html", help="Write an HTML report to this path.", default=None)
    run_parser.add_argument("--json", help="Write JSON results to this path.", default=None)
    run_parser.set_defaults(func=cmd_run)

    baseline_parser = subparsers.add_parser(
        "baseline", help="Run the eval harness and save the aggregate as a baseline."
    )
    baseline_parser.add_argument("cases", help="Path to a cases JSON file.")
    baseline_parser.add_argument("--save", required=True, help="Path to save the baseline JSON.")
    baseline_parser.set_defaults(func=cmd_baseline)

    check_parser = subparsers.add_parser(
        "check", help="Run the eval harness and compare to a baseline (CI gate)."
    )
    check_parser.add_argument("cases", help="Path to a cases JSON file.")
    check_parser.add_argument("--baseline", required=True, help="Path to the baseline JSON.")
    check_parser.add_argument(
        "--tolerance", type=float, default=0.05, help="Allowed drift before flagging a regression."
    )
    check_parser.set_defaults(func=cmd_check)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
