"""Streamlit dashboard for RAGuard.

Run with:
    streamlit run raguard/dashboard.py

Provides:
  - a scorecard view of the current eval run
  - a "run regression check" button that applies the same direction-aware
    comparison logic as `raguard check`
  - a flagged-case drill-down table (hallucination / PII fabrication)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from raguard.harness import EvalHarness, aggregate
from raguard.regression import compare_to_baseline, load_baseline
from raguard.cli import load_cases

DEFAULT_CASES_PATH = "sample_data/eval_cases.json"
DEFAULT_BASELINE_PATH = "sample_data/baseline.json"


@st.cache_data(show_spinner=False)
def _run_eval(cases_path: str):
    cases = load_cases(cases_path)
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = aggregate(results)
    return cases, results, agg


def _results_dataframe(results, cases) -> pd.DataFrame:
    case_lookup = {c.id: c for c in cases}
    rows = []
    for r in results:
        case = case_lookup.get(r.case_id)
        rows.append(
            {
                "case_id": r.case_id,
                "question": getattr(case, "question", ""),
                "groundedness": round(r.groundedness, 3),
                "relevance": round(r.relevance, 3),
                "retrieval_precision": round(r.retrieval_precision, 3),
                "retrieval_recall": round(r.retrieval_recall, 3),
                "citation_accuracy": round(r.citation_accuracy, 3),
                "faithfulness_score": round(r.faithfulness_score, 2),
                "hallucination_flag": r.hallucination_flag,
                "has_fabricated_pii": r.has_fabricated_pii,
                "fabricated_pii": ", ".join(r.fabricated_pii),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="RAGuard Dashboard", layout="wide")
    st.title("RAGuard Evaluation Dashboard")
    st.caption("Reference-free evaluation harness for RAG/LLM pipelines, with CI regression gating.")

    with st.sidebar:
        st.header("Configuration")
        cases_path = st.text_input("Eval cases JSON", value=DEFAULT_CASES_PATH)
        baseline_path = st.text_input("Baseline JSON", value=DEFAULT_BASELINE_PATH)
        tolerance = st.slider("Regression tolerance", 0.0, 0.2, 0.05, 0.01)

    if not Path(cases_path).exists():
        st.error(f"Cases file not found: {cases_path}")
        return

    cases, results, agg = _run_eval(cases_path)

    st.subheader("Scorecard")
    cols = st.columns(4)
    scorecard_items = list(agg.items())
    for i, (key, value) in enumerate(scorecard_items):
        with cols[i % 4]:
            display_value = f"{value:.3f}" if isinstance(value, float) else str(value)
            st.metric(label=key, value=display_value)

    st.divider()
    st.subheader("Regression Check")
    if st.button("Run regression check"):
        if not Path(baseline_path).exists():
            st.warning(f"Baseline file not found: {baseline_path}")
        else:
            baseline = load_baseline(baseline_path)
            regressions = compare_to_baseline(agg, baseline, tolerance=tolerance)
            if regressions:
                st.error(f"{len(regressions)} metric(s) regressed:")
                reg_df = pd.DataFrame(
                    [
                        {
                            "metric": r.metric,
                            "baseline": round(r.baseline_value, 3),
                            "current": round(r.current_value, 3),
                            "delta": round(r.delta, 3),
                            "direction": r.direction,
                        }
                        for r in regressions
                    ]
                )
                st.dataframe(reg_df, use_container_width=True)
            else:
                st.success("No regressions detected. All metrics within tolerance.")

    st.divider()
    st.subheader("Flagged Cases")
    df = _results_dataframe(results, cases)
    flagged_df = df[df["hallucination_flag"] | df["has_fabricated_pii"]]
    if flagged_df.empty:
        st.info("No flagged cases.")
    else:
        st.dataframe(flagged_df, use_container_width=True)

    st.divider()
    st.subheader("All Cases")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
