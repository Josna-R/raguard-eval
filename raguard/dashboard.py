"""Streamlit dashboard for RAGuard eval results. Run with:
streamlit run raguard/dashboard.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from raguard.cli import load_cases
from raguard.harness import EvalHarness

st.set_page_config(page_title="RAGuard — LLM Eval Dashboard", page_icon="🛡️")
st.title("🛡️ RAGuard — RAG Evaluation & Observability")
st.caption("Reference-free groundedness, relevance, and hallucination detection for RAG pipelines.")

data_path = st.text_input("Eval cases JSON", value="sample_data/eval_cases.json")

if st.button("Run evaluation") or True:
    try:
        cases = load_cases(data_path)
    except FileNotFoundError:
        st.error(f"Couldn't find {data_path}")
        st.stop()

    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = EvalHarness.aggregate(results)

    st.subheader("Aggregate scorecard")
    cols = st.columns(4)
    cols[0].metric("Avg. groundedness", agg["avg_groundedness"])
    cols[1].metric("Avg. relevance", agg["avg_relevance"])
    cols[2].metric("Avg. faithfulness (1-5)", agg["avg_faithfulness_score"])
    cols[3].metric("Hallucination flag rate", f"{agg['hallucination_flag_rate']*100:.0f}%")

    st.subheader("Per-case results")
    df = pd.DataFrame([r.__dict__ for r in results])
    st.dataframe(df, use_container_width=True)

    st.subheader("Flagged cases (likely hallucinations)")
    flagged = df[df["flagged_hallucination"]]
    if flagged.empty:
        st.success("No hallucinations flagged in this batch.")
    else:
        for _, row in flagged.iterrows():
            st.warning(f"**{row['id']}** — {row['question']}\n\n{row['faithfulness_rationale']}")
