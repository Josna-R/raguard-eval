# RAGuard — LLM Evaluation & Observability Harness

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Tests](https://img.shields.io/badge/tests-5%2F5%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey) ![Offline](https://img.shields.io/badge/runs-offline%2C%20no%20API%20key-success)

A reference-free evaluation harness for RAG/LLM pipelines: scores every
(question, retrieved context, generated answer) triple for **groundedness**
(is the answer actually supported by the retrieved context — the standard
hallucination proxy), **relevance** (does it address the question), and
**retrieval precision/recall** (did the retriever find the right chunks),
then aggregates into a scorecard with a Streamlit dashboard. Runs fully
offline with zero API cost; an optional LLM-as-judge mode is available for
higher-fidelity faithfulness scoring.

**TL;DR:** feed it a batch of (question, context, answer) triples and it
tells you which ones are likely hallucinated — no reference answers, no
API key, no GPU required. See [Get the code](#get-the-code) below.

## What it does

Point it at a JSON file of RAG outputs (question + retrieved context +
generated answer). For each one it computes a **groundedness** score
(does the answer's content actually appear in the retrieved context?), a
**relevance** score (does the answer address the question?), and — if you
supply labeled relevant-chunk IDs — **retrieval precision/recall**. Anything
below a groundedness threshold gets auto-flagged as a likely hallucination.
Results are viewable as a CLI table or an interactive Streamlit dashboard.

## Why this project

Fills a gap that's increasingly what companies hire for once a GenAI system
is in production: not "can you build a RAG pipeline" but "can you tell me
when it's lying." This directly targets the evaluation/observability skill
set (LangSmith, MLflow, W&B) already on the resume, and mirrors the
explainability layer built in the Fraudulent Job Posting Detection project
(surfacing *why* a model produced an output, not just the output itself).

## How it catches hallucinations

The bundled `sample_data/eval_cases.json` includes two deliberately paired
cases on the same question/context: one faithful answer and one that
invents unsupported facts. Running the harness demonstrates the detector
actually works:

```
ID                      Grounded  Relevance  Faithful  Precision  Hallucination?
--------------------------------------------------------------------------------
case-1-grounded            0.899      0.464      4.60      0.500              no
case-2-hallucinated        0.115      0.198      1.46      0.500             YES
case-3-grounded            0.504      0.282      3.02      1.000              no
case-4-off-topic           0.000      0.000      1.00      0.500             YES
```

## Get the code

```bash
git clone https://github.com/Josna-R/raguard-eval.git
cd raguard-eval
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the test suite (fully offline) -- includes a test asserting the
# planted hallucination fixture is actually flagged, and the faithful
# answer is not.
pytest tests/ -v

# Run the CLI scorecard
python -m raguard.cli sample_data/eval_cases.json

# Or the interactive dashboard
streamlit run raguard/dashboard.py
```

The CLI prints the table shown above straight away using the bundled
`sample_data/eval_cases.json`. The dashboard opens at `http://localhost:8501`
with the same data, plus aggregate KPI cards and a list of flagged cases.

## Project structure

```
raguard-eval/
├── raguard/
│   ├── metrics.py     # groundedness / relevance / retrieval precision & recall
│   ├── judge.py         # Judge interface + HeuristicJudge / OpenAIJudge
│   ├── harness.py        # EvalHarness — runs metrics + judge over a batch, aggregates
│   ├── cli.py              # command-line scorecard
│   └── dashboard.py        # Streamlit dashboard
├── sample_data/eval_cases.json  # paired grounded/hallucinated fixtures
├── tests/test_harness.py         # hallucination-detection + metrics tests
└── requirements.txt
```

## Using it on your own RAG pipeline's outputs

Log each (question, retrieved_contexts, answer) triple from your pipeline
(e.g. from the DocuMind RAG project in this portfolio) as an `EvalCase`,
batch them into a JSON file matching `sample_data/eval_cases.json`'s shape,
and point the CLI/dashboard at it. For labeled retrieval-quality metrics,
add `retrieved_ids` / `relevant_ids` per case.

## Upgrading to an LLM-as-judge

```bash
pip install openai
export OPENAI_API_KEY=sk-...
export RAGUARD_JUDGE=openai
python -m raguard.cli sample_data/eval_cases.json
```

## Tech stack

Python · scikit-learn (TF-IDF/cosine similarity) · pandas · Streamlit ·
pluggable OpenAI LLM-as-judge

## Possible extensions

- Swap TF-IDF similarity for sentence-transformer embeddings for
  semantic (not just lexical) groundedness scoring.
- Add a CI gate that fails a PR if `hallucination_flag_rate` regresses.
- Track scores over time in MLflow to catch drift after a prompt or model change.
- Add per-category breakdowns (e.g. by document source or question type).

## Author

Built by [Josna Deepa Rayana](https://github.com/Josna-R), AI/ML Engineer &
Data Scientist. Part of a three-project GenAI portfolio alongside
[DocuMind RAG](https://github.com/Josna-R/documind-rag) (retrieval-augmented
generation) and [AgentCrew](https://github.com/Josna-R/agentcrew-multiagent)
(multi-agent orchestration).
