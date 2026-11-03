# RAGuard

[![CI](https://github.com/Josna-R/raguard-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Josna-R/raguard-eval/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Offline by default](https://img.shields.io/badge/runs-offline%20by%20default-brightgreen)

**A reference-free evaluation harness for RAG/LLM pipelines, with CI
regression gating.** Runs fully offline by default -- no API key
required -- with an optional LLM-as-judge upgrade.

RAGuard answers the question a single eval run can't: *did this change
make things worse?* It scores each answer on six reference-free metrics,
rolls them into a scorecard, and compares that scorecard against a saved
baseline so a regression in hallucination rate, citation accuracy, or PII
fabrication can block a pull request automatically.

Built by [Josna Deepa Rayana](https://github.com/Josna-R) as a portfolio
project demonstrating practical RAG evaluation and CI tooling for
AI/ML engineering roles.

## Why RAGuard

Most RAG eval setups require hand-written gold answers for every test
question, which doesn't scale to production traffic. RAGuard's metrics are
**reference-free**: they only use what the pipeline already produces --
the question, the answer, the retrieved context, and the ids the retriever
and generator touched. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the full design rationale, including why `citation_accuracy` and
`pii_fabrication_check` catch failure modes that a pure groundedness score
misses.

## Features

- **Six reference-free-ish metrics** per case: `groundedness`,
  `relevance`, `retrieval_precision_at_k`, `retrieval_recall`,
  `citation_accuracy`, and `pii_fabrication_check` (with `find_pii` as a
  standalone PII scanner).
- **Pluggable LLM-as-judge**: `HeuristicJudge` (default, offline, free) or
  `OpenAIJudge` (opt-in via `RAGUARD_JUDGE=openai`).
- **CI regression gate**: `raguard check` compares a run's scorecard to a
  saved baseline, direction-aware per metric, and exits non-zero on a
  regression.
- **Self-contained HTML reports**: inline CSS, no JS, all untrusted
  question/answer/context text HTML-escaped (XSS-tested).
- **Streamlit dashboard**: scorecard, a "run regression check" button, and
  a flagged-case drill-down table.
- **29+ passing tests** covering every metric in isolation, harness
  scoring/aggregation, regression comparison in both directions, HTML
  escaping, and the full CLI.

## Get the code

```bash
git clone https://github.com/Josna-R/raguard-eval.git
cd raguard-eval
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the test suite
pytest tests/ -v

# 3. Run the eval harness against the sample fixtures
python -m raguard.cli run sample_data/eval_cases.json \
  --html sample_data/report.html \
  --json sample_data/results.json

# 4. Save the current scorecard as a baseline
python -m raguard.cli baseline sample_data/eval_cases.json \
  --save sample_data/baseline.json

# 5. The CI gate: compare a run to the baseline, exit 1 on regression
python -m raguard.cli check sample_data/eval_cases.json \
  --baseline sample_data/baseline.json

# 6. Launch the dashboard
streamlit run raguard/dashboard.py
```

No API key, network access, or paid service is required for any of the
above -- the default judge is a deterministic, offline heuristic.

### Docker

```bash
docker build -t raguard .
docker run raguard                      # runs the CI regression gate
docker run -p 8501:8501 raguard \
  streamlit run raguard/dashboard.py --server.address=0.0.0.0
```

### Upgrading to an LLM-as-judge

By default, faithfulness scoring uses `HeuristicJudge`, a deterministic
mapping of the lexical `groundedness` score onto a 1-5 scale. To use a
real model as judge instead:

```bash
export RAGUARD_JUDGE=openai
export OPENAI_API_KEY=sk-...
# optional, defaults to gpt-4o-mini:
export RAGUARD_OPENAI_MODEL=gpt-4o-mini

python -m raguard.cli run sample_data/eval_cases.json
```

`OpenAIJudge` is never required -- the test suite and CI run entirely
against `HeuristicJudge`.

## CLI reference

```
raguard run <cases.json> [--html out.html] [--json out.json]
raguard baseline <cases.json> --save <path>
raguard check <cases.json> --baseline <path> [--tolerance 0.05]
```

- `run` scores every case, prints the aggregate scorecard, and optionally
  writes an HTML report and/or raw JSON results.
- `baseline` runs the eval and saves the resulting aggregate as a JSON
  baseline for future comparisons.
- `check` runs the eval, compares the aggregate to a saved baseline, and
  **exits 1 if any metric regressed** -- this is the CI gate. Each metric
  is compared direction-aware: `hallucination_flag_rate` and
  `pii_fabrication_rate` are lower-is-better, everything else is
  higher-is-better.

## Project structure

```
raguard/
  metrics.py     # groundedness, relevance, retrieval precision/recall,
                  # citation_accuracy, find_pii, pii_fabrication_check
  judge.py       # Judge ABC, HeuristicJudge (default), OpenAIJudge (opt-in)
  harness.py     # EvalCase, EvalResult, EvalHarness, aggregate()
  regression.py  # Regression, save/load_baseline, compare_to_baseline
  report.py      # render_html_report / save_html_report (escaped, inline CSS)
  cli.py         # run / baseline / check subcommands
  dashboard.py   # Streamlit app
sample_data/
  eval_cases.json  # 7 fixture cases, each proving a specific failure mode
  baseline.json    # committed regression baseline
  report.html      # committed sample HTML report
tests/           # pytest suite (metrics, judge, harness, regression,
                  # report/XSS, CLI)
docs/
  ARCHITECTURE.md # design rationale, data-flow diagram, non-goals
.github/workflows/ci.yml  # pytest -> raguard check (gate) -> upload report
Dockerfile
requirements.txt
```

## The sample fixtures

`sample_data/eval_cases.json` has exactly 7 cases, each built to prove one
thing:

| Case | What it proves |
|---|---|
| `case-1-well-grounded` | A normal, well-supported answer scores high groundedness and isn't flagged. |
| `case-2-hallucinated` | An answer with near-zero lexical overlap with retrieved context trips `hallucination_flag`. |
| `case-3-off-topic` | A grounded-*sounding* non-answer: `groundedness` stays comfortably high while `relevance` drops -- a failure mode groundedness alone can't see. |
| `case-4-hallucinated-bad-citation` | A second hallucination example that also carries a partially-phantom citation (`citation_accuracy = 0.5`). |
| `case-5-pii-fabricated` | A fabricated phone number and email appear in the answer but not in context -- `pii_fabrication_check` flags both. |
| `case-6-pii-sourced` | The *same kind* of PII string, but genuinely present in the retrieved context -- correctly **not** flagged, proving the check discriminates fabricated from sourced. |
| `case-7-phantom-citation` | Decent groundedness (0.5+) but the one citation it makes points at a chunk id that was never retrieved -- `citation_accuracy = 0`, a failure groundedness can't see. |

`sample_data/baseline.json` is a real baseline computed by running the
harness against these fixtures (not hand-typed numbers) -- see the
scorecard below.

## Tech stack

- **Python 3.9+**
- **scikit-learn** -- TF-IDF vectorization + cosine similarity for
  `groundedness` / `relevance`
- **pandas** -- tabular results in the dashboard
- **Streamlit** -- interactive dashboard
- **pytest** -- test suite
- **openai** (optional) -- `OpenAIJudge` backend
- Standard library `argparse`, `dataclasses`, `re`, `html`, `json` for the
  CLI, data model, PII regexes, report escaping, and baseline storage

## Possible extensions

- Swap/augment TF-IDF groundedness with an embedding-based semantic
  similarity metric.
- Swap regex-based `find_pii` for a dedicated PII/NER model.
- Per-metric regression tolerances instead of one global
  `DEFAULT_TOLERANCE`.
- A trend view in the dashboard (scorecard over time across baselines).
- Async/batched `OpenAIJudge` calls for larger eval sets.
- Config-file-driven case generation from real production traces instead
  of hand-written fixtures.

## License

MIT
