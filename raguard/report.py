"""Self-contained HTML report rendering for eval results (no JS, inline CSS)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Sequence, Union

from raguard.harness import EvalResult

_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }
h1 { margin-bottom: 0.2rem; }
.subtitle { color: #666; margin-top: 0; margin-bottom: 1.5rem; }
.scorecard { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }
.card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; min-width: 150px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.03em; }
.card .value { font-size: 1.6rem; font-weight: 600; margin-top: 0.25rem; }
table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
th, td { border: 1px solid #ddd; padding: 0.5rem 0.7rem; text-align: left; font-size: 0.9rem; vertical-align: top; }
th { background: #f0f0f0; position: sticky; top: 0; }
tr.flagged { background: #fdecea; }
tr.flagged td.case-id { font-weight: 600; color: #a61b1b; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-right: 0.25rem; }
.badge.halluc { background: #f8d7da; color: #842029; }
.badge.pii { background: #fff3cd; color: #664d03; }
.truncate { max-width: 320px; overflow-wrap: break-word; }
footer { margin-top: 2rem; color: #999; font-size: 0.8rem; }
"""

_METRIC_LABELS = {
    "n_cases": "Cases",
    "avg_groundedness": "Avg Groundedness",
    "avg_relevance": "Avg Relevance",
    "avg_faithfulness_score": "Avg Faithfulness",
    "avg_retrieval_precision": "Avg Precision",
    "avg_retrieval_recall": "Avg Recall",
    "avg_citation_accuracy": "Avg Citation Acc.",
    "hallucination_flag_rate": "Hallucination Rate",
    "pii_fabrication_rate": "PII Fabrication Rate",
}


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def _scorecard_html(aggregate: Dict) -> str:
    cards = []
    for key, label in _METRIC_LABELS.items():
        if key not in aggregate:
            continue
        cards.append(
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{_fmt(aggregate[key])}</div></div>'
        )
    return '<div class="scorecard">' + "".join(cards) + "</div>"


def _context_preview(contexts) -> str:
    joined = " | ".join(contexts) if contexts else ""
    escaped = html.escape(joined)
    if len(escaped) > 240:
        escaped = escaped[:240] + "…"
    return escaped


def _case_row(result: EvalResult, case_lookup: Dict) -> str:
    flags = []
    if result.hallucination_flag:
        flags.append('<span class="badge halluc">HALLUCINATION</span>')
    if result.has_fabricated_pii:
        flags.append('<span class="badge pii">PII FABRICATED</span>')
    flag_html = "".join(flags)

    row_class = "flagged" if (result.hallucination_flag or result.has_fabricated_pii) else ""

    case = case_lookup.get(result.case_id)
    question = html.escape(getattr(case, "question", "") or "") if case else ""
    answer = html.escape(getattr(case, "answer", "") or "") if case else ""
    contexts = _context_preview(getattr(case, "contexts", []) if case else [])

    fabricated_pii_str = html.escape(", ".join(result.fabricated_pii)) if result.fabricated_pii else "&mdash;"

    return f"""
    <tr class="{row_class}">
      <td class="case-id">{html.escape(result.case_id)}</td>
      <td class="truncate">{question}</td>
      <td class="truncate">{answer}</td>
      <td class="truncate">{contexts}</td>
      <td>{result.groundedness:.3f}</td>
      <td>{result.relevance:.3f}</td>
      <td>{result.retrieval_precision:.3f}</td>
      <td>{result.retrieval_recall:.3f}</td>
      <td>{result.citation_accuracy:.3f}</td>
      <td>{result.faithfulness_score:.2f}</td>
      <td>{flag_html}</td>
      <td>{fabricated_pii_str}</td>
    </tr>"""


def render_html_report(results: Sequence[EvalResult], aggregate: Dict, cases: Sequence = ()) -> str:
    """Render a self-contained HTML report (inline CSS, no JS).

    `cases` is an optional sequence of EvalCase objects (matched by
    `case_id`) used to show the question/answer/context text alongside
    each scored row. All question/answer/context text is HTML-escaped
    before being written into the page -- untrusted case content (e.g. a
    user-submitted question containing a script tag) must never render
    as live markup.
    """
    case_lookup = {c.id: c for c in cases} if cases else {}

    rows = "".join(_case_row(r, case_lookup) for r in results)

    n_flagged = sum(1 for r in results if r.hallucination_flag or r.has_fabricated_pii)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAGuard Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>RAGuard Evaluation Report</h1>
<p class="subtitle">Reference-free RAG/LLM evaluation &mdash; {len(results)} case(s) evaluated, {n_flagged} flagged.</p>

{_scorecard_html(aggregate)}

<table>
  <thead>
    <tr>
      <th>Case ID</th>
      <th>Question</th>
      <th>Answer</th>
      <th>Contexts</th>
      <th>Groundedness</th>
      <th>Relevance</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Citation Acc.</th>
      <th>Faithfulness</th>
      <th>Flags</th>
      <th>Fabricated PII</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<footer>Generated by RAGuard.</footer>
</body>
</html>"""


def save_html_report(
    path: Union[str, Path],
    results: Sequence[EvalResult],
    aggregate: Dict,
    cases: Sequence = (),
) -> None:
    """Render and write the HTML report to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_html_report(results, aggregate, cases=cases)
    with open(path, "w") as f:
        f.write(html_content)
