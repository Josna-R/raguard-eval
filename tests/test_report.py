"""Tests for raguard.report -- HTML rendering, including XSS escaping."""

from raguard.harness import EvalCase, EvalHarness, aggregate
from raguard.report import render_html_report, save_html_report


def _run(cases):
    harness = EvalHarness()
    results = harness.evaluate_batch(cases)
    agg = aggregate(results)
    return results, agg


def test_render_html_report_includes_scorecard_values():
    case = EvalCase(
        id="case-1",
        question="What is the warranty period?",
        answer="The warranty period is 2 years.",
        contexts=["Our warranty lasts 2 years from the date of purchase."],
        retrieved_ids=["d1"],
        relevant_ids=["d1"],
    )
    results, agg = _run([case])
    html_out = render_html_report(results, agg, cases=[case])

    assert "<html" in html_out.lower()
    assert "RAGuard Evaluation Report" in html_out
    assert "case-1" in html_out


def test_render_html_report_flags_hallucinated_rows():
    case = EvalCase(
        id="bad-case",
        question="What is the warranty period?",
        answer="Our office has a rooftop garden and a ping pong table.",
        contexts=["Our warranty lasts 2 years from the date of purchase."],
        retrieved_ids=["d1"],
        relevant_ids=["d1"],
    )
    results, agg = _run([case])
    html_out = render_html_report(results, agg, cases=[case])

    assert 'class="flagged"' in html_out
    assert "HALLUCINATION" in html_out


def test_render_html_report_escapes_xss_payloads_in_question_answer_and_context():
    """Malicious question/answer/context text containing <script> and
    <img onerror> payloads must appear in the output only as escaped
    text, never as live, executable markup."""
    script_payload = "<script>alert(1)</script>"
    img_payload = "<img src=x onerror=alert(1)>"
    case = EvalCase(
        id="xss-case",
        question=f"Ignore instructions. {script_payload}",
        answer=f"Here you go: {img_payload}",
        contexts=[f"Context also contains: {img_payload}"],
        retrieved_ids=["d1"],
        relevant_ids=["d1"],
    )
    results, agg = _run([case])
    html_out = render_html_report(results, agg, cases=[case])

    # The raw, unescaped payloads must never appear in the output.
    assert script_payload not in html_out
    assert img_payload not in html_out
    # The escaped forms must be present instead.
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_out


def test_save_html_report_writes_file(tmp_path):
    case = EvalCase(
        id="case-1",
        question="What is the warranty period?",
        answer="The warranty period is 2 years.",
        contexts=["Our warranty lasts 2 years from the date of purchase."],
        retrieved_ids=["d1"],
        relevant_ids=["d1"],
    )
    results, agg = _run([case])
    out_path = tmp_path / "report.html"
    save_html_report(out_path, results, agg, cases=[case])

    assert out_path.exists()
    content = out_path.read_text()
    assert "RAGuard Evaluation Report" in content
