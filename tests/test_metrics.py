"""Tests for raguard.metrics -- each metric tested in isolation."""

from raguard.metrics import (
    citation_accuracy,
    find_pii,
    groundedness,
    pii_fabrication_check,
    relevance,
    retrieval_precision_at_k,
    retrieval_recall,
)


def test_groundedness_supported_vs_unsupported_and_empty_inputs():
    supported = groundedness(
        "The warranty period is 2 years from the date of purchase.",
        ["Our product warranty lasts for 2 years from the date of purchase."],
    )
    unsupported = groundedness(
        "Our mascot is a friendly dinosaur named Rex.",
        ["Our product warranty lasts for 2 years from the date of purchase."],
    )
    assert supported > 0.5
    assert unsupported < 0.15
    assert groundedness("", ["some context"]) == 0.0
    assert groundedness("an answer", []) == 0.0


def test_relevance_echoed_question_vs_off_topic_answer():
    high = relevance(
        "The maximum file size for uploads is 5 GB.",
        "What is the maximum file size for uploads?",
    )
    low = relevance(
        "Our quarterly revenue grew by 12 percent this year.",
        "How do I restore a deleted file?",
    )
    assert high > 0.5
    assert low < 0.15


def test_relevance_catches_off_topic_answer_that_groundedness_misses():
    """A grounded-but-off-topic answer: high lexical overlap with the
    retrieved context, but it never actually answers the question.
    groundedness should be comfortably non-hallucinatory while relevance
    stays low -- proving relevance catches a failure mode groundedness
    cannot see."""
    question = "How do I restore a file I accidentally deleted?"
    answer = (
        "Deleted files are retained in Trash for 30 days before permanent "
        "removal, and permanently deleted files cannot be recovered after "
        "that window. This policy is reviewed periodically for compliance."
    )
    contexts = [
        "Deleted files are moved to Trash and retained there for 30 days "
        "before permanent removal. To restore a file, open Trash and click "
        "Restore.",
        "Permanently deleted files cannot be recovered after the 30-day "
        "retention window.",
    ]
    g = groundedness(answer, contexts)
    r = relevance(answer, question)
    assert g > 0.3  # not flagged as a hallucination
    assert r < 0.2  # but doesn't actually answer "how do I restore"


def test_retrieval_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert retrieval_precision_at_k(retrieved, ["a", "c"], k=4) == 0.5
    # "d" is outside the top-2 window, so precision@2 should be 0.
    assert retrieval_precision_at_k(retrieved, ["d"], k=2) == 0.0
    assert retrieval_precision_at_k([], ["a"], k=3) == 0.0
    assert retrieval_precision_at_k(["a", "b"], ["a"], k=0) == 0.0


def test_retrieval_recall():
    retrieved = ["a", "b", "c"]
    assert abs(retrieval_recall(retrieved, ["a", "c", "z"]) - (2 / 3)) < 1e-9
    assert retrieval_recall(["a", "b"], []) == 1.0  # vacuous
    assert retrieval_recall([], ["a", "b"]) == 0.0


def test_citation_accuracy_basic_and_edge_cases():
    assert citation_accuracy(["a", "b"], ["a", "b", "c"]) == 1.0
    assert citation_accuracy(["a", "z"], ["a", "b", "c"]) == 0.5
    assert citation_accuracy([], ["a", "b", "c"]) == 1.0  # vacuous


def test_citation_accuracy_catches_phantom_citation_groundedness_misses():
    """A phantom citation: the answer is lexically well-grounded in the
    retrieved context (groundedness is comfortably high), but it cites a
    chunk id that was never retrieved at all. groundedness has no way to
    see this -- it only looks at text. citation_accuracy must catch it."""
    answer = "Files at rest are encrypted using AES-256 encryption."
    contexts = ["Files at rest are encrypted using AES-256 encryption."]
    g = groundedness(answer, contexts)
    c_acc = citation_accuracy(["doc_99"], ["doc_1", "doc_2"])

    assert g > 0.5
    assert c_acc == 0.0


def test_find_pii_detects_all_pattern_types_and_ignores_clean_text():
    text = (
        "Contact Jane at jane.doe@example.com or (415) 555-0199. "
        "SSN on file: 123-45-6789. Card: 4111 1111 1111 1111."
    )
    found = find_pii(text)
    assert any("jane.doe@example.com" in item for item in found)
    assert any("555-0199" in item for item in found)
    assert "123-45-6789" in found
    assert any("4111" in item for item in found)
    assert find_pii("The warranty lasts two years.") == []


def test_pii_fabrication_check_flags_fabricated_not_sourced_pii():
    fabricated = pii_fabrication_check(
        "Call our hotline at (415) 555-0199 for help.",
        ["Our support hours are 9am to 5pm on weekdays."],
    )
    assert len(fabricated) == 1

    # The same kind of PII string, but it genuinely appears in the
    # retrieved context, so it must NOT be flagged as fabricated -- this
    # is what distinguishes fabrication from faithful relaying of
    # retrieved information.
    not_fabricated = pii_fabrication_check(
        "You can reach support at (415) 555-0199.",
        ["Our support line is (415) 555-0199, available 24/7."],
    )
    assert not_fabricated == []

    assert pii_fabrication_check("No contact info here.", ["some context"]) == []
