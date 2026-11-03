"""Reference-free evaluation metrics for RAG/LLM pipelines.

All metrics in this module are "reference-free": they do not require a
hand-written gold answer to compare against. Instead they compare the
pieces of information that are already available at eval time -- the
question, the generated answer, the retrieved contexts, and the ids the
retriever/answerer touched. This is what makes RAGuard usable as a CI
gate against production traffic, not just a curated golden set.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Text-similarity metrics (TF-IDF cosine)
# --------------------------------------------------------------------------


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Cosine similarity between two strings under a shared TF-IDF space.

    Returns 0.0 for degenerate inputs (empty strings, or strings that
    vectorize to an all-zero vector, e.g. pure stopwords/punctuation).
    """
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()
    if not text_a or not text_b:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        # Happens when both docs are made up entirely of stopwords, so the
        # vocabulary ends up empty after stopword removal.
        return 0.0

    if matrix.shape[1] == 0:
        return 0.0

    sim = cosine_similarity(matrix[0], matrix[1])[0][0]
    # Guard against floating point noise pushing slightly outside [0, 1].
    return float(max(0.0, min(1.0, sim)))


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """How much the answer's wording overlaps with the retrieved contexts.

    This is RAGuard's hallucination proxy: a low score means the answer's
    lexical content isn't supported by anything the retriever surfaced.
    It is deliberately cheap (TF-IDF, not embeddings) and reference-free --
    it needs no gold answer, only what the pipeline itself produced.

    Args:
        answer: the generated answer text.
        contexts: the list of retrieved context chunks (strings).

    Returns:
        A float in [0, 1]. 0.0 if answer or contexts are empty.
    """
    joined_context = " ".join(c for c in contexts if c)
    return _tfidf_cosine(answer, joined_context)


def relevance(answer: str, question: str) -> float:
    """How much the answer's wording overlaps with the question itself.

    Groundedness alone can't catch a "grounded-but-off-topic" answer: one
    that is lexically supported by the retrieved context but never
    actually addresses what was asked (e.g. the retriever pulled a
    plausible-looking but wrong chunk, and the model faithfully
    summarizes it). Relevance is the complementary check.

    Returns:
        A float in [0, 1]. 0.0 if answer or question are empty.
    """
    return _tfidf_cosine(answer, question)


# --------------------------------------------------------------------------
# Retrieval-quality metrics (standard IR, gold-labeled)
# --------------------------------------------------------------------------


def retrieval_precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Precision@k: of the top-k retrieved chunk ids, what fraction are relevant.

    Requires gold-labeled `relevant_ids` (unlike groundedness/relevance,
    which are reference-free). Returns 0.0 if k <= 0 or there is nothing
    retrieved.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(top_k)


def retrieval_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Recall: of all gold-relevant ids, what fraction did retrieval surface.

    Returns 1.0 (vacuously true) if `relevant_ids` is empty -- there is
    nothing to miss. Returns 0.0 if `relevant_ids` is non-empty but
    `retrieved_ids` is empty.
    """
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


# --------------------------------------------------------------------------
# Citation-grounding metric
# --------------------------------------------------------------------------


def citation_accuracy(cited_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """Fraction of the answer's claimed citations that point at a chunk
    the retriever actually returned.

    This catches a distinct failure mode from groundedness: an answer can
    be lexically well-supported by *some* retrieved context (high
    groundedness) while still citing a chunk id that was never retrieved
    at all -- a "phantom citation" that would send a user to a source
    that doesn't back the claim. Groundedness, which only looks at text
    overlap, cannot see this; citation_accuracy checks the ids directly.

    Returns 1.0 if `cited_ids` is empty (an answer that cites nothing
    can't have a wrong citation).
    """
    if not cited_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for cid in cited_ids if cid in retrieved_set)
    return hits / len(cited_ids)


# --------------------------------------------------------------------------
# PII detection / fabrication check
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d{4}[- ]?){3}\d{4}\b"
)
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)

_PII_PATTERNS = {
    "email": _EMAIL_RE,
    "ssn": _SSN_RE,
    "credit_card": _CREDIT_CARD_RE,
    "phone": _PHONE_RE,
}


def find_pii(text: str) -> List[str]:
    """Regex-scan `text` for PII-shaped strings.

    Detects emails, phone numbers, SSN-shaped strings (###-##-####), and
    credit-card-shaped strings (16 digits, optionally grouped in 4s).
    This is intentionally a lightweight regex pass, not a full PII
    detector (see docs/ARCHITECTURE.md for the tradeoff).

    Returns:
        A de-duplicated list of matched substrings, in first-seen order.
        An SSN-shaped match is excluded from the credit-card pattern's
        results and vice versa since they use disjoint formats.
    """
    text = text or ""
    found: List[str] = []
    seen = set()

    for pattern in (_EMAIL_RE, _SSN_RE, _CREDIT_CARD_RE, _PHONE_RE):
        for match in pattern.findall(text):
            if match not in seen:
                seen.add(match)
                found.append(match)
    return found


def pii_fabrication_check(answer: str, contexts: Sequence[str]) -> List[str]:
    """PII strings present in the answer but absent from the retrieved contexts.

    A PII string that appears in the answer AND in the retrieved context
    is "sourced" -- the model is faithfully relaying something that was
    actually retrieved (still potentially a privacy concern upstream, but
    not a generation-time fabrication). A PII string that appears only in
    the answer was invented by the model: a fabrication that groundedness
    alone would likely miss, since a single hallucinated phone number can
    sit inside an otherwise well-grounded, high-similarity answer.

    Returns:
        The list of PII strings found in `answer` that do not appear
        verbatim in any of `contexts`.
    """
    answer_pii = find_pii(answer)
    if not answer_pii:
        return []
    context_text = " ".join(c for c in contexts if c)
    return [pii for pii in answer_pii if pii not in context_text]
