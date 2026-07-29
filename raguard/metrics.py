"""
Reference-free RAG evaluation metrics.

Everything here runs fully offline using TF-IDF cosine similarity as a
lightweight semantic-overlap proxy -- no API key or GPU needed, so the
harness can run in CI on every PR. Swap `similarity_fn` for a real
sentence-transformer or an LLM-as-judge (see judge.py) when you want
higher-fidelity scores in exchange for cost/latency.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _pairwise_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english").fit([a, b])
    mat = vec.transform([a, b])
    return float(cosine_similarity(mat[0], mat[1])[0][0])


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """How much of the answer's content is supported by the retrieved context.
    Low groundedness is the standard proxy for hallucination: the model said
    something the retrieved documents don't actually support."""
    if not contexts:
        return 0.0
    joined = " ".join(contexts)
    return _pairwise_similarity(answer, joined)


def relevance(answer: str, question: str) -> float:
    """Does the answer actually address the question asked (vs. a grounded
    but off-topic non-answer)?"""
    return _pairwise_similarity(answer, question)


def retrieval_precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of the top-k retrieved chunks that are actually relevant,
    per a human/gold-labeled relevant-chunk set."""
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
    return hits / len(retrieved_ids)


def retrieval_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of all known-relevant chunks that were actually retrieved."""
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for rid in relevant_ids if rid in retrieved_set)
    return hits / len(relevant_ids)
