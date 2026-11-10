"""
reranker.py — Local cross-score reranker
=========================================
No Hugging Face, no model downloads.

Score = BM25 term-overlap + keyword exact-match bonus + length-penalty.
This approximates cross-encoder relevance for code/documentation retrieval
without any network dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokeniser (strips punctuation)."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Single-document BM25 score against a query."""
    doc_freq = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        tf = doc_freq.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1 + 1)  # single-doc approx; always 1 query doc
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm
    return score


def _keyword_bonus(query_tokens: set[str], doc_tokens: list[str]) -> float:
    """Fraction of distinct query tokens found in doc (0–1)."""
    if not query_tokens:
        return 0.0
    found = sum(1 for t in query_tokens if t in doc_tokens)
    return found / len(query_tokens)


class Reranker:
    """
    Local reranker: BM25 term-overlap + keyword coverage bonus.
    Replaces sentence-transformers CrossEncoder with a zero-download solution.
    """

    def rerank(self, query: str, chunks: list, top_k: int = 12) -> list:
        if not chunks:
            return []

        q_tokens = _tokenize(query)
        q_set = set(q_tokens)

        # Average document length for BM25 normalisation
        doc_token_lists = [_tokenize(f"{getattr(c, 'symbol', '')} {c.content}") for c in chunks]
        avg_dl = sum(len(d) for d in doc_token_lists) / max(len(doc_token_lists), 1)

        raw_lexical = [
            _bm25_score(q_tokens, doc_tokens, avg_dl) * 0.8 + _keyword_bonus(q_set, doc_tokens) * 0.2
            for doc_tokens in doc_token_lists
        ]
        max_lex = max(raw_lexical, default=0.0)

        for chunk, lex in zip(chunks, raw_lexical):
            prior_score = float(getattr(chunk, "_score", 0.0))
            lex_norm = (lex / max_lex) if max_lex > 0.0 else 0.0
            # Blend with hybrid prior score so semantic/graph/priority signals are preserved
            chunk._score = prior_score * 0.65 + lex_norm * 0.35  # type: ignore[attr-defined]

        chunks.sort(key=lambda x: getattr(x, "_score", 0.0), reverse=True)
        return chunks[:top_k]
