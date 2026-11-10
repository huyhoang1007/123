from __future__ import annotations

import re
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
        tokenized_corpus = [
            _tokenize(f"{c.symbol} {c.summary} {c.content}") or ["_empty_"]
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def score(self, query: str) -> list[float]:
        if not self.bm25:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return [0.0] * len(self.chunks)
        raw_scores = self.bm25.get_scores(tokenized_query).tolist()
        max_score = max(raw_scores, default=0.0)
        if max_score <= 0.0:
            return [0.0 for _ in raw_scores]
        return [max(0.0, float(s) / max_score) for s in raw_scores]
