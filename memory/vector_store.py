"""
vector_store.py — In-memory numpy vector store
===============================================
No LanceDB, no PyArrow, no fixed dimension constraint.
Stores chunk embeddings in RAM and performs brute-force cosine search.
Fast enough for typical workspaces (<10k chunks).
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    """
    In-memory cosine-similarity vector store.
    Replaces lancedb to remove the fixed 384-dim Hugging Face constraint.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._paths: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._matrix: np.ndarray | None = None  # shape (N, D)

    def clear(self) -> None:
        self._ids.clear()
        self._paths.clear()
        self._id_to_idx.clear()
        self._matrix = None

    def rebuild(self, chunks: list) -> None:
        """Replace the full index so every vector shares one embedding basis."""
        self.clear()
        self.upsert(chunks)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _index_of(self, id_: str) -> int | None:
        return self._id_to_idx.get(id_)

    @staticmethod
    def _fit_dim(vec: np.ndarray, dim: int) -> np.ndarray:
        if vec.shape[0] < dim:
            return np.pad(vec, (0, dim - vec.shape[0]))
        if vec.shape[0] > dim:
            return vec[:dim]
        return vec

    # ── Public API ───────────────────────────────────────────────────────────

    def upsert(self, chunks: list) -> None:
        """Insert or overwrite chunk embeddings in a single O(N) matrix stack."""
        target_dim = self._matrix.shape[1] if self._matrix is not None else None
        pending_rows: list[np.ndarray] = []

        for c in chunks:
            if not c.embedding:
                continue
            vec = np.asarray(c.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm  # pre-normalise for cosine via dot

            if target_dim is None:
                target_dim = vec.shape[0]
            else:
                vec = self._fit_dim(vec, target_dim)

            idx = self._id_to_idx.get(c.id)
            if idx is not None:
                if self._matrix is not None and idx < self._matrix.shape[0]:
                    self._matrix[idx] = vec
                else:
                    pending_rows[idx - (self._matrix.shape[0] if self._matrix is not None else 0)] = vec
                self._paths[idx] = c.path
            else:
                self._id_to_idx[c.id] = len(self._ids)
                self._ids.append(c.id)
                self._paths.append(c.path)
                pending_rows.append(vec)

        if pending_rows:
            stacked = np.vstack(pending_rows)
            self._matrix = stacked if self._matrix is None else np.vstack([self._matrix, stacked])

    def search(self, query_vec: list[float], top_k: int = 50) -> list[dict]:
        """Return top-k chunks by cosine similarity."""
        if self._matrix is None or len(self._ids) == 0:
            return []

        q = np.array(query_vec, dtype=np.float32)
        # Match dimension
        d = self._matrix.shape[1]
        if q.shape[0] < d:
            q = np.pad(q, (0, d - q.shape[0]))
        elif q.shape[0] > d:
            q = q[:d]

        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        # Brute-force cosine (dot of normalised vectors)
        scores = self._matrix @ q  # shape (N,)
        n = min(top_k, len(self._ids))
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {"id": self._ids[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def count(self) -> int:
        return len(self._ids)
