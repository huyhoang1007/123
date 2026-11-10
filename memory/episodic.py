"""Persistent, locally ranked memory tiers for the MCP harness."""

from __future__ import annotations

from collections import Counter
from enum import Enum
import math
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np

from retrieval.embeddings import EmbeddingEngine

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_DEFAULT_MEMORY_DB = os.path.expanduser("~/.mcp-harness/memory.sqlite3")
_LEGACY_MEMORY_DB = os.path.expanduser("~/.gemini/long-term-memory.db")


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStore:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        db_path: str | None = None,
        legacy_db_path: str | None = None,
    ):
        is_default_db = db_path is None and "MCP_HARNESS_MEMORY_DB" not in os.environ
        self.db_path = db_path or os.environ.get(
            "MCP_HARNESS_MEMORY_DB",
            _DEFAULT_MEMORY_DB,
        )
        if self.db_path != ":memory:":
            directory = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_access REAL NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0,
                UNIQUE(type, key)
            )
        """)
        self._conn.commit()

        # Per-store cached semantic index to avoid refitting TF-IDF + SVD on every search
        self._embed_engine = EmbeddingEngine()
        self._index_dirty = True
        self._cached_version: tuple[int, float] | None = None
        self._cached_rows: list[sqlite3.Row] = []
        self._cached_docs: list[str] = []
        self._cached_matrix: np.ndarray | None = None

        source_legacy = legacy_db_path or (_LEGACY_MEMORY_DB if is_default_db else None)
        if source_legacy and os.path.exists(source_legacy):
            self._sync_legacy_readonly(source_legacy)

    def _sync_legacy_readonly(self, legacy_path: str) -> int:
        """Import legacy ~/.gemini/long-term-memory.db in strict read-only mode."""
        imported = 0
        try:
            uri = Path(legacy_path).resolve().as_uri() + "?mode=ro"
            src = sqlite3.connect(uri, uri=True)
            now = time.time()
            rows_to_insert: list[tuple] = []
            tables = {
                r[0] for r in src.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "memories" in tables:
                for key, value, category in src.execute(
                    "SELECT key, value, COALESCE(category, '') FROM memories"
                ).fetchall():
                    if key and value:
                        cat = str(category).strip()
                        mtype = "procedural" if cat in ("workflow", "command", "rule") else "semantic"
                        rows_to_insert.append((mtype, str(key), str(value), cat, now, now, now))
            if "kv" in tables:
                for key, value in src.execute("SELECT key, value FROM kv").fetchall():
                    if key and value:
                        rows_to_insert.append(("semantic", f"kv.{key}", str(value), "preference,kv", now, now, now))
            src.close()
            if rows_to_insert:
                cur = self._conn.executemany("""
                    INSERT OR IGNORE INTO memories
                        (type, key, value, tags, created_at, updated_at, last_access)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows_to_insert)
                self._conn.commit()
                imported = max(cur.rowcount, 0)
                if imported > 0:
                    self._index_dirty = True
        except Exception:
            pass
        return imported

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _document(row: sqlite3.Row) -> str:
        return f"{row['key']} {row['tags']} {row['value']}"

    @staticmethod
    def _lexical_score(query: str, document: str) -> float:
        query_counts = Counter(_TOKEN_RE.findall(query.lower()))
        document_counts = Counter(_TOKEN_RE.findall(document.lower()))
        if not query_counts or not document_counts:
            return 0.0
        numerator = sum(
            query_counts[token] * document_counts[token]
            for token in query_counts.keys() & document_counts.keys()
        )
        query_norm = math.sqrt(sum(count * count for count in query_counts.values()))
        document_norm = math.sqrt(
            sum(count * count for count in document_counts.values())
        )
        return numerator / (query_norm * document_norm)

    def _ensure_index(self) -> None:
        """Build or refresh the cached TF-IDF/SVD matrix only when rows changed."""
        version_row = self._conn.execute(
            "SELECT COUNT(*) AS cnt, COALESCE(MAX(updated_at), 0.0) AS max_u FROM memories"
        ).fetchone()
        version = (int(version_row["cnt"]), float(version_row["max_u"]))
        if not self._index_dirty and self._cached_version == version:
            return

        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        docs = [self._document(row) for row in rows]
        self._cached_rows = rows
        self._cached_docs = docs
        if docs:
            vecs = np.asarray(self._embed_engine.rebuild(docs), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            nonzero = (norms > 0).squeeze(axis=1)
            if np.any(nonzero):
                vecs[nonzero] = vecs[nonzero] / norms[nonzero]
            self._cached_matrix = vecs
        else:
            self._cached_matrix = None
        self._cached_version = version
        self._index_dirty = False

    @staticmethod
    def _entry(row: sqlite3.Row) -> dict:
        return {
            "type": row["type"],
            "key": row["key"],
            "value": row["value"],
            "tags": row["tags"],
            "time": row["created_at"],
        }

    def search(
        self,
        query: str,
        mtype: MemoryType | None = None,
        top_k: int = 5,
        min_sim: float = 0.12,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        self._ensure_index()
        if not self._cached_rows:
            return []

        if self._cached_matrix is not None:
            q_vec = np.asarray(self._embed_engine.embed_text(query), dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm > 0:
                q_vec = q_vec / q_norm
            semantic_scores = np.maximum(0.0, self._cached_matrix @ q_vec).tolist()
        else:
            semantic_scores = [0.0] * len(self._cached_rows)

        target_type = mtype.value if mtype is not None else None
        ranked = []
        for row, document, semantic_score in zip(self._cached_rows, self._cached_docs, semantic_scores):
            if target_type is not None and row["type"] != target_type:
                continue
            score = 0.8 * self._lexical_score(query, document) + 0.2 * float(semantic_score)
            if score >= min_sim:
                ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:top_k]
        if selected:
            now = time.time()
            self._conn.executemany(
                "UPDATE memories SET hits = hits + 1, last_access = ? WHERE id = ?",
                [(now, row["id"]) for _, row in selected],
            )
            self._conn.commit()
        return [self._entry(row) for _, row in selected]

    def save(self, mtype: MemoryType, key: str, value: str, tags: str) -> dict:
        now = time.time()
        self._conn.execute("""
            INSERT INTO memories (type, key, value, tags, created_at, updated_at, last_access)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(type, key) DO UPDATE SET
                value = excluded.value,
                tags = excluded.tags,
                updated_at = excluded.updated_at,
                last_access = excluded.last_access
        """, (mtype.value, key, value, tags, now, now, now))
        self._conn.commit()
        self._index_dirty = True
        row = self._conn.execute(
            "SELECT * FROM memories WHERE type = ? AND key = ?",
            (mtype.value, key),
        ).fetchone()
        return self._entry(row)

    def delete(self, mtype: MemoryType, key: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE type = ? AND key = ?", (mtype.value, key)
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            self._index_dirty = True
        return cursor.rowcount > 0

    def list_keys(self, mtype: MemoryType | None = None, limit: int = 30) -> list[str]:
        sql = "SELECT key FROM memories"
        params: tuple = ()
        if mtype is not None:
            sql += " WHERE type = ?"
            params = (mtype.value,)
        sql += " ORDER BY last_access DESC LIMIT ?"
        return [row["key"] for row in self._conn.execute(sql, (*params, limit))]

    def evict_lru(self, keep_top: int) -> int:
        keep_top = max(keep_top, 0)
        before = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self._conn.execute("""
            DELETE FROM memories
            WHERE id NOT IN (
                SELECT id FROM memories ORDER BY last_access DESC, id DESC LIMIT ?
            )
        """, (keep_top,))
        self._conn.commit()
        removed = before - self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if removed > 0:
            self._index_dirty = True
        return removed

    def stats(self) -> dict:
        summary = {"total": 0}
        rows = self._conn.execute("""
            SELECT type, COUNT(*) AS count, COALESCE(SUM(hits), 0) AS hits,
                   COALESCE(SUM((LENGTH(key) + LENGTH(value) + LENGTH(tags)) / 4), 0) AS tokens
            FROM memories GROUP BY type
        """).fetchall()
        for row in rows:
            summary[row["type"]] = {
                "count": row["count"],
                "tokens": row["tokens"],
                "hits": row["hits"],
            }
            summary["total"] += row["count"]
        return summary


class EpisodicMemory:
    def __init__(self, store): self.store = store

    def save(self, key, value, tags):
        return self.store.save(MemoryType.EPISODIC, key, value, tags)


class SemanticMemory:
    def __init__(self, store): self.store = store

    def save(self, key, value, tags):
        return self.store.save(MemoryType.SEMANTIC, key, value, tags)


class ProceduralMemory:
    def __init__(self, store): self.store = store

    def save(self, key, value, tags):
        return self.store.save(MemoryType.PROCEDURAL, key, value, tags)
