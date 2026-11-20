from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path
import unittest

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_UNIT_TMP = tempfile.TemporaryDirectory()
os.environ.setdefault("MCP_HARNESS_LOG_DIR", _UNIT_TMP.name)
os.environ.setdefault("MCP_HARNESS_MEMORY_DB", str(Path(_UNIT_TMP.name) / "unit_mem.sqlite3"))

import numpy as np

import server
from chunking.compressor import ContextCompressor
from chunking.treesitter_chunker import ASTChunker, Chunk
from context.assembler import ContextAssembler
from context.sanitizer import sanitize
from memory.episodic import EpisodicMemory, MemoryStore, MemoryType, ProceduralMemory, SemanticMemory
from memory.vector_store import VectorStore
from retrieval.bm25 import BM25Index
from retrieval.cache import RetrievalCache
from retrieval.embeddings import EmbeddingEngine
from retrieval.graph import DependencyGraph
from retrieval.reranker import Reranker


class EmbeddingEngineTests(unittest.TestCase):
    def test_rebuild_keeps_index_vectors_on_the_current_basis(self):
        engine = EmbeddingEngine()
        initial = [
            "auth login session cookie",
            "database migration schema index",
            "flutter widget layout color",
        ]
        engine.rebuild(initial)
        all_texts = initial + [
            "http retry timeout backoff",
            "vector search cosine ranking",
            "unittest fixture monkeypatch",
        ]

        rebuilt_vectors = np.asarray(engine.rebuild(all_texts))
        self.assertTrue(
            np.allclose(rebuilt_vectors[:len(initial)], engine._transform(initial))
        )

        before_query = engine._transform(initial)
        engine.embed_text("retrieve the auth session")
        self.assertTrue(np.allclose(before_query, engine._transform(initial)))


class ASTChunkerAndGraphTests(unittest.TestCase):
    def test_stdlib_ast_chunker_extracts_functions_and_classes(self):
        chunker = ASTChunker()
        chunks = chunker.chunk_paths([str(_ROOT / "server.py")])
        self.assertGreater(len(chunks), 15)
        symbols = {c.symbol for c in chunks}
        self.assertIn("retrieve_context", symbols)
        self.assertIn("memory_save", symbols)
        self.assertIn("memory_search", symbols)

        graph = DependencyGraph()
        graph.build(chunks)
        self.assertGreater(graph.graph.number_of_edges(), 0)

    def test_bm25_tokenizes_python_identifiers_and_normalizes_scores(self):
        chunker = ASTChunker()
        chunks = chunker.chunk_paths([str(_ROOT / "server.py")])
        bm25 = BM25Index()
        bm25.build(chunks)
        scores = bm25.score("retrieve_context hybrid")
        self.assertEqual(len(scores), len(chunks))
        self.assertLessEqual(max(scores), 1.0)
        self.assertGreater(max(scores), 0.5)

    def test_gemini_compressor_preserves_raw_code_without_double_json_encoding(self):
        comp = ContextCompressor()
        chunk = Chunk(
            id="c1", path="demo.py", type="function_definition",
            symbol="my_func", content="def my_func():\n    return 42",
            summary="function_definition my_func", tokens=10,
        )
        compressed = comp.compress([chunk], model="gemini-2.5-flash")[0]
        self.assertEqual(compressed, "def my_func():\n    return 42")
        self.assertFalse(compressed.startswith("{"))


class RetrievalCacheTests(unittest.TestCase):
    def test_cache_invalidates_on_source_change_and_returns_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("initial source", encoding="utf-8")
            cache = RetrievalCache(ttl=60)
            chunk = Chunk(
                id="chunk-1", path=str(source), type="file", symbol="module",
                content="original", summary="", tokens=1,
            )
            options = {"rerank_top_k": 12}
            cache.set("query", [str(source)], [chunk], options)

            cached = cache.get("query", [str(source)], options)
            self.assertIsNotNone(cached)
            cached[0].content = "mutated by presentation"
            self.assertEqual(
                cache.get("query", [str(source)], options)[0].content, "original"
            )

            source.write_text("changed source with a different size", encoding="utf-8")
            self.assertIsNone(cache.get("query", [str(source)], options))

    def test_cache_ignores_git_and_node_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("def ok(): pass\n", encoding="utf-8")
            fp_before = RetrievalCache.path_fingerprint(str(root))
            git_dir = root / ".git" / "objects" / "00"
            git_dir.mkdir(parents=True)
            (git_dir / "deadbeef").write_text("blob", encoding="utf-8")
            nm_dir = root / "node_modules" / "pkg"
            nm_dir.mkdir(parents=True)
            (nm_dir / "index.js").write_text("module.exports = 1;", encoding="utf-8")
            fp_after = RetrievalCache.path_fingerprint(str(root))
            self.assertEqual(fp_before, fp_after)


class RerankerAndSanitizerTests(unittest.TestCase):
    def test_reranker_preserves_hybrid_prior_score(self):
        c_sem = Chunk(
            id="sem", path="auth.py", type="function_definition",
            symbol="refresh_session", content="def renew_jwt(): pass",
            summary="", tokens=5,
        )
        c_sem._score = 0.95
        c_kw = Chunk(
            id="kw", path="noise.py", type="function_definition",
            symbol="noise", content="auth flow " * 3,
            summary="", tokens=10,
        )
        c_kw._score = 0.05
        reranked = Reranker().rerank("auth flow", [c_sem, c_kw], top_k=2)
        self.assertEqual(reranked[0].id, "sem")

    def test_sanitizer_is_case_insensitive(self):
        self.assertEqual(sanitize("Ignore Previous Instructions"), "[REDACTED]")
        self.assertEqual(sanitize("SYSTEM PROMPT leak", escape_xml=False), "[REDACTED] leak")

    def test_gemini_output_modes_and_claude_xml_escaping(self):
        asm = ContextAssembler()
        chunk = Chunk(
            id="c1", path="a.py", type="function_definition",
            symbol="fn", content="if a < b:\n    return '</content>'",
            summary="", tokens=8,
        )
        mem = [{"type": "semantic", "key": "k", "value": "v", "time": 1.0}]
        concise = asm.assemble("Base", [chunk], mem, "q", "concise", "gemini-2.5-flash")
        structured = asm.assemble("Base", [chunk], mem, "q", "structured", "gemini-2.5-flash")
        code_only = asm.assemble("Base", [chunk], mem, "q", "code_only", "gemini-2.5-flash")
        self.assertLess(len(concise), len(structured))
        self.assertNotIn("memory", code_only)
        claude = asm.assemble("Base", [chunk], mem, "q", "concise", "claude-sonnet-4")
        self.assertIn("&lt;/content&gt;", claude)

    def test_vector_store_upsert_updates_and_appends(self):
        vs = VectorStore()
        c1 = Chunk(id="1", path="a.py", type="fn", symbol="f1", content="x", summary="", tokens=1, embedding=[1.0, 0.0])
        c2 = Chunk(id="2", path="b.py", type="fn", symbol="f2", content="y", summary="", tokens=1, embedding=[0.0, 1.0])
        vs.rebuild([c1, c2])
        self.assertEqual(vs.count(), 2)
        c1_updated = Chunk(id="1", path="a.py", type="fn", symbol="f1", content="x", summary="", tokens=1, embedding=[0.0, 1.0])
        c3 = Chunk(id="3", path="c.py", type="fn", symbol="f3", content="z", summary="", tokens=1, embedding=[1.0, 0.0])
        vs.upsert([c1_updated, c3])
        self.assertEqual(vs.count(), 3)
        res = vs.search([1.0, 0.0], top_k=1)
        self.assertEqual(res[0]["id"], "3")


class RetiredToolsContractTests(unittest.TestCase):
    def test_compaction_and_deleted_tools_are_not_registered(self):
        for tool_name in (
            "compact_conversation",
            "handoff_conversation",
            "restore_conversation_handoff",
            "memory_delete",
            "memory_list",
            "memory_evict",
            "memory_stats",
            "get_token_budget",
        ):
            self.assertFalse(hasattr(server, tool_name), f"{tool_name} must not be on server")
            self.assertNotIn(tool_name, server.mcp._tool_manager._tools, f"{tool_name} must not be in MCP")


class MemoryStoreTests(unittest.TestCase):
    def test_memory_persists_and_filters_by_ranked_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "memory.sqlite3")
            first_store = MemoryStore(db_path)
            first_store.save(
                MemoryType.SEMANTIC,
                "auth-refresh-policy",
                "Refresh token rotation prevents token replay after login.",
                "auth,security",
            )
            first_store.save(
                MemoryType.SEMANTIC,
                "ui-color-scale",
                "Use neutral colors for dashboard controls.",
                "ui,design",
            )

            second_store = MemoryStore(db_path)
            matches = second_store.search(
                "refresh token rotation", mtype=MemoryType.SEMANTIC,
                min_sim=0.2,
            )
            self.assertEqual([match["key"] for match in matches], ["auth-refresh-policy"])
            self.assertEqual(
                second_store.search("unrelated gardening question", min_sim=0.95), []
            )
            stats = second_store.stats()
            self.assertEqual(stats["total"], 2)
            self.assertGreaterEqual(stats["semantic"]["hits"], 1)
            first_store.close()
            second_store.close()

    def test_memory_store_syncs_legacy_db_readonly_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            legacy_db = tmp / "legacy-long-term-memory.db"
            conn = sqlite3.connect(legacy_db)
            try:
                conn.execute("CREATE TABLE memories (key TEXT, value TEXT, category TEXT)")
                conn.execute("CREATE TABLE kv (key TEXT, value TEXT)")
                conn.execute(
                    "INSERT INTO memories (key, value, category) VALUES (?, ?, ?)",
                    ("ps-utf8", "Prepend UTF-8 output encoding in scripts.", "workflow"),
                )
                conn.execute(
                    "INSERT INTO memories (key, value, category) VALUES (?, ?, ?)",
                    ("arch-soc", "Keep modules decoupled.", "architecture"),
                )
                conn.execute(
                    "INSERT INTO kv (key, value) VALUES (?, ?)",
                    ("shell", "powershell-5.1"),
                )
                conn.commit()
            finally:
                conn.close()

            before_bytes = legacy_db.read_bytes()
            db_path = str(tmp / "memory.sqlite3")
            store = MemoryStore(db_path, legacy_db_path=str(legacy_db))
            try:
                stats = store.stats()
                self.assertEqual(stats["total"], 3)
                self.assertEqual(stats["procedural"]["count"], 1)
                self.assertEqual(stats["semantic"]["count"], 2)
                kv_hits = store.search("powershell shell", mtype=MemoryType.SEMANTIC, min_sim=0.1)
                self.assertTrue(any(h["key"] == "kv.shell" for h in kv_hits))
                self.assertEqual(legacy_db.read_bytes(), before_bytes)
            finally:
                store.close()

    def test_memory_index_caches_across_searches_and_invalidates_on_save(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "cache_test.sqlite3"))
            try:
                store.save(MemoryType.SEMANTIC, "k1", "alpha beta gamma", "t1")
                store.save(MemoryType.SEMANTIC, "k2", "delta epsilon zeta", "t2")
                res1 = store.search("alpha beta", min_sim=0.25)
                self.assertEqual([r["key"] for r in res1], ["k1"])
                cached_matrix = store._cached_matrix
                res2 = store.search("delta epsilon", min_sim=0.25)
                self.assertEqual([r["key"] for r in res2], ["k2"])
                self.assertIs(store._cached_matrix, cached_matrix)

                store.save(MemoryType.SEMANTIC, "k3", "omega theta iota", "t3")
                self.assertTrue(store._index_dirty)
                res3 = store.search("omega theta", min_sim=0.25)
                self.assertEqual([r["key"] for r in res3], ["k3"])
                self.assertIsNot(store._cached_matrix, cached_matrix)
            finally:
                store.close()


class IndexRefreshTests(unittest.TestCase):
    def setUp(self):
        self._orig_mem = server.mem_store
        self._orig_ep = server.episodic
        self._orig_sem = server.semantic
        self._orig_proc = server.procedural
        self._tmpdir = tempfile.TemporaryDirectory()
        self._test_mem = MemoryStore(str(Path(self._tmpdir.name) / "test_mem.sqlite3"))
        server.mem_store = self._test_mem
        server.episodic = EpisodicMemory(self._test_mem)
        server.semantic = SemanticMemory(self._test_mem)
        server.procedural = ProceduralMemory(self._test_mem)

        server._all_chunks.clear()
        server._indexed_paths.clear()
        server._indexed_fingerprints.clear()
        server.vec_store.clear()
        server.bm25.build([])
        server.graph.build([])
        server.cache.invalidate_all()

    def tearDown(self):
        server._all_chunks.clear()
        server._indexed_paths.clear()
        server._indexed_fingerprints.clear()
        server.vec_store.clear()
        server.bm25.build([])
        server.graph.build([])
        server.cache.invalidate_all()

        self._test_mem.close()
        self._tmpdir.cleanup()
        server.mem_store = self._orig_mem
        server.episodic = self._orig_ep
        server.semantic = self._orig_sem
        server.procedural = self._orig_proc

    def test_refresh_replaces_chunks_instead_of_accumulating_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "context.txt"
            source.write_text("first revision", encoding="utf-8")
            server._ensure_indexed([str(source)])
            self.assertEqual(len(server._all_chunks), 1)

            source.write_text("second revision with changed content", encoding="utf-8")
            server._ensure_indexed([str(source)])
            self.assertEqual(len(server._all_chunks), 1)
            self.assertIn("second revision", server._all_chunks[0].content)

            server._ensure_indexed([str(source)], force_reindex=True)
            self.assertEqual(len(server._all_chunks), 1)

    def test_default_prompt_cap_uses_dynamic_target_ratio_and_eleven_active_tools(self):
        self.assertEqual(server.DEFAULT_MAX_PROMPT_TOKENS, 0)
        self.assertFalse(server.DEFAULT_INCLUDE_REPORT)
        budget = server.get_budget("gemini-2.5-flash")
        self.assertEqual(budget.inject_budget, 952_000)
        limit = server._prompt_token_limit(
            context_window=budget.context_window,
            inject_budget=budget.inject_budget,
            target_ratio=0.20,
            max_prompt_tokens=server.DEFAULT_MAX_PROMPT_TOKENS,
        )
        self.assertEqual(limit, 200_000)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.py"
            source.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
            large_base = "system rule token " * 3_000  # > 4,000 tokens
            out = server.retrieve_context(
                base_prompt=large_base,
                paths=[str(source)],
                query="greet",
            )
            self.assertNotIn("rejected output", out)
            self.assertFalse(out.startswith("Error"))
            self.assertIn("greet", out)

            mem_out = server.memory_inject(
                base_prompt=large_base,
                query="greet",
            )
            self.assertNotIn("rejected output", mem_out)
            self.assertFalse(mem_out.startswith("Error"))

        registered = set(server.mcp._tool_manager._tools.keys())
        self.assertEqual(registered, {
            "retrieve_context",
            "memory_save",
            "memory_search",
            "memory_inject",
            "invalidate_cache",
            "reindex_paths",
            "estimate_tokens",
            "system_port_status",
            "system_port_kill",
            "system_env_status",
            "sqlite_db_backup",
        })

        env = server.system_env_status()
        self.assertNotEqual(env["git_branch"], "unknown")

        with tempfile.TemporaryDirectory() as non_git_dir:
            isolated = server.system_env_status(workspace_path=non_git_dir)
            self.assertEqual(isolated["git_branch"], "unknown")
            self.assertFalse(isolated["git_dirty"])


def tearDownModule() -> None:
    try:
        if getattr(server, "mem_store", None) is not None:
            server.mem_store.close()
        MemoryStore._instance = None
        root_logger = logging.getLogger()
        for handler in list(getattr(server, "_log_handlers", [])):
            root_logger.removeHandler(handler)
            handler.close()
    finally:
        _UNIT_TMP.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
