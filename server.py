"""
server.py — Harness context engineering v5.0
=========================================
Context Orchestration + Mini-RAG Engine

MCP Tools exposed (11 active tools):
  retrieve_context       — hybrid retrieval (semantic + BM25 + graph)
  memory_save            — save to episodic / semantic / procedural memory
  memory_search          — search across memory tiers
  memory_inject          — inject relevant memory into prompt
  invalidate_cache       — clear retrieval cache
  reindex_paths          — force re-chunking and re-indexing of paths
  estimate_tokens        — calculate token footprint
  system_port_status     — inspect port listener and owning process
  system_port_kill       — terminate process holding TCP port
  system_env_status      — get git status, active ports, python version, disk space
  sqlite_db_backup       — native sqlite3.backup() tool
"""

from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from copy import deepcopy
from context.sanitizer import sanitize, sanitize_chunks
from context.assembler import ContextAssembler
from context.budgeting import get_budget
from memory.vector_store import VectorStore
from memory.episodic import EpisodicMemory, SemanticMemory, ProceduralMemory, MemoryStore
from chunking.compressor import ContextCompressor
from chunking.treesitter_chunker import ASTChunker, Chunk, count_tokens
from retrieval.cache import RetrievalCache
from retrieval.graph import DependencyGraph
from retrieval.reranker import Reranker
from retrieval.bm25 import BM25Index
from retrieval.embeddings import EmbeddingEngine
from mcp.server.fastmcp import FastMCP

import os
import json
import logging
import logging.handlers
import time


# ── Sub-modules ─────────────────────────────────────────────────────────


# ── Logging ─────────────────────────────────────────────────────────────
# Primary log: provider-neutral location under ~/.mcp-harness/
LOG_DIR = os.path.expanduser("~/.mcp-harness")
LOG_FILE = os.path.join(LOG_DIR, "harness-v3.log")
os.makedirs(LOG_DIR, exist_ok=True)
_LEGACY_LINK = os.path.join(os.path.expanduser("~/.gemini"), "mcp-harness-v3.log")

_log_handlers: list[logging.Handler] = [
    logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=524288, backupCount=2, encoding="utf-8"
    ),
]

# If a stale legacy file already exists and is not the primary log file,
# write to it as well so older Get-Content -Wait commands keep working.
try:
    if os.path.exists(_LEGACY_LINK) and not os.path.samefile(LOG_FILE, _LEGACY_LINK):
        _log_handlers.append(logging.FileHandler(_LEGACY_LINK, encoding="utf-8"))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger("harness.server")


class _SuppressDiscoverWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "server/discover" in msg and (
            "Failed to validate request" in msg or "validation errors for" in msg
        ):
            return False
        if record.exc_info and record.exc_info[1] is not None:
            if "server/discover" in str(record.exc_info[1]):
                return False
        return True


logging.getLogger().addFilter(_SuppressDiscoverWarningFilter())
for _h in _log_handlers:
    _h.addFilter(_SuppressDiscoverWarningFilter())

DEFAULT_MAX_PROMPT_TOKENS = 0
DEFAULT_INCLUDE_REPORT = False


def _coerce_target_ratio(target_ratio: float) -> float:
    """Return a sane prompt target ratio in (0, 1]."""
    try:
        ratio = float(target_ratio)
    except (TypeError, ValueError):
        return 0.20
    if ratio <= 0:
        return 0.20
    return min(ratio, 1.0)


def _prompt_token_limit(
    *,
    context_window: int,
    inject_budget: int,
    target_ratio: float,
    max_prompt_tokens: int,
) -> int:
    """Resolve the global prompt cap, never exceeding the inject budget."""
    if max_prompt_tokens and max_prompt_tokens > 0:
        return max(1, min(int(max_prompt_tokens), inject_budget))

    ratio_limit = int(context_window * _coerce_target_ratio(target_ratio))
    return max(1, min(ratio_limit, inject_budget))


def _optimizer_report_comment(
    *,
    before_tokens: int,
    after_tokens: int,
    target_tokens: int,
    chunks_before: int,
    chunks_after: int,
    memory_before: int,
    memory_after: int,
    status: str,
) -> str:
    saved_tokens = max(before_tokens - after_tokens, 0)
    saved_percent = (
        round(saved_tokens * 100 / before_tokens, 1)
        if before_tokens > 0 else 0.0
    )
    return (
        "<!-- prompt_optimizer "
        f"status={status} before={before_tokens} after={after_tokens} "
        f"saved={saved_tokens} saved_percent={saved_percent}% "
        f"target={target_tokens} chunks={chunks_after}/{chunks_before} "
        f"memory={memory_after}/{memory_before} -->"
    )


def _with_optimizer_report(
    prompt: str,
    *,
    before_tokens: int,
    target_tokens: int,
    chunks_before: int,
    chunks_after: int,
    memory_before: int,
    memory_after: int,
    status: str,
) -> str:
    """Prepend a compact report and make its `after` token count exact."""
    provisional = _optimizer_report_comment(
        before_tokens=before_tokens,
        after_tokens=count_tokens(prompt),
        target_tokens=target_tokens,
        chunks_before=chunks_before,
        chunks_after=chunks_after,
        memory_before=memory_before,
        memory_after=memory_after,
        status=status,
    )
    after_tokens = count_tokens(f"{provisional}\n{prompt}")
    final = _optimizer_report_comment(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        target_tokens=target_tokens,
        chunks_before=chunks_before,
        chunks_after=chunks_after,
        memory_before=memory_before,
        memory_after=memory_after,
        status=status,
    )
    return f"{final}\n{prompt}"


def _fit_prompt_to_limit(
    *,
    base_prompt: str,
    chunks: list[Chunk],
    memory_entries: list[dict],
    query: str,
    output_mode: str,
    model: str,
    target_tokens: int,
    before_tokens: int,
    include_report: bool,
    base_prompt_policy: str,
) -> tuple[str, list[Chunk], list[dict], str]:
    """
    Enforce the global prompt cap against the fully assembled output.

    If retrieved context and memory cannot be trimmed enough because the base
    prompt itself exceeds the cap, return a clear rejection unless callers opt
    into unsafe base prompt truncation with base_prompt_policy="truncate".
    """
    selected_chunks = list(chunks)
    selected_memory = list(memory_entries)
    chunks_before = len(selected_chunks)
    memory_before = len(selected_memory)

    def render(ch_list: list[Chunk], mem_list: list[dict], status: str) -> tuple[str, int]:
        prompt = assembler.assemble(
            base_prompt=base_prompt,
            chunks=ch_list,
            memory_entries=mem_list,
            query=query,
            output_mode=output_mode,
            model=model,
        )
        if include_report:
            prompt = _with_optimizer_report(
                prompt,
                before_tokens=before_tokens,
                target_tokens=target_tokens,
                chunks_before=chunks_before,
                chunks_after=len(ch_list),
                memory_before=memory_before,
                memory_after=len(mem_list),
                status=status,
            )
        return prompt, count_tokens(prompt)

    prompt, tokens = render(selected_chunks, selected_memory, "ok")
    if tokens <= target_tokens:
        return prompt, selected_chunks, selected_memory, "ok"

    # Binary search largest prefix of chunks that fits (O(log K) instead of O(K))
    if selected_chunks:
        prompt0, tokens0 = render([], selected_memory, "trimmed")
        if tokens0 > target_tokens:
            selected_chunks = []
            prompt, tokens = prompt0, tokens0
        else:
            lo, hi = 0, len(selected_chunks)
            best_k, best_prompt = 0, prompt0
            while lo <= hi:
                mid = (lo + hi) // 2
                cand_prompt, cand_tokens = render(selected_chunks[:mid], selected_memory, "trimmed")
                if cand_tokens <= target_tokens:
                    best_k, best_prompt = mid, cand_prompt
                    lo = mid + 1
                else:
                    hi = mid - 1
            selected_chunks = selected_chunks[:best_k]
            return best_prompt, selected_chunks, selected_memory, "trimmed"

    while tokens > target_tokens and selected_memory:
        selected_memory.pop()
        prompt, tokens = render(selected_chunks, selected_memory, "trimmed")

    if tokens <= target_tokens:
        return prompt, selected_chunks, selected_memory, "trimmed"

    if base_prompt_policy.lower() == "truncate":
        return _fit_with_truncated_base_prompt(
            base_prompt=base_prompt,
            chunks=[],
            memory_entries=[],
            query=query,
            output_mode=output_mode,
            model=model,
            target_tokens=target_tokens,
            before_tokens=before_tokens,
            include_report=include_report,
            chunks_before=chunks_before,
            memory_before=memory_before,
        )

    base_tokens = count_tokens(base_prompt)
    rejection = (
        "Error: prompt_optimizer rejected output because the base prompt alone "
        "does not fit the global prompt target.\n"
        f"base_prompt_tokens={base_tokens:,}; target_tokens={target_tokens:,}; "
        "set a higher max_prompt_tokens/target_ratio, reduce base_prompt, or "
        "use base_prompt_policy='truncate' if lossy truncation is acceptable."
    )
    if include_report:
        rejection = _with_optimizer_report(
            rejection,
            before_tokens=before_tokens,
            target_tokens=target_tokens,
            chunks_before=chunks_before,
            chunks_after=0,
            memory_before=memory_before,
            memory_after=0,
            status="rejected_base_prompt",
        )
    return rejection, [], [], "rejected_base_prompt"


def _fit_with_truncated_base_prompt(
    *,
    base_prompt: str,
    chunks: list[Chunk],
    memory_entries: list[dict],
    query: str,
    output_mode: str,
    model: str,
    target_tokens: int,
    before_tokens: int,
    include_report: bool,
    chunks_before: int,
    memory_before: int,
) -> tuple[str, list[Chunk], list[dict], str]:
    """Last-resort lossy base prompt truncation."""
    marker = "\n\n[base_prompt truncated by prompt_optimizer]\n"
    available = max(1, target_tokens - count_tokens(query) - 120)
    while available > 0:
        truncated_base = _truncate_to_tokens(base_prompt, available) + marker
        prompt = assembler.assemble(
            base_prompt=truncated_base,
            chunks=chunks,
            memory_entries=memory_entries,
            query=query,
            output_mode=output_mode,
            model=model,
        )
        if include_report:
            prompt = _with_optimizer_report(
                prompt,
                before_tokens=before_tokens,
                target_tokens=target_tokens,
                chunks_before=chunks_before,
                chunks_after=0,
                memory_before=memory_before,
                memory_after=0,
                status="truncated_base_prompt",
            )
        if count_tokens(prompt) <= target_tokens:
            return prompt, [], [], "truncated_base_prompt"
        available = int(available * 0.8)

    rejection = (
        "Error: prompt_optimizer could not fit even the truncated base prompt "
        f"under target_tokens={target_tokens:,}."
    )
    return rejection, [], [], "rejected_base_prompt"


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Token-aware truncation with a character fallback."""
    if count_tokens(text) <= max_tokens:
        return text
    try:
        import tiktoken as _tiktoken
        enc = _tiktoken.get_encoding("cl100k_base")
        return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    except Exception:
        return text[:max_tokens * 3]


# ── Singletons ──────────────────────────────────────────────────────────
mcp = FastMCP("Harness context engineering")
embedder = EmbeddingEngine()
bm25 = BM25Index()
reranker = Reranker()
graph = DependencyGraph()
cache = RetrievalCache(ttl=600)
chunker = ASTChunker()
compressor = ContextCompressor()
vec_store = VectorStore()
assembler = ContextAssembler()
mem_store = MemoryStore.instance()
episodic = EpisodicMemory(mem_store)
semantic = SemanticMemory(mem_store)
procedural = ProceduralMemory(mem_store)

# ── In-memory chunk index ───────────────────────────────────────────────
_indexed_paths: set[str] = set()
_indexed_fingerprints: dict[str, str] = {}
_all_chunks: list[Chunk] = []


def _normalise_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _chunk_belongs_to_path(
    chunk: Chunk,
    root: str,
    is_dir_root: bool,
) -> bool:
    """Return whether a chunk must be replaced when source_path is reindexed."""
    chunk_path = _normalise_path(chunk.path)
    if is_dir_root:
        try:
            return os.path.commonpath([chunk_path, root]) == root
        except ValueError:
            return False
    return chunk_path == root


def _rebuild_indexes() -> None:
    """Rebuild every derived index from the canonical chunk list atomically."""
    texts = [
        f"{chunk.symbol} {chunk.summary}\n{chunk.content[:1500]}"
        for chunk in _all_chunks
    ]
    vectors = embedder.rebuild(texts)
    for chunk, vector in zip(_all_chunks, vectors):
        chunk.embedding = vector
    vec_store.rebuild(_all_chunks)
    bm25.build(_all_chunks)
    graph.build(_all_chunks)


def _ensure_indexed(
        paths: list[str],
        force_reindex: bool = False,
        fingerprints: dict[str, str] | None = None) -> list[Chunk]:
    """
    Build/update the chunk index for the given paths.
    Skips paths already indexed unless force_reindex=True.
    """
    fp_map = fingerprints if fingerprints is not None else {}
    new_paths = []
    for path in paths:
        normalised = _normalise_path(path)
        fingerprint = fp_map.get(path) or cache.path_fingerprint(path)
        fp_map[path] = fingerprint
        if (
            force_reindex
            or normalised not in _indexed_paths
            or _indexed_fingerprints.get(normalised) != fingerprint
        ):
            new_paths.append(path)
    if not new_paths and _all_chunks:
        return _all_chunks

    t0 = time.time()
    chunks = chunker.chunk_paths(new_paths)
    sanitize_chunks(chunks)

    # Precompute root metadata once (O(M * P) instead of O(M^2 * P))
    existing_norm_paths = {_normalise_path(c.path) for c in _all_chunks}
    root_specs = []
    for path in new_paths:
        root = _normalise_path(path)
        prefix = f"{root}{os.sep}"
        is_dir = os.path.isdir(root) or any(p.startswith(prefix) for p in existing_norm_paths)
        root_specs.append((root, is_dir))

    # Remove stale chunks before adding replacements. This handles changed and
    # deleted files without accumulating duplicate UUID-based chunk records.
    _all_chunks[:] = [
        chunk for chunk in _all_chunks
        if not any(_chunk_belongs_to_path(chunk, root, is_dir) for root, is_dir in root_specs)
    ]
    _all_chunks.extend(chunks)
    _rebuild_indexes()

    for path in new_paths:
        normalised = _normalise_path(path)
        _indexed_paths.add(normalised)
        _indexed_fingerprints[normalised] = fp_map[path]
    cache.invalidate_all()
    logger.info(
        f"[Server] refreshed {len(chunks)} chunks in "
        f"{time.time() - t0:.1f}s | total={len(_all_chunks)}")
    return _all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def retrieve_context(
    base_prompt: str,
    paths: list[str],
    query: str,
    model: str = "gemini-3.8-flash",
    output_mode: str = "concise",
    retrieve_top_n: int = 50,
    rerank_top_k: int = 12,
    memory_top_k: int = 4,
    min_memory_sim: float = 0.18,
    graph_expand: bool = True,
    compress: bool = True,
    force_reindex: bool = False,
    target_ratio: float = 0.20,
    max_prompt_tokens: int = 0,
    include_report: bool = DEFAULT_INCLUDE_REPORT,
    base_prompt_policy: str = "reject",
) -> str:
    """Hybrid AST + vector + BM25 + graph RAG retrieval with memory injection and token-budget trimming.
    output_mode: 'concise' | 'structured' | 'code_only' | 'minimal'. base_prompt_policy: 'reject' | 'truncate'."""
    try:
        budget = get_budget(model)
        target_tokens = _prompt_token_limit(
            context_window=budget.context_window,
            inject_budget=budget.inject_budget,
            target_ratio=target_ratio,
            max_prompt_tokens=max_prompt_tokens,
        )
        logger.info(
            f"model={model!r} budget={budget.context_window:,} "
            f"inject={budget.inject_budget:,} target={target_tokens:,}")
        logger.info(f"retrieve_context | query={query[:100]!r} | paths={paths}")

        # ── 1. Compute path fingerprints once & refresh canonical index ──────
        fingerprints = {p: cache.path_fingerprint(p) for p in paths}
        all_chunks = _ensure_indexed(paths, force_reindex, fingerprints=fingerprints)
        cache_options = {
            "retrieve_top_n": retrieve_top_n,
            "rerank_top_k": rerank_top_k,
            "graph_expand": graph_expand,
        }
        cached = (
            None
            if force_reindex
            else cache.get(query, paths, cache_options, fingerprints=fingerprints)
        )
        if cached:
            logger.info(f"  [Cache HIT] {len(cached)} chunks")
            final_chunks = cached
        else:
            # ── 2. Retrieve from the fresh index ──────────────────────────────
            if not all_chunks:
                logger.info("  No chunks indexed from provided paths")
                final_chunks = []
            else:
                # ── 3. Hybrid scoring ────────────────────────────────────────
                query_vec = embedder.embed_text(query)
                vec_results = vec_store.search(query_vec, top_k=retrieve_top_n)
                bm25_scores = bm25.score(query)

                # Build chunk lookup by id and union semantic + BM25 top candidates
                chunk_by_id = {c.id: c for c in all_chunks}
                sem_by_id = {r["id"]: float(r.get("score", 0.0)) for r in vec_results}
                bm25_by_idx = {c.id: s for c, s in zip(bm25.chunks, bm25_scores)}
                top_bm25_ids = [
                    cid for cid, sc in sorted(
                        bm25_by_idx.items(), key=lambda kv: kv[1], reverse=True
                    )[:retrieve_top_n] if sc > 0.0
                ]
                candidate_ids = list(dict.fromkeys([*sem_by_id.keys(), *top_bm25_ids]))

                candidates: list[Chunk] = []
                for cid in candidate_ids:
                    c = chunk_by_id.get(cid)
                    if c is None:
                        continue

                    sem_score = max(0.0, sem_by_id.get(cid, 0.0))
                    bm25_score = bm25_by_idx.get(cid, 0.0)
                    prio_score = c.priority / 10.0
                    grph_score = graph.symbol_score(query, c)

                    final = (
                        sem_score * 0.60 +
                        bm25_score * 0.25 +
                        prio_score * 0.10 +
                        grph_score * 0.05
                    )
                    c._score = final   # type: ignore[attr-defined]
                    candidates.append(c)

                candidates.sort(
                    key=lambda x: getattr(
                        x, "_score", 0), reverse=True)

                # ── 4. Rerank top-N ──────────────────────────────────────────
                reranked = reranker.rerank(
                    query, candidates[:retrieve_top_n], top_k=rerank_top_k)

                # ── 5. Graph expansion ───────────────────────────────────────
                if graph_expand:
                    extras = graph.expand(reranked, max_extra=4)
                    for e in extras:
                        if e.id not in {c.id for c in reranked}:
                            e._score = 0.3   # type: ignore[attr-defined]
                            reranked.append(e)

                # Cache raw copies. Compression below is presentation-specific
                # and must never mutate the source index or a cache entry.
                cache.set(query, paths, reranked, cache_options, fingerprints=fingerprints)
                final_chunks = deepcopy(reranked)

        # ── 6. Compress candidates (before budget enforcement) ───────────────
        # Compress first so budget is enforced on ACTUAL output token counts,
        # not inflated pre-compression counts.
        if compress:
            compressed_strings = compressor.compress(final_chunks, model=model)
            for c, cs in zip(final_chunks, compressed_strings):
                c.content = cs
                c.tokens = count_tokens(cs)  # re-count post-compression

        # ── 7. Memory retrieval ──────────────────────────────────────────────
        memory_entries = mem_store.search(
            query, top_k=memory_top_k, min_sim=min_memory_sim)
        logger.info(f"  {len(memory_entries)} memory entries injected")

        # ── 8. Assemble once without trimming for before/after telemetry ─────
        before_prompt = assembler.assemble(
            base_prompt=base_prompt,
            chunks=final_chunks,
            memory_entries=memory_entries,
            query=query,
            output_mode=output_mode,
            model=model,
        )
        before_tokens = count_tokens(before_prompt)

        # ── 9. Enforce global prompt budget on the actual assembled output ───
        result, selected_chunks, selected_memory, status = _fit_prompt_to_limit(
            base_prompt=base_prompt,
            chunks=final_chunks,
            memory_entries=memory_entries,
            query=query,
            output_mode=output_mode,
            model=model,
            target_tokens=target_tokens,
            before_tokens=before_tokens,
            include_report=include_report,
            base_prompt_policy=base_prompt_policy,
        )
        after_tokens = count_tokens(result)
        saved_tokens = max(before_tokens - after_tokens, 0)
        saved_percent = (
            round(saved_tokens * 100 / before_tokens, 1)
            if before_tokens > 0 else 0.0
        )
        logger.info(
            "  prompt_optimizer | "
            f"status={status} before={before_tokens:,} after={after_tokens:,} "
            f"saved={saved_tokens:,} saved_percent={saved_percent}% "
            f"target={target_tokens:,} chunks={len(selected_chunks)}/{len(final_chunks)} "
            f"memory={len(selected_memory)}/{len(memory_entries)}"
        )
        return result

    except Exception as e:
        logger.error(f"  ❌ {e}", exc_info=True)
        return f"Error in retrieve_context: {e}"


# ── Memory tools ────────────────────────────────────────────────────────

@mcp.tool()
def memory_save(
    key: str,
    value: str,
    mtype: str = "semantic",
    tags: str = "",
) -> str:
    """
    Save knowledge to long-term memory.

    :param key:   Short identifying key (used for search).
    :param value: Content to remember.
    :param mtype: Memory tier — "episodic" | "semantic" | "procedural"
    :param tags:  Comma-separated labels for filtering.
    """
    try:
        value = sanitize(value, escape_xml=False)
        store = {
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural}.get(
            mtype,
            semantic)
        result = store.save(key, value, tags)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_search(
    query: str,
    mtype: str = "",
    top_k: int = 5,
    min_sim: float = 0.12,
) -> str:
    """
    Search long-term memory by semantic similarity.

    :param query:   Search query.
    :param mtype:   Filter by tier: "episodic" | "semantic" | "procedural" | "" (all)
    :param top_k:   Max results.
    :param min_sim: Minimum similarity threshold (0–1).
    """
    try:
        from memory.episodic import MemoryType
        mt = None
        if mtype:
            try:
                mt = MemoryType(mtype)
            except ValueError:
                pass
        results = mem_store.search(
            query, mtype=mt, top_k=top_k, min_sim=min_sim)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def memory_inject(
    base_prompt: str,
    query: str,
    top_k: int = 4,
    min_sim: float = 0.18,
    output_mode: str = "concise",
    model: str = "gemini-3.8-flash",
    target_ratio: float = 0.20,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    include_report: bool = DEFAULT_INCLUDE_REPORT,
    base_prompt_policy: str = "reject",
) -> str:
    """Inject relevant memory entries into base_prompt within token budget (without file retrieval).
    output_mode: 'concise' | 'structured' | 'code_only' | 'minimal'. base_prompt_policy: 'reject' | 'truncate'."""
    try:
        entries = mem_store.search(query, top_k=top_k, min_sim=min_sim)
        budget = get_budget(model)
        target_tokens = _prompt_token_limit(
            context_window=budget.context_window,
            inject_budget=budget.inject_budget,
            target_ratio=target_ratio,
            max_prompt_tokens=max_prompt_tokens,
        )
        before_prompt = assembler.assemble(
            base_prompt=base_prompt, chunks=[], memory_entries=entries,
            query=query, output_mode=output_mode, model=model,
        )
        result, _, _, status = _fit_prompt_to_limit(
            base_prompt=base_prompt,
            chunks=[],
            memory_entries=entries,
            query=query,
            output_mode=output_mode,
            model=model,
            target_tokens=target_tokens,
            before_tokens=count_tokens(before_prompt),
            include_report=include_report,
            base_prompt_policy=base_prompt_policy,
        )
        logger.info(
            "  memory_inject optimizer | "
            f"status={status} after={count_tokens(result):,} "
            f"target={target_tokens:,}"
        )
        return result
    except Exception as e:
        return f"Error: {e}"




@mcp.tool()
def estimate_tokens(text: str) -> str:
    """
    Estimate token count for any text.
    Useful for checking prompt size before sending to model.
    """
    tok = count_tokens(text)
    chars = len(text)
    # Representative models across active providers
    _benchmark_models = [
        # Gemini
        "gemini-3.8-flash",
        "gemini-3-flash",
        "gemini-3-pro",
        "gemini-2.5-pro",
        # Claude
        "claude-sonnet-4",
        "claude-opus-4",
        # OpenAI GPT-5 & Chat
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4o-mini",
        # OpenAI Reasoning
        "o3",
        "o1",
        "o4-mini",
    ]
    budgets = {
        m: get_budget(m).inject_budget
        for m in _benchmark_models
    }
    return json.dumps({
        "tokens": tok,
        "chars": chars,
        "ratio_chars_per_tok": round(chars / tok, 2) if tok > 0 else 0,
        "fits_in": {
            m: tok <= b
            for m, b in budgets.items()
        },
    }, ensure_ascii=False)




@mcp.tool()
def system_port_status(port: int) -> str:
    """
    Check network port status and owning process on Windows without opening PowerShell.
    :param port: TCP local port number (e.g. 8080, 3000).
    """
    try:
        import subprocess

        if not 1 <= port <= 65535:
            return json.dumps({"port": port, "error": "port out of range 1-65535"}, ensure_ascii=False)

        pid = None
        pname = None
        is_listening = False

        # Parse netstat directly (passive check, supports IPv4 + IPv6 without TCP connect)
        res = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
        )
        port_str = str(port)
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if "LISTENING" not in line_str:
                continue
            parts = line_str.split()
            if len(parts) >= 5:
                local_addr = parts[1]
                if ":" in local_addr and local_addr.rsplit(":", 1)[-1] == port_str:
                    is_listening = True
                    if parts[-1].isdigit():
                        pid = int(parts[-1])
                    break

        if is_listening and pid:
            p_res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if p_res.stdout.strip():
                pname = p_res.stdout.strip().split(",")[0].strip('"')

        return json.dumps({
            "port": port,
            "listening": is_listening,
            "pid": pid,
            "process_name": pname,
            "status": "BUSY" if is_listening else "FREE",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"port": port, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def system_port_kill(port: int) -> str:
    """
    Terminate the process listening on a specified TCP port on Windows.
    :param port: TCP local port number to free (e.g. 8080, 3000).
    """
    try:
        import subprocess
        if not 1 <= port <= 65535:
            return json.dumps({"port": port, "error": "port out of range 1-65535", "success": False})
        status_raw = system_port_status(port)
        status_data = json.loads(status_raw)
        pid = status_data.get("pid")
        if not pid:
            return json.dumps({
                "port": port,
                "status": "ALREADY_FREE",
                "success": True,
                "message": f"Port {port} is not in use.",
            }, ensure_ascii=False)

        pname = status_data.get("process_name") or "unknown"
        protected = {"system", "svchost.exe", "lsass.exe", "csrss.exe", "wininit.exe", "services.exe"}
        if pid in (0, 4, os.getpid()) or pname.lower() in protected:
            return json.dumps({
                "port": port,
                "status": "REFUSED",
                "success": False,
                "reason": f"Protected process {pname} (PID {pid}) will not be killed.",
            }, ensure_ascii=False)
        res = subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
        )
        killed = (res.returncode == 0)
        return json.dumps({
            "port": port,
            "killed_pid": pid,
            "process_name": pname,
            "success": killed,
            "output": res.stdout.strip() if killed else res.stderr.strip(),
            "status": "FREED" if killed else "FAILED",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"port": port, "error": str(e), "success": False}, ensure_ascii=False)



@mcp.tool()
def system_env_status() -> dict:
    """
    Get current environment status: git branch/dirty, active ports, python version, encoding, free disk.
    """
    import subprocess
    import shutil

    workspace = next(
        (p for p in (_ROOT, *_ROOT.parents) if (p / ".git").exists()),
        _ROOT,
    )

    git_branch = "unknown"
    git_dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(workspace),
        )
        if res.returncode == 0:
            git_branch = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(workspace),
        )
        if res.returncode == 0:
            git_dirty = bool(res.stdout.strip())
    except Exception:
        pass

    active_ports: list[int] = []
    try:
        res = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            found_ports = set()
            for line in res.stdout.splitlines():
                line_str = line.strip()
                if "LISTENING" in line_str:
                    parts = line_str.split()
                    if len(parts) >= 2:
                        local_addr = parts[1]
                        if ":" in local_addr:
                            port_str = local_addr.rsplit(":", 1)[-1]
                            if port_str.isdigit():
                                found_ports.add(int(port_str))
            active_ports = sorted(list(found_ports))
    except Exception:
        pass

    free_disk_gb = 0.0
    try:
        _, _, free = shutil.disk_usage(str(workspace))
        free_disk_gb = round(free / (1024 ** 3), 2)
    except Exception:
        pass

    return {
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "active_ports": active_ports,
        "python_version": sys.version.split()[0],
        "encoding": sys.getdefaultencoding(),
        "free_disk_gb": free_disk_gb,
    }


@mcp.tool()
def sqlite_db_backup(db_path: str) -> dict:
    """
    Create a backup of a SQLite database using native sqlite3.backup().
    :param db_path: Path to the SQLite database file.
    """
    import sqlite3
    try:
        resolved_path = os.path.abspath(db_path)
        if not os.path.exists(resolved_path):
            return {
                "success": False,
                "backup_path": "",
                "size_bytes": 0,
                "error": f"Database file not found: {db_path}",
            }

        ts = int(time.time())
        if resolved_path.endswith(".db"):
            backup_path = f"{resolved_path}.bak.{ts}"
        else:
            backup_path = f"{resolved_path}.db.bak.{ts}"

        src = sqlite3.connect(f"{Path(resolved_path).as_uri()}?mode=ro", uri=True, timeout=10.0)
        try:
            dst = sqlite3.connect(backup_path)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

        size_bytes = os.path.getsize(backup_path)
        return {
            "success": True,
            "backup_path": backup_path,
            "size_bytes": size_bytes,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "backup_path": "",
            "size_bytes": 0,
            "error": str(e),
        }


@mcp.tool()
def invalidate_cache() -> str:
    """Clear the retrieval cache. Call when workspace files have changed."""
    cache.invalidate_all()
    return json.dumps({"status": "cache cleared"})


@mcp.tool()
def reindex_paths(paths: list[str]) -> str:
    """
    Force re-chunking and re-indexing of given paths.
    Use after significant file edits to refresh the vector index.
    """
    logger.info(f"[Server] force reindex: {paths}")
    try:
        chunks = _ensure_indexed(paths, force_reindex=True)
        return json.dumps({"status": "reindexed", "total_chunks": len(chunks)})
    except Exception as e:
        return f"Error: {e}"


# ── Entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("═" * 70)
    logger.info("Harness context engineering v5.0 starting")
    logger.info(f"  Log (primary) : {LOG_FILE}")
    logger.info(f'  Get-Content "{LOG_FILE}" -Wait')
    logger.info("  Tools  : retrieve_context · estimate_tokens")
    logger.info("           memory_save · memory_search · memory_inject")
    logger.info("           invalidate_cache · reindex_paths")
    logger.info("           system_port_status · system_port_kill")
    logger.info("           system_env_status · sqlite_db_backup")
    logger.info(
        "  Models : Gemini · Claude · GPT-4o/4.1/5 · o1/o3/o4")
    logger.info(
        "  Stack  : numpy vector store + BM25 + AST chunker")
    logger.info(
        "           local reranker + dependency graph + safety layer")
    logger.info("═" * 70)
    mcp.run()
