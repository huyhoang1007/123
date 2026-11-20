"""
bench_retrieve.py - Measure indexing, retrieval latency, chunk counts, and token savings
for harness-context `retrieve_context` across target paths and output modes.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_REPORT_RE = re.compile(
    r"<!--\s*prompt_optimizer\s+status=(?P<status>\S+)\s+before=(?P<before>\d+)\s+after=(?P<after>\d+)\s+"
    r"saved=(?P<saved>\d+)\s+saved_percent=(?P<saved_percent>[\d.]+)%\s+target=(?P<target>\d+)\s+"
    r"chunks=(?P<chunks_after>\d+)/(?P<chunks_before>\d+)\s+memory=(?P<mem_after>\d+)/(?P<mem_before>\d+)\s*-->"
)


def _raw_files_tokens(paths: list[str]) -> tuple[int, int]:
    from chunking.treesitter_chunker import count_tokens

    total_files = 0
    total_tokens = 0
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
    for raw in paths:
        p = pathlib.Path(raw).expanduser().resolve()
        if p.is_file():
            total_files += 1
            try:
                total_tokens += count_tokens(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        elif p.is_dir():
            for f in p.rglob("*"):
                if not f.is_file() or any(part in skip_dirs for part in f.parts):
                    continue
                if f.suffix.lower() in (".py", ".ps1", ".md", ".json", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go"):
                    total_files += 1
                    try:
                        total_tokens += count_tokens(f.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        pass
    return total_files, total_tokens


def run_benchmark(
    *,
    paths: list[str],
    query: str,
    model: str = "gemini-2.5-flash",
    modes: tuple[str, ...] = ("concise", "structured"),
    base_prompt: str = "Follow project conventions.",
    max_prompt_tokens: int = 0,
    isolated: bool = True,
) -> dict:
    """Run retrieve_context across requested output modes and return token/latency metrics."""
    prev_log_dir = os.environ.get("MCP_HARNESS_LOG_DIR")
    prev_mem_db = os.environ.get("MCP_HARNESS_MEMORY_DB")
    tmp_ctx = tempfile.TemporaryDirectory(ignore_cleanup_errors=True) if isolated else None
    prev_server_mem = None
    prev_singleton = None
    bench_mem_store = None
    bench_db = ""

    try:
        if tmp_ctx is not None:
            os.environ["MCP_HARNESS_LOG_DIR"] = tmp_ctx.name
            bench_db = str(pathlib.Path(tmp_ctx.name) / "bench_mem.sqlite3")
            os.environ["MCP_HARNESS_MEMORY_DB"] = bench_db

        import logging
        import server
        from chunking.treesitter_chunker import count_tokens
        from memory.episodic import MemoryStore

        if tmp_ctx is not None:
            prev_server_mem = server.mem_store
            prev_singleton = MemoryStore._instance
            if prev_server_mem is not None and os.path.abspath(prev_server_mem.db_path) == os.path.abspath(bench_db):
                bench_mem_store = prev_server_mem
            else:
                bench_mem_store = MemoryStore(db_path=bench_db, legacy_db_path="")
                MemoryStore._instance = bench_mem_store
                server.mem_store = bench_mem_store

        raw_files, raw_tokens = _raw_files_tokens(paths)
        results: list[dict] = []

        for idx, mode in enumerate(modes):
            t0 = time.perf_counter()
            out = server.retrieve_context(
                base_prompt=base_prompt,
                paths=paths,
                query=query,
                model=model,
                output_mode=mode,
                force_reindex=(idx == 0),
                max_prompt_tokens=max_prompt_tokens,
                include_report=True,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            out_tokens = count_tokens(out)
            m = _REPORT_RE.search(out)
            parsed = {
                "status": m.group("status") if m else "unknown",
                "before_tokens": int(m.group("before")) if m else out_tokens,
                "after_tokens": int(m.group("after")) if m else out_tokens,
                "saved_percent": float(m.group("saved_percent")) if m else 0.0,
                "chunks_kept": f"{m.group('chunks_after')}/{m.group('chunks_before')}" if m else "n/a",
            }
            vs_raw_pct = round(max(raw_tokens - out_tokens, 0) * 100.0 / raw_tokens, 1) if raw_tokens > 0 else 0.0
            results.append({
                "output_mode": mode,
                "elapsed_ms": elapsed_ms,
                "output_chars": len(out),
                "output_tokens": out_tokens,
                "vs_raw_corpus_saved_percent": vs_raw_pct,
                **parsed,
            })

        return {
            "model": model,
            "query": query,
            "corpus_files": raw_files,
            "corpus_raw_tokens": raw_tokens,
            "modes": results,
        }
    finally:
        if tmp_ctx is not None:
            try:
                if bench_mem_store is not None:
                    bench_mem_store._conn.close()
                import server as _srv
                from memory.episodic import MemoryStore as _MS
                _srv.mem_store = prev_server_mem
                _MS._instance = prev_singleton
                tmp_prefix = os.path.abspath(tmp_ctx.name).lower()
                for lg in (logging.getLogger(), logging.getLogger("harness.server")):
                    for h in list(lg.handlers):
                        base = getattr(h, "baseFilename", "")
                        if base and os.path.abspath(base).lower().startswith(tmp_prefix):
                            h.close()
                            lg.removeHandler(h)
            except Exception:
                pass
            if prev_log_dir is None:
                os.environ.pop("MCP_HARNESS_LOG_DIR", None)
            else:
                os.environ["MCP_HARNESS_LOG_DIR"] = prev_log_dir
            if prev_mem_db is None:
                os.environ.pop("MCP_HARNESS_MEMORY_DB", None)
            else:
                os.environ["MCP_HARNESS_MEMORY_DB"] = prev_mem_db
            tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark harness-context retrieve_context token savings & latency.")
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--modes", nargs="+", default=["concise", "structured"])
    parser.add_argument("--max-prompt-tokens", type=int, default=0)
    args = parser.parse_args(argv)

    rep = run_benchmark(
        paths=args.paths,
        query=args.query,
        model=args.model,
        modes=tuple(args.modes),
        max_prompt_tokens=args.max_prompt_tokens,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
