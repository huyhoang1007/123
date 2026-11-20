"""
harness_ctl.py - Maintenance, upgrade, diagnostics, and CLI helper for harness-context.
Runs cleanly on Windows PowerShell 5.1 with UTF-8 I/O and zero .pyc clutter (-B).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import pathlib
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_TOOLS = (
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
)

DEPS = (
    ("mcp", "mcp.server.fastmcp"),
    ("numpy", "numpy"),
    ("rank_bm25", "rank_bm25"),
    ("scikit-learn", "sklearn"),
    ("tiktoken", "tiktoken"),
    ("tree-sitter", "tree_sitter"),
)


def _default_harness_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("MCP_HARNESS_HOME") or (pathlib.Path.home() / ".mcp-harness"))


def _default_memory_db() -> pathlib.Path:
    override = os.environ.get("MCP_HARNESS_MEMORY_DB")
    return pathlib.Path(override) if override else (_default_harness_dir() / "memory.sqlite3")


def check_syntax(repo_root: pathlib.Path = REPO_ROOT) -> list[str]:
    """Parse all Python files under repo_root with ast.parse; return list of error strings."""
    errors: list[str] = []
    for py_file in sorted(repo_root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        try:
            ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as exc:
            errors.append(f"{py_file.relative_to(repo_root)}: {exc}")
    return errors


def _run_git(args: list[str], repo_root: pathlib.Path = REPO_ROOT, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def backup_sqlite_db(db_path: pathlib.Path, out_dir: pathlib.Path | None = None) -> dict:
    """Create an online SQLite backup using read-only source URI and sqlite3.backup()."""
    resolved = db_path.expanduser().resolve()
    if not resolved.is_file():
        return {"success": False, "error": f"Database not found: {resolved}", "backup_path": ""}

    ts = int(time.time())
    target_dir = out_dir.expanduser().resolve() if out_dir else resolved.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    backup_path = target_dir / f"{resolved.name}.bak.{ts}"
    counter = 1
    while backup_path.exists():
        backup_path = target_dir / f"{resolved.name}.bak.{ts}.{counter}"
        counter += 1

    src = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=10.0)
    try:
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    return {"success": True, "backup_path": str(backup_path), "size_bytes": backup_path.stat().st_size}


def clean_pycache(repo_root: pathlib.Path = REPO_ROOT) -> int:
    """Remove __pycache__ directories inside repo_root and return count removed."""
    removed = 0
    for d in list(repo_root.rglob("__pycache__")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def probe_stdio(server_path: pathlib.Path | None = None, timeout_s: float = 10.0) -> dict:
    """Spawn server.py over stdio with -B and UTF-8, verify initialize + 11 tools."""
    target = (server_path or (REPO_ROOT / "server.py")).resolve()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-B", str(target)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    line_q: "queue.Queue[str | None]" = queue.Queue()

    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line_q.put(line)
        finally:
            line_q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    def _read_json() -> dict:
        try:
            raw = line_q.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for MCP stdio response") from exc
        if raw is None:
            raise RuntimeError("MCP stdio closed unexpectedly")
        return json.loads(raw)

    try:
        assert proc.stdin is not None
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "harness_ctl", "version": "1.0"}},
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        _read_json()

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()
        tools_res = _read_json()
        tools = [t["name"] for t in tools_res.get("result", {}).get("tools", [])]
        missing = [name for name in REQUIRED_TOOLS if name not in tools]
        return {"ok": len(missing) == 0, "tool_count": len(tools), "missing": missing, "tools": tools}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def run_tests(repo_root: pathlib.Path = REPO_ROOT, pattern: str = "test_*.py") -> int:
    """Run harness-context unit/E2E tests with -B and PYTHONUTF8=1."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(repo_root / "tests"), "-p", pattern, "-v"],
        env=env,
        check=False,
    )
    return res.returncode


def gather_doctor_report(repo_root: pathlib.Path = REPO_ROOT) -> dict:
    """Collect read-only diagnostics on dependencies, syntax, git state, storage, and registration."""
    deps = {label: (importlib.util.find_spec(mod) is not None) for label, mod in DEPS}
    syntax_errors = check_syntax(repo_root)

    git_info: dict[str, object] = {"is_repo": (repo_root / ".git").exists()}
    if git_info["is_repo"]:
        br = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        st = _run_git(["status", "--porcelain"], repo_root)
        ab = _run_git(["rev-list", "--left-right", "--count", "HEAD...@{u}"], repo_root)
        git_info["branch"] = br.stdout.strip() if br.returncode == 0 else "unknown"
        dirty_lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
        git_info["dirty_files"] = len(dirty_lines)
        if ab.returncode == 0 and len(ab.stdout.split()) == 2:
            ahead, behind = ab.stdout.split()
            git_info["ahead"] = int(ahead)
            git_info["behind"] = int(behind)

    mem_db = _default_memory_db()
    log_file = _default_harness_dir() / "harness-v3.log"
    mcp_cfg = pathlib.Path.home() / ".gemini" / "config" / "mcp_config.json"
    registered = False
    if mcp_cfg.is_file():
        try:
            cfg = json.loads(mcp_cfg.read_text(encoding="utf-8"))
            srv = (cfg.get("mcpServers") or {}).get("harness-context-optimizer")
            registered = bool(srv and not srv.get("disabled"))
        except Exception:
            registered = False

    return {
        "python": sys.version.split()[0],
        "repo_root": str(repo_root),
        "dependencies": deps,
        "syntax_errors": syntax_errors,
        "git": git_info,
        "memory_db": {"path": str(mem_db), "exists": mem_db.is_file(), "bytes": mem_db.stat().st_size if mem_db.is_file() else 0},
        "log_file": {"path": str(log_file), "exists": log_file.is_file(), "bytes": log_file.stat().st_size if log_file.is_file() else 0},
        "registered_in_mcp_config": registered,
    }


def cmd_update(args: argparse.Namespace) -> int:
    """Safely update harness-context: syntax -> DB backup -> git pull --ff-only -> tests."""
    syntax_errors = check_syntax(REPO_ROOT)
    if syntax_errors:
        print(json.dumps({"status": "aborted", "stage": "pre_syntax", "errors": syntax_errors}, indent=2))
        return 1

    mem_db = _default_memory_db()
    backup_info = backup_sqlite_db(mem_db) if mem_db.is_file() else {"skipped": "memory.sqlite3 not found"}

    st = _run_git(["status", "--porcelain"], REPO_ROOT)
    if st.stdout.strip():
        print(json.dumps({
            "status": "blocked_dirty_worktree",
            "reason": "Working tree has uncommitted changes; commit or stash before pulling updates.",
            "dirty": st.stdout.splitlines(),
            "backup": backup_info,
        }, indent=2))
        return 2

    if not args.no_fetch:
        fetch = _run_git(["fetch", "--prune"], REPO_ROOT, timeout=30.0)
        if fetch.returncode != 0:
            print(json.dumps({"status": "fetch_failed", "stderr": fetch.stderr.strip(), "backup": backup_info}, indent=2))
            return 1

    pull = _run_git(["pull", "--ff-only"], REPO_ROOT, timeout=30.0)
    if pull.returncode != 0:
        print(json.dumps({
            "status": "pull_ff_only_failed",
            "stdout": pull.stdout.strip(),
            "stderr": pull.stderr.strip(),
            "backup": backup_info,
        }, indent=2))
        return 1

    post_syntax = check_syntax(REPO_ROOT)
    if post_syntax:
        print(json.dumps({"status": "failed", "stage": "post_syntax", "errors": post_syntax, "backup": backup_info}, indent=2))
        return 1

    test_rc = 0 if args.skip_tests else run_tests(REPO_ROOT)
    cleaned = clean_pycache(REPO_ROOT) if args.clean_pycache else 0
    print(json.dumps({
        "status": "ok" if test_rc == 0 else "tests_failed",
        "git_pull": pull.stdout.strip(),
        "backup": backup_info,
        "tests_returncode": test_rc,
        "pycache_cleaned": cleaned,
    }, indent=2))
    return test_rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintenance, update, and CLI helper for harness-context.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check dependencies, syntax, git state, storage, and MCP registration.")

    p_test = sub.add_parser("test", help="Run harness-context unit and E2E tests.")
    p_test.add_argument("--pattern", default="test_*.py")

    p_bak = sub.add_parser("backup", help="Safely backup memory.sqlite3 via sqlite3.backup().")
    p_bak.add_argument("--db", default=str(_default_memory_db()))
    p_bak.add_argument("--out-dir", default=None)

    p_probe = sub.add_parser("probe", help="Verify FastMCP stdio initialize + tools/list.")
    p_probe.add_argument("--timeout", type=float, default=10.0)

    p_upd = sub.add_parser("update", help="Backup DB, git pull --ff-only if clean, and run tests.")
    p_upd.add_argument("--no-fetch", action="store_true")
    p_upd.add_argument("--skip-tests", action="store_true")
    p_upd.add_argument("--clean-pycache", action="store_true")

    p_clean = sub.add_parser("clean-pycache", help="Remove __pycache__ directories inside harness-context.")

    p_est = sub.add_parser("estimate", help="Estimate token count and model budget.")
    p_est.add_argument("--text", default="")
    p_est.add_argument("--file", default="")
    p_est.add_argument("--model", default="gemini-2.5-flash")

    p_ret = sub.add_parser("retrieve", help="Run retrieve_context directly from CLI.")
    p_ret.add_argument("--query", required=True)
    p_ret.add_argument("--paths", nargs="+", required=True)
    p_ret.add_argument("--base-prompt", default="Follow project conventions.")
    p_ret.add_argument("--model", default="gemini-2.5-flash")
    p_ret.add_argument("--output-mode", default="concise", choices=("concise", "structured", "code_only", "minimal"))
    p_ret.add_argument("--max-prompt-tokens", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        rep = gather_doctor_report(REPO_ROOT)
        print(json.dumps(rep, indent=2))
        return 0 if not rep["syntax_errors"] else 1
    if args.cmd == "test":
        return run_tests(REPO_ROOT, pattern=args.pattern)
    if args.cmd == "backup":
        out = pathlib.Path(args.out_dir) if args.out_dir else None
        res = backup_sqlite_db(pathlib.Path(args.db), out_dir=out)
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1
    if args.cmd == "probe":
        res = probe_stdio(timeout_s=args.timeout)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1
    if args.cmd == "update":
        return cmd_update(args)
    if args.cmd == "clean-pycache":
        print(json.dumps({"removed_pycache_dirs": clean_pycache(REPO_ROOT)}))
        return 0
    if args.cmd == "estimate":
        from dataclasses import asdict
        from chunking.treesitter_chunker import count_tokens
        from context.budgeting import get_budget
        txt = pathlib.Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        tok = count_tokens(txt)
        budget = get_budget(args.model)
        print(json.dumps({"model": args.model, "tokens": tok, "budget": asdict(budget)}, indent=2))
        return 0
    if args.cmd == "retrieve":
        import server
        out = server.retrieve_context(
            base_prompt=args.base_prompt,
            paths=args.paths,
            query=args.query,
            model=args.model,
            output_mode=args.output_mode,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        print(out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
