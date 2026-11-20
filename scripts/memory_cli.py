"""
memory_cli.py - CLI manager for harness-context SQLite memory tiers (episodic, semantic, procedural).
Read-only operations (`stats`, `export`) open SQLite with ?mode=ro; write operations (`save`, `import`)
create an online `.backup()` snapshot first when the target DB already exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pathlib
import sqlite3
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness_ctl import _default_memory_db, backup_sqlite_db  # noqa: E402
from memory.episodic import MemoryStore, MemoryType  # noqa: E402


def _resolve_db(db_arg: str | None) -> pathlib.Path:
    return pathlib.Path(db_arg).expanduser().resolve() if db_arg else _default_memory_db().expanduser().resolve()


def _open_readonly(db_path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def stats_readonly(db_path: pathlib.Path) -> dict:
    """Return tier counts, estimated tokens, hits, and top tags without modifying the DB."""
    if not db_path.is_file():
        return {"exists": False, "path": str(db_path), "total": 0, "tiers": {}, "top_tags": []}

    conn = _open_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT type, COUNT(*) AS count, COALESCE(SUM(hits), 0) AS hits,
                   COALESCE(SUM((LENGTH(key) + LENGTH(value) + LENGTH(tags)) / 4), 0) AS tokens
            FROM memories GROUP BY type
            """
        ).fetchall()
        tag_rows = conn.execute("SELECT tags FROM memories WHERE tags != ''").fetchall()
    finally:
        conn.close()

    tiers: dict[str, dict[str, int]] = {}
    total = 0
    for r in rows:
        cnt = int(r["count"])
        total += cnt
        tiers[str(r["type"])] = {"count": cnt, "tokens": int(r["tokens"]), "hits": int(r["hits"])}

    tag_counts: Counter[str] = Counter()
    for tr in tag_rows:
        for t in str(tr["tags"]).split(","):
            clean = t.strip()
            if clean:
                tag_counts[clean] += 1

    return {
        "exists": True,
        "path": str(db_path),
        "bytes": db_path.stat().st_size,
        "total": total,
        "tiers": tiers,
        "top_tags": [{"tag": k, "count": v} for k, v in tag_counts.most_common(10)],
    }


def export_readonly(db_path: pathlib.Path, mtype: str | None = None) -> list[dict]:
    """Export memory entries in read-only mode."""
    if not db_path.is_file():
        return []
    conn = _open_readonly(db_path)
    try:
        if mtype:
            rows = conn.execute(
                "SELECT type, key, value, tags, created_at, updated_at, hits FROM memories WHERE type = ? ORDER BY updated_at DESC",
                (mtype,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT type, key, value, tags, created_at, updated_at, hits FROM memories ORDER BY type, updated_at DESC"
            ).fetchall()
    finally:
        conn.close()

    return [
        {
            "type": str(r["type"]),
            "key": str(r["key"]),
            "value": str(r["value"]),
            "tags": str(r["tags"]),
            "created_at": float(r["created_at"]),
            "updated_at": float(r["updated_at"]),
            "hits": int(r["hits"]),
        }
        for r in rows
    ]


def save_entry(
    db_path: pathlib.Path,
    *,
    mtype: str,
    key: str,
    value: str,
    tags: str = "",
    no_backup: bool = False,
) -> dict:
    """Save or upsert one memory entry, backing up an existing DB first unless disabled."""
    backup_info = None
    if db_path.is_file() and not no_backup:
        backup_info = backup_sqlite_db(db_path)
    store = MemoryStore(db_path=str(db_path), legacy_db_path="")
    try:
        entry = store.save(MemoryType(mtype), key, value, tags)
    finally:
        store._conn.close()
    return {"saved": entry, "backup": backup_info}


def import_entries(db_path: pathlib.Path, entries: list[dict], *, no_backup: bool = False) -> dict:
    """Bulk upsert entries from JSON list into MemoryStore after an online backup."""
    backup_info = None
    if db_path.is_file() and not no_backup:
        backup_info = backup_sqlite_db(db_path)
    imported = 0
    store = MemoryStore(db_path=str(db_path), legacy_db_path="")
    try:
        for item in entries:
            mtype = MemoryType(str(item["type"]))
            key = str(item["key"])
            value = str(item["value"])
            tags = str(item.get("tags") or "")
            store.save(mtype, key, value, tags)
            imported += 1
    finally:
        store._conn.close()
    return {"imported": imported, "backup": backup_info}


def search_entries(
    db_path: pathlib.Path,
    *,
    query: str,
    mtype: str | None = None,
    top_k: int = 5,
    min_sim: float = 0.12,
) -> list[dict]:
    """Search MemoryStore using local TF-IDF/LSA + lexical ranking."""
    if not db_path.is_file():
        return []
    store = MemoryStore(db_path=str(db_path), legacy_db_path="")
    try:
        mt = MemoryType(mtype) if mtype else None
        return store.search(query=query, mtype=mt, top_k=top_k, min_sim=min_sim)
    finally:
        store._conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI manager for harness-context SQLite memory tiers.")
    parser.add_argument("--db", default=None, help="Path to memory.sqlite3 (defaults to ~/.mcp-harness/memory.sqlite3).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Show read-only tier counts, tokens, hits, and top tags.")

    p_exp = sub.add_parser("export", help="Export memory entries to JSON (read-only).")
    p_exp.add_argument("--mtype", choices=("episodic", "semantic", "procedural"), default=None)
    p_exp.add_argument("--out", default="", help="Optional output JSON file path (prints to stdout if omitted).")

    p_imp = sub.add_parser("import", help="Import memory entries from a JSON file (backs up DB first).")
    p_imp.add_argument("--file", required=True)
    p_imp.add_argument("--no-backup", action="store_true")

    p_save = sub.add_parser("save", help="Save or update a memory entry.")
    p_save.add_argument("--mtype", required=True, choices=("episodic", "semantic", "procedural"))
    p_save.add_argument("--key", required=True)
    p_save.add_argument("--value", required=True)
    p_save.add_argument("--tags", default="")
    p_save.add_argument("--no-backup", action="store_true")

    p_srch = sub.add_parser("search", help="Search memory entries by query.")
    p_srch.add_argument("--query", required=True)
    p_srch.add_argument("--mtype", choices=("episodic", "semantic", "procedural"), default=None)
    p_srch.add_argument("--top-k", type=int, default=5)
    p_srch.add_argument("--min-sim", type=float, default=0.12)

    p_inj = sub.add_parser("inject", help="Preview formatted memory injection block for a base prompt.")
    p_inj.add_argument("--query", required=True)
    p_inj.add_argument("--base-prompt", default="Follow project conventions.")
    p_inj.add_argument("--top-k", type=int, default=4)
    p_inj.add_argument("--min-sim", type=float, default=0.18)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = _resolve_db(args.db)

    if args.cmd == "stats":
        print(json.dumps(stats_readonly(db_path), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "export":
        data = export_readonly(db_path, mtype=args.mtype)
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        if args.out:
            out_path = pathlib.Path(args.out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload + "\n", encoding="utf-8")
            print(json.dumps({"exported": len(data), "out": str(out_path)}))
        else:
            print(payload)
        return 0
    if args.cmd == "import":
        raw = json.loads(pathlib.Path(args.file).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Import JSON must be a list of memory objects.")
        print(json.dumps(import_entries(db_path, raw, no_backup=args.no_backup), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "save":
        res = save_entry(
            db_path,
            mtype=args.mtype,
            key=args.key,
            value=args.value,
            tags=args.tags,
            no_backup=args.no_backup,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "search":
        hits = search_entries(
            db_path,
            query=args.query,
            mtype=args.mtype,
            top_k=args.top_k,
            min_sim=args.min_sim,
        )
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "inject":
        hits = search_entries(db_path, query=args.query, top_k=args.top_k, min_sim=args.min_sim)
        mem_block = "\n".join(f"- [{h['type']}] {h['key']}: {h['value']}" for h in hits)
        out = args.base_prompt if not mem_block else f"{args.base_prompt}\n\n[Memories]\n{mem_block}"
        print(out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
