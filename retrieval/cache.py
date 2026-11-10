"""TTL cache for immutable retrieval selections."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from chunking.treesitter_chunker import ASTChunker


class RetrievalCache:
    def __init__(self, ttl: int = 600):
        self.ttl = ttl
        self.cache: dict[str, tuple[list, float]] = {}

    @staticmethod
    def path_fingerprint(path: str) -> str:
        """Fingerprint a requested source path, pruning ignored dirs/extensions."""
        source = Path(path)
        digest = hashlib.sha256()

        if not source.exists():
            return "missing"

        files: list[Path] = []
        if source.is_file():
            files.append(source)
        else:
            skip_dirs = ASTChunker._SKIP_DIRS
            skip_exts = ASTChunker._SKIP_EXTS
            for root, dirs, filenames in os.walk(source):
                dirs[:] = sorted(
                    d for d in dirs if d not in skip_dirs and not d.startswith(".git")
                )
                for fname in sorted(filenames):
                    if os.path.splitext(fname)[1].lower() in skip_exts:
                        continue
                    files.append(Path(root) / fname)

        for file_path in files:
            try:
                stat = file_path.stat()
            except OSError:
                continue
            digest.update(os.path.normcase(str(file_path.resolve())).encode())
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
        return digest.hexdigest()

    def _hash(
        self,
        query: str,
        paths: list[str],
        options: dict | None = None,
        fingerprints: dict[str, str] | None = None,
    ) -> str:
        fp_map = fingerprints or {}
        payload = {
            "query": query,
            "paths": sorted(
                (path, fp_map[path] if path in fp_map else self.path_fingerprint(path))
                for path in paths
            ),
            "options": options or {},
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def get(
        self,
        query: str,
        paths: list[str],
        options: dict | None = None,
        fingerprints: dict[str, str] | None = None,
    ) -> list | None:
        key = self._hash(query, paths, options, fingerprints=fingerprints)
        entry = self.cache.get(key)
        if entry is None:
            return None

        data, timestamp = entry
        if time.time() - timestamp >= self.ttl:
            del self.cache[key]
            return None
        return deepcopy(data)

    def set(
        self,
        query: str,
        paths: list[str],
        chunks: list,
        options: dict | None = None,
        fingerprints: dict[str, str] | None = None,
    ) -> None:
        key = self._hash(query, paths, options, fingerprints=fingerprints)
        self.cache[key] = (deepcopy(chunks), time.time())

    def invalidate_all(self) -> None:
        self.cache.clear()
