"""
chunking/treesitter_chunker.py — AST-based code chunker
=========================================================
Uses tree-sitter if available, or Python stdlib `ast` + regex fallback
to split files into function/class-level chunks with zero external C-bindings.
// ponytail: stdlib ast + regex chunker fallback; install tree-sitter wheels only if multi-language CST needed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import os
import re
import uuid

# ── Token counting ────────────────────────────────────────────────────────────

try:
    import tiktoken as _tiktoken
    _ENCODER = _tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False
    _ENCODER = None  # type: ignore[assignment]


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or 3.5 chars/token heuristic."""
    if not text:
        return 0
    if _HAS_TIKTOKEN and _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return max(1, int(len(text) / 3.5))


# ── Chunk dataclass ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    path: str
    type: str
    symbol: str
    content: str
    summary: str
    tokens: int
    embedding: list[float] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    priority: int = 5


# ── AST Chunker ───────────────────────────────────────────────────────────────

_JS_TS_DECL_RE = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s+([A-Za-z_]\w*)|class\s+([A-Za-z_]\w*)|"
    r"(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>)",
    re.MULTILINE,
)
_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


class ASTChunker:
    def __init__(self):
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            import tree_sitter_typescript
            import tree_sitter
            self.ts = tree_sitter
            self.lang_py = tree_sitter.Language(tree_sitter_python.language())
            self.lang_js = tree_sitter.Language(tree_sitter_javascript.language())
            self.lang_ts = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        except ImportError:
            self.ts = None

    def get_parser(self, ext: str):
        if not self.ts:
            return None
        parser = self.ts.Parser()
        if ext == ".py":
            parser.language = self.lang_py
        elif ext in [".js", ".jsx"]:
            parser.language = self.lang_js
        elif ext in [".ts", ".tsx"]:
            parser.language = self.lang_ts
        else:
            return None
        return parser

    _SKIP_DIRS = {
        ".git", ".hg", ".svn", ".venv", "venv", "node_modules",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".next",
        "dist", "build", "brain", "antigravity-browser-profile",
        "skills_disabled", "mcp_disabled", "presence", "conversations",
    }
    _SKIP_EXTS = {
        ".db", ".sqlite", ".sqlite3", ".pyc", ".pyo", ".pyd",
        ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".jpeg",
        ".gif", ".webp", ".ico", ".pdf", ".zip", ".tar", ".gz",
        ".mp4", ".mp3", ".wav", ".woff", ".woff2", ".ttf", ".lock",
        ".disabled", ".pb", ".pbtxt",
    }

    def chunk_paths(self, paths: list[str]) -> list[Chunk]:
        chunks = []
        for path in paths:
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    dirs[:] = [
                        d for d in dirs
                        if d not in self._SKIP_DIRS and not d.startswith(".git")
                    ]
                    for file in files:
                        if os.path.splitext(file)[1].lower() in self._SKIP_EXTS:
                            continue
                        filepath = os.path.join(root, file)
                        chunks.extend(self._chunk_file(filepath))
            else:
                if os.path.splitext(path)[1].lower() not in self._SKIP_EXTS:
                    chunks.extend(self._chunk_file(path))
        return chunks

    def _chunk_python_stdlib(self, filepath: str, content: str) -> list[Chunk]:
        """Parse Python files into function/class chunks using stdlib `ast`."""
        try:
            tree = ast.parse(content, filename=filepath)
        except SyntaxError:
            return [self._whole_file_chunk(filepath, content)]

        lines = content.splitlines()
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        chunks: list[Chunk] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = max(0, (node.lineno or 1) - 1)
                if node.decorator_list:
                    start = max(0, (node.decorator_list[0].lineno or 1) - 1)
                end = node.end_lineno or (start + 1)
                snippet = "\n".join(lines[start:end])
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                first_doc = doc[0] if doc else ""
                summary = f"function_definition {node.name} (L{start + 1}-L{end})"
                if first_doc:
                    summary += f": {first_doc[:120]}"
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    path=filepath,
                    type="function_definition",
                    symbol=node.name,
                    content=snippet,
                    summary=summary,
                    tokens=count_tokens(snippet),
                    imports=imports[:20],
                    exports=[node.name],
                    priority=6,
                ))
            elif isinstance(node, ast.ClassDef):
                start = max(0, (node.lineno or 1) - 1)
                if node.decorator_list:
                    start = max(0, (node.decorator_list[0].lineno or 1) - 1)
                end = node.end_lineno or (start + 1)
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                first_doc = doc[0] if doc else ""
                methods = [
                    child for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                # Split oversized classes (>120 lines) into class header + method chunks
                if (end - start) > 120 and methods:
                    first_method_start = max(0, (methods[0].lineno or (start + 2)) - 1)
                    if methods[0].decorator_list:
                        first_method_start = max(0, (methods[0].decorator_list[0].lineno or 1) - 1)
                    header_snippet = "\n".join(lines[start:max(start + 1, first_method_start)])
                    header_summary = f"class_definition {node.name} header (L{start + 1}-L{end})"
                    if first_doc:
                        header_summary += f": {first_doc[:120]}"
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        path=filepath,
                        type="class_definition",
                        symbol=node.name,
                        content=header_snippet,
                        summary=header_summary,
                        tokens=count_tokens(header_snippet),
                        imports=imports[:20],
                        exports=[node.name],
                        priority=7,
                    ))
                    for m in methods:
                        ms = max(0, (m.lineno or 1) - 1)
                        if m.decorator_list:
                            ms = max(0, (m.decorator_list[0].lineno or 1) - 1)
                        me = m.end_lineno or (ms + 1)
                        m_snip = "\n".join(lines[ms:me])
                        m_doc = (ast.get_docstring(m) or "").strip().splitlines()
                        m_first = m_doc[0] if m_doc else ""
                        m_sym = f"{node.name}.{m.name}"
                        m_sum = f"method_definition {m_sym} (L{ms + 1}-L{me})"
                        if m_first:
                            m_sum += f": {m_first[:120]}"
                        chunks.append(Chunk(
                            id=str(uuid.uuid4()),
                            path=filepath,
                            type="method_definition",
                            symbol=m_sym,
                            content=m_snip,
                            summary=m_sum,
                            tokens=count_tokens(m_snip),
                            imports=imports[:20],
                            exports=[m_sym],
                            priority=6,
                        ))
                else:
                    snippet = "\n".join(lines[start:end])
                    summary = f"class_definition {node.name} (L{start + 1}-L{end})"
                    if first_doc:
                        summary += f": {first_doc[:120]}"
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        path=filepath,
                        type="class_definition",
                        symbol=node.name,
                        content=snippet,
                        summary=summary,
                        tokens=count_tokens(snippet),
                        imports=imports[:20],
                        exports=[node.name],
                        priority=7,
                    ))

        if not chunks:
            return [self._whole_file_chunk(filepath, content, imports=imports[:20])]
        return chunks

    def _chunk_regex(self, filepath: str, content: str, ext: str) -> list[Chunk]:
        """Lightweight declaration chunker for JS/TS and Markdown files."""
        pattern = _MD_HEADING_RE if ext == ".md" else _JS_TS_DECL_RE
        matches = list(pattern.finditer(content))
        if len(matches) < 2:
            return [self._whole_file_chunk(filepath, content)]

        chunks: list[Chunk] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            snippet = content[start:end].strip()
            if not snippet:
                continue
            if ext == ".md":
                symbol = match.group(2).strip()[:60]
                kind = "section"
            else:
                symbol = next((g for g in match.groups() if g), "anonymous")
                kind = "declaration"
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                path=filepath,
                type=kind,
                symbol=symbol,
                content=snippet,
                summary=f"{kind} {symbol}",
                tokens=count_tokens(snippet),
                exports=[symbol],
            ))
        return chunks or [self._whole_file_chunk(filepath, content)]

    @staticmethod
    def _whole_file_chunk(
        filepath: str,
        content: str,
        imports: list[str] | None = None,
    ) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            path=filepath,
            type="file",
            symbol="module",
            content=content,
            summary=os.path.basename(filepath),
            tokens=count_tokens(content),
            imports=imports or [],
        )

    def _chunk_file(self, filepath: str) -> list[Chunk]:
        ext = os.path.splitext(filepath)[1].lower()
        try:
            with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        parser = self.get_parser(ext)
        if parser is None:
            if ext == ".py":
                return self._chunk_python_stdlib(filepath, content)
            if ext in (".js", ".jsx", ".ts", ".tsx", ".md"):
                return self._chunk_regex(filepath, content, ext)
            return [self._whole_file_chunk(filepath, content)]

        chunks: list[Chunk] = []
        tree = parser.parse(bytes(content, "utf8"))
        root_node = tree.root_node
        content_bytes = bytes(content, "utf8")

        def traverse(node):
            if node.type in [
                "class_definition",
                "function_definition",
                "class_declaration",
                "function_declaration",
                "method_definition",
            ]:
                chunk_content = content_bytes[node.start_byte:node.end_byte].decode("utf-8")
                symbol_name = "unknown"
                for child in node.children:
                    if child.type == "identifier":
                        symbol_name = content_bytes[child.start_byte:child.end_byte].decode("utf-8")
                        break
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    path=filepath,
                    type=node.type,
                    symbol=symbol_name,
                    content=chunk_content,
                    summary=f"{node.type} {symbol_name}",
                    tokens=count_tokens(chunk_content),
                ))
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return chunks or [self._whole_file_chunk(filepath, content)]
