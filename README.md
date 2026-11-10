# Harness context engineering v5.0

> **MCP Tool Server** chạy local trên Windows PowerShell / Antigravity — tối ưu hóa context window với Hybrid RAG, Memory Tiers, dynamic token budgeting, và global prompt optimization.

---

## 🧩 Tech Stack

| Thành phần | Chi tiết |
|---|---|
| **Ngôn ngữ** | Python 3.10+ |
| **Framework** | [`FastMCP`](https://github.com/jlowin/fastmcp) — MCP Server qua `stdio` transport |
| **Embeddings** | **Local TF-IDF + LSA** (Scikit-learn) — Không download, 128-dim |
| **Vector Store** | **In-memory Numpy** (Cosine Similarity) — Nhanh, không phụ thuộc DB external |
| **BM25** | Full-text sparse retrieval (rank-bm25) |
| **Reranker** | **Local Hybrid Reranker** (BM25 + Keyword Overlap) |
| **Chunker** | AST-aware chunker (`treesitter_chunker`: tree-sitter hoặc stdlib `ast` fallback) |
| **Memory DB** | SQLite persistent store — 3 tiers, cached TF-IDF/LSA + lexical relevance, LRU eviction |
| **Logging** | `~/.mcp-harness/harness-v3.log` |
| **Config** | `~/.gemini/config/mcp_config.json` (Antigravity Global) / `.mcp.json` |
| **Entry point** | `server.py` |

---

## 📁 Cấu trúc thư mục

```
harness-context/
├── server.py                  # Entry point — FastMCP server, 11 tools
│
├── chunking/
│   ├── treesitter_chunker.py  # AST-aware code chunker + token counter
│   └── compressor.py          # XML/JSON context compressor
│
├── retrieval/
│   ├── embeddings.py          # Local TF-IDF + LSA embedding engine
│   ├── bm25.py                # BM25 sparse index
│   ├── reranker.py            # Local hybrid reranker
│   ├── graph.py               # Dependency graph (symbol expansion)
│   └── cache.py               # TTL retrieval cache
│
├── memory/
│   ├── episodic.py            # MemoryStore: Episodic / Semantic / Procedural tiers
│   └── vector_store.py        # In-memory Numpy vector store
│
├── context/
│   ├── assembler.py           # Prompt assembler (multi-mode output)
│   ├── budgeting.py           # Dynamic token budget per model
│   └── sanitizer.py           # Input/chunk safety sanitizer
│
└── tests/
    ├── test_harness.py        # Unit tests cho các module & store
    └── test_tools_e2e.py      # E2E tests cho 11 MCP tools & hook_guard.py
```

---

## 🎯 11 MCP Tools

Các tool này được gọi từ MCP client (Antigravity / Gemini / Claude), không phải CLI shell trực tiếp. Payload bên dưới là JSON arguments truyền vào tool.

### 🔍 Retrieval

#### `retrieve_context`
> **MAIN TOOL** — Hybrid RAG pipeline đầy đủ.

Pipeline: `index → hybrid score (BM25 + semantic + graph) → rerank → graph expand → compress → assemble`

| Param | Default | Mô tả |
|---|---|---|
| `base_prompt` | — | System prompt gốc |
| `paths` | — | Danh sách file/folder cần index |
| `query` | — | Task hiện tại |
| `model` | `gemini-3.8-flash` | Model đích (ảnh hưởng token budget & định dạng output) |
| `output_mode` | `concise` | `concise` / `structured` / `code_only` / `minimal` |
| `retrieve_top_n` | `50` | Pool candidates trước rerank |
| `rerank_top_k` | `12` | Chunks sau rerank |
| `memory_top_k` | `4` | Memory entries inject thêm |
| `graph_expand` | `true` | Mở rộng qua dependency graph |
| `compress` | `true` | Nén chunks theo model đích |
| `force_reindex` | `false` | Force re-chunk ngay cả khi đã index |
| `target_ratio` | `0.20` | Giới hạn prompt cuối theo % context window |
| `max_prompt_tokens` | `0` | Giới hạn tuyệt đối; `0` dùng `target_ratio` & `inject_budget` |
| `include_report` | `false` | Chỉ thêm report token khi cần audit/debug |
| `base_prompt_policy` | `reject` | `reject` hoặc `truncate` nếu base prompt quá lớn |

Budget được enforce trên prompt cuối đã assemble:
`base_prompt + retrieved_context + memory + query + wrapper`. Nếu base prompt
tự nó vượt target, tool sẽ reject với báo cáo rõ, trừ khi bật
`base_prompt_policy: "truncate"`.

**Use case:**
- Trước khi sửa một flow lớn, lấy đúng file/chunk liên quan thay vì nhét toàn bộ repo vào context.
- Hỏi kiến trúc hoặc dependency của một module.
- Chuẩn bị context ngắn gọn cho model có context window nhỏ hơn.

**Ví dụ:**
```json
{
  "base_prompt": "You are a senior backend engineer. Use retrieved context only when relevant.",
  "paths": ["src/", "server.py"],
  "query": "optimize auth flow",
  "model": "gpt-5.5",
  "output_mode": "structured",
  "retrieve_top_n": 50,
  "rerank_top_k": 12,
  "memory_top_k": 4,
  "graph_expand": true,
  "compress": true,
  "max_prompt_tokens": 4000
}
```

#### `reindex_paths`
Force re-chunk và re-index các path chỉ định. Chunks cũ của path được thay thế,
toàn bộ vector index được rebuild trên cùng embedding basis, và retrieval cache
được xóa để không trả context cũ.

**Use case:** vừa refactor hoặc tạo file mới, cần index lại để lần `retrieve_context` sau thấy nội dung mới.

**Ví dụ:**
```json
{
  "paths": ["src/auth/", "server.py"]
}
```

#### `invalidate_cache`
Xóa TTL retrieval cache. Cache cũng tự miss khi fingerprint của file/folder thay đổi.

**Use case:** kết quả retrieval cũ không còn đúng vì vừa đổi nhiều file hoặc đổi nhánh.

**Ví dụ:**
```json
{}
```

---

### 🧠 Memory

| Tool | Mô tả |
|---|---|
| `memory_save` | Lưu knowledge vào memory tier (`episodic` / `semantic` / `procedural`) |
| `memory_search` | Tìm kiếm local relevance bằng key, tags và nội dung memory |
| `memory_inject` | Inject memory entries vào system prompt (không cần RAG) |

**3 Memory Tiers:**
- `episodic` — sự kiện, lỗi đã gặp, quyết định cụ thể
- `semantic` — kiến thức, patterns, quy tắc dự án
- `procedural` — quy trình, workflow, cách làm

Memory được lưu tại `~/.mcp-harness/memory.sqlite3`; có thể đổi vị trí bằng
biến môi trường `MCP_HARNESS_MEMORY_DB`.

#### `memory_save`
Lưu một mẩu knowledge vào memory store.

**Use case:** lưu convention của repo, quyết định kỹ thuật, hoặc lỗi đã debug xong để các lần sau retrieve/inject lại.

**Ví dụ:**
```json
{
  "key": "auth-refresh-token-policy",
  "value": "Refresh token rotation is mandatory; never reuse old refresh tokens after successful refresh.",
  "mtype": "semantic",
  "tags": "auth,security"
}
```

#### `memory_search`
Tìm memory theo local TF-IDF/LSA + lexical relevance. `min_sim` được áp dụng
trước khi inject nên memory không liên quan không đi vào prompt.

**Use case:** kiểm tra repo đã từng có quyết định hoặc ghi chú liên quan trước khi sửa code.

**Ví dụ:**
```json
{
  "query": "refresh token rotation",
  "mtype": "semantic",
  "top_k": 5,
  "min_sim": 0.12
}
```

#### `memory_inject`
Inject memory liên quan vào `base_prompt` mà không cần đọc source code.

**Use case:** task cần project rules hoặc decision history, nhưng không cần RAG trên file.

**Ví dụ:**
```json
{
  "base_prompt": "Follow project conventions and answer concisely.",
  "query": "implement auth refresh flow",
  "top_k": 4,
  "min_sim": 0.18,
  "output_mode": "concise",
  "model": "gpt-5.5"
}
```

---

### 📊 Utilities & System Tools

| Tool | Mô tả |
|---|---|
| `estimate_tokens` | Ước tính số token cho một đoạn text bất kỳ |
| `system_port_status` | Kiểm tra trạng thái cổng TCP và tiến trình đang chiếm giữ |
| `system_port_kill` | Giải phóng cổng TCP bằng cách kết thúc tiến trình sở hữu |
| `system_env_status` | Lấy thông tin môi trường (git branch/dirty, active ports, python version, disk space) |
| `sqlite_db_backup` | Backup an toàn database SQLite bằng native `sqlite3.backup()` |

#### `estimate_tokens`
Ước tính token và kiểm tra đoạn text có fit trong các model phổ biến không.

**Ví dụ:**
```json
{
  "text": "Long prompt or retrieved context here..."
}
```

#### `system_port_status` / `system_port_kill`
Kiểm tra và giải phóng port TCP an toàn trên Windows.

#### `system_env_status`
Kiểm tra nhanh tình trạng môi trường làm việc.

#### `sqlite_db_backup`
Backup SQLite database với timestamp snapshot.

---

## 🔄 Luồng hoạt động

```
AI assistant
    └── gọi MCP tool qua stdio
        └── server.py (FastMCP)
            │
            ├── retrieve_context()
            │   ├── AST Chunker       → chunk file theo cú pháp
            │   ├── Embedding Engine  → local TF-IDF + LSA vectors (full rebuild khi corpus đổi)
            │   ├── BM25 Index        → sparse keyword scoring
            │   ├── Hybrid Score      → 60% semantic + 25% BM25 + 10% priority + 5% graph
            │   ├── Reranker          → local BM25 + keyword overlap reranker
            │   ├── Graph Expander    → dependency symbol expansion
            │   ├── Compressor        → provider-aware (Gemini / Codex / OpenAI / Claude)
            │   ├── Memory Store      → inject relevant memories
            │   └── Assembler         → build final enriched prompt
            │
            └── memory_*()
                └── MemoryStore (SQLite) → Episodic / Semantic / Procedural
```

---

## ⚙️ Cấu hình

**`~/.gemini/config/mcp_config.json`** (Antigravity Global — Windows):
```json
{
  "mcpServers": {
    "harness-context-optimizer": {
      "command": "python",
      "args": ["<USER_HOME>/.gemini/config/sidecars/harness-context/server.py"],
      "disabled": false
    }
  }
}
```

---

## 🛠 Lệnh hữu ích (Windows PowerShell 5.1)

```powershell
# Xem log realtime
Get-Content "$env:USERPROFILE\.mcp-harness\harness-v3.log" -Wait

# Kiểm tra sidecar & toàn bộ hệ thống customization
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\.gemini\config\scripts\boost_health.ps1"

# Khởi động lại process MCP sidecar nếu server.py vừa sửa
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\.gemini\config\scripts\restart_mcp_sidecar.ps1" -ForceKillRunning

# Chạy toàn bộ unit + E2E tests
python -m unittest discover -s "$env:USERPROFILE\.gemini\config\sidecars\harness-context\tests" -v
```

---

## 📌 Ghi chú quan trọng

- Log file chính: `~/.mcp-harness/harness-v3.log` (tự rotate tại 512 KB, tối đa 2 bản backup)
- Legacy log (nếu đã tồn tại): `~/.gemini/mcp-harness-v3.log` vẫn được ghi song song
- Config path `~/.gemini/config/mcp_config.json` (Antigravity Global) phải trỏ đến `server.py`
- Nếu chưa cài `tiktoken`, bộ đếm token tự động fallback sang ước lượng `chars / 3.5`
