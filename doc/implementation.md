# Implementation Notes

Status snapshot of what has been built, what remains, and key decisions made along the way. Complements the design-level `metaplan.md`.

---

## Current state (Phase 1, Steps 1--3 complete)

### Crate map

| Crate | Status | Lines | Tests | Purpose |
|-------|--------|-------|-------|---------|
| `llm-core` | Steps 1+3 done | 1075 | 55 | Traits, types, streaming, errors |
| `llm-openai` | Step 2 done | 945 | 29 | OpenAI Chat API provider (streaming + non-streaming) |
| `llm-store` | Step 3 done | 1049 | 42 | JSONL conversation file I/O and queries |
| `llm-cli` | Not started | -- | -- | Binary entry point (Step 5) |

Total: 3069 lines, 126 tests, all passing.

### What works

- **`llm-core`**: `Prompt`, `Response`, `Chunk`, `Usage`, `ModelInfo`, `Attachment`, `Tool`, `ToolCall`, `ToolResult`, `Options` types. `Provider` async trait with streaming `ResponseStream`. Stream collection utilities (`collect_text`, `collect_tool_calls`, `collect_usage`). `LlmError` with `Model`, `NeedsKey`, `Provider`, `Config`, `Io`, `Store` variants.

- **`llm-openai`**: `OpenAiProvider` implementing `Provider` for gpt-4o and gpt-4o-mini. Streaming via SSE with incremental `SseParser`. Non-streaming fallback. Token usage extraction. Tested with `wiremock` HTTP mocking.

- **`llm-store`**: `LogStore` struct with `open`, `log_response`, `read_conversation`. Directory-based `list_conversations` (mtime-sorted) and `latest_conversation_id`. Record types (`ConversationRecord`, `ResponseRecord`, `LineRecord` tagged enum) for JSONL serialization. `conversation_name` helper with truncation and whitespace collapsing.

### What remains in Phase 1

- **Step 4**: `Config` and `KeyStore` in `llm-core` (TOML parsing, XDG paths, key resolution chain)
- **Step 5**: `llm-cli` binary with `prompt`, `keys`, `models`, `logs list` commands

---

## Key decisions

### JSONL over SQLite (storage layer)

The Python `llm` uses SQLite with 12 tables and 21 migrations. We chose JSONL files (one per conversation, append-only) because:

1. The data is hierarchical (conversation > response > tool calls), not relational. JSON nests it naturally; SQL flattens it into junction tables.
2. JSONL is already the project's wire format (subprocess IPC, streaming output, `--json` flag). One format throughout.
3. Standard Unix tools (`cat`, `grep`, `jq`) work directly on log files.
4. No schema migrations. Serde's `#[serde(default)]` handles forward/backward compat.
5. Eliminates `rusqlite` dependency (bundled SQLite adds compile time and binary size).

Trade-off: no FTS5 for full-text search. At typical scales (<10k conversations), `grep`/`rg` across files is fast enough. A search index can be added later if needed.

Migration from Python `llm`: planned `llm import --from-sqlite` (future work).

### JSONL file format

```
$XDG_DATA_HOME/llm/logs/{conversation_id}.jsonl
```

Line 1 --- conversation header:
```json
{"type":"conversation","v":1,"id":"01j...","model":"gpt-4o","name":"Hello world","created":"2026-04-03T12:00:00Z"}
```

Lines 2+ --- one per response, all data denormalized inline:
```json
{"type":"response","id":"01j...","model":"gpt-4o","prompt":"Hello","system":null,"response":"Hi!","options":{},"usage":{"input":5,"output":8,"details":null},"tool_calls":[],"tool_results":[],"attachments":[],"schema":null,"schema_id":null,"duration_ms":230,"datetime":"2026-04-03T12:00:01Z"}
```

The `"v":1` field in the header enables future format evolution. Adding fields is always safe (readers ignore unknowns); renaming or removing fields requires a version bump.

### `Response` as a core type

`Response` lives in `llm-core::types`, not in `llm-store`, because both the CLI (for formatting/display) and the store (for persistence) need it. It represents a materialized response after stream collection --- all text concatenated, tool calls assembled, usage extracted.

### Serde strategy for LineRecord

`LineRecord` uses `#[serde(tag = "type")]` internally-tagged representation. The `"type"` field dispatches between `"conversation"` and `"response"` variants. `ResponseRecord` uses `#[serde(flatten)]` on its inner `Response` to keep all fields at the top level of the JSON object (no nesting). The `Response` variant is `Box<ResponseRecord>` to satisfy clippy's `large_enum_variant` lint.

### ID generation

ULIDs via the `ulid` crate. 26-char lowercase strings, monotonically ordered by timestamp. Conversation IDs double as filenames (`{id}.jsonl`).

### Timestamps

`chrono::Utc::now().to_rfc3339()` for ISO 8601 timestamps in conversation headers. Response datetimes are passed in by the caller (the CLI will set them at response completion time).

---

## Test strategy

- All tests are inline `#[cfg(test)] mod tests` within each module.
- `llm-store` tests use `tempfile::TempDir` for isolated filesystem state.
- `llm-openai` tests use `wiremock::MockServer` for HTTP mocking.
- No integration tests yet (planned for Step 5 when the CLI exists).
- TDD was used throughout: tests written before implementation in each cycle.

---

## Dependencies

| Crate | Key deps |
|-------|----------|
| `llm-core` | `serde`, `serde_json`, `thiserror`, `tokio`, `futures`, `async-trait`, `tokio-stream` |
| `llm-openai` | `llm-core`, `reqwest` (stream+json), `wiremock` (dev) |
| `llm-store` | `llm-core`, `serde_json`, `ulid`, `chrono`, `tempfile` (dev) |

`rusqlite` is listed as an optional workspace dependency for the future `llm import --from-sqlite` command.
