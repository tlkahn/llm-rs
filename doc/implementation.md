# Implementation Notes

Status snapshot of what has been built, what remains, and key decisions made along the way. Complements the design-level `metaplan.md`.

---

## Current state (Phase 1 complete)

Phase 1 goal was: `echo "Hello" | llm` works end-to-end --- streams to stdout, logs to JSONL. All five steps are done.

### Crate map

| Crate | Status | Lines | Tests | Purpose |
|-------|--------|------:|------:|---------|
| `llm-core` | Complete | 1745 | 88 | Traits, types, streaming, errors, config, keys |
| `llm-openai` | Complete | 945 | 29 | OpenAI Chat API provider (streaming SSE + non-streaming) |
| `llm-store` | Complete | 1049 | 42 | JSONL conversation file I/O and queries |
| `llm-cli` | Complete | 1209 | 29 | Binary: prompt, keys, models, logs commands |

Total: ~4950 lines, 188 tests, all passing.

### What works

**`llm-core`** (Steps 1 + 4)

- `Prompt`, `Response`, `Chunk`, `Usage`, `ModelInfo`, `Attachment`, `Tool`, `ToolCall`, `ToolResult`, `Options` types.
- `Provider` async trait with streaming `ResponseStream` (`Pin<Box<dyn Stream<Item=Result<Chunk>>>>`).
- Stream collection utilities: `collect_text`, `collect_tool_calls`, `collect_usage`.
- `LlmError` with six variants: `Model`, `NeedsKey`, `Provider`, `Config`, `Io`, `Store`.
- `Paths`: pure XDG path resolution with `LLM_USER_PATH` override.
- `Config`: TOML config loading with serde defaults, alias resolution, `LLM_DEFAULT_MODEL` env override.
- `KeyStore`: TOML-backed key storage with 0o600 permissions on Unix.
- `resolve_key()`: 4-level key resolution chain (explicit -> store -> env var -> error).

**`llm-openai`** (Step 2)

- `OpenAiProvider` implementing `Provider` for `gpt-4o` and `gpt-4o-mini`.
- Streaming via SSE with incremental `SseParser` (handles partial HTTP chunks, `[DONE]` signal).
- Non-streaming fallback (single JSON response).
- Token usage extraction from both streaming and non-streaming responses.
- `OPENAI_BASE_URL` env var support for API endpoint override.

**`llm-store`** (Step 3)

- `LogStore`: `open`, `log_response` (create or append), `read_conversation`.
- `list_conversations`: directory-based listing sorted by mtime (newest first), reads only first line per file.
- `latest_conversation_id`: O(1) lookup via mtime.
- Record types: `ConversationRecord`, `ResponseRecord`, `LineRecord` (tagged enum for JSONL dispatch).
- `ConversationSummary` with `Serialize` for JSON output.
- `conversation_name`: human-readable name generation with truncation and whitespace collapsing.

**`llm-cli`** (Step 5)

- `llm prompt <text>` with flags: `-m/--model`, `-s/--system`, `--no-stream`, `-n/--no-log`, `--key`, `-u/--usage`.
- Default subcommand: `llm "text"` and `echo "text" | llm` work without writing `prompt`.
- Stdin piping: reads from stdin when not a terminal; combines with positional arg if both present.
- `llm keys set/get/list/path` --- `set` uses `rpassword` for hidden terminal input, reads plain line when piped.
- `llm models list` --- prints model IDs with provider names.
- `llm models default [model]` --- get or set the default model (read-modify-write on `config.toml`).
- `llm logs list [--json] [-r/--response] [-n/--count N]` --- conversation summaries, JSONL output, most-recent response text.
- Exit codes: 0 (success), 1 (runtime/IO), 2 (config/key/model), 3 (provider/network).
- Automatic JSONL logging on every prompt (unless `-n` flag or `config.logging = false`).
- Provider registry via `providers()` function with `#[cfg(feature)]`-gated provider construction.

### What remains

Phase 1 is the minimum viable CLI. Remaining phases from `metaplan.md`:

- **Phase 2 (v0.2):** Conversations (`-c`, `--cid`, `llm chat`), Anthropic + Ollama providers, options, attachments, aliases, extract.
- **Phase 3 (v0.3):** Tool calling, structured output, schema DSL.
- **Phase 4 (v0.4):** Subprocess provider/tool protocol, `--verbose`, shell completions.

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

`chrono::Utc::now().to_rfc3339()` for ISO 8601 timestamps in conversation headers. Response datetimes are set at response completion time by the CLI.

### Configuration system (Step 4)

Pure XDG path resolution (no `dirs` crate). `$HOME/.config/llm/` for config, `$HOME/.local/share/llm/` for data. `LLM_USER_PATH` flattens both into a single directory (Python compat). Config and keys are TOML files (`config.toml`, `keys.toml`) consolidating what Python scattered across 6+ JSON/txt files.

**Path resolution order** (`Paths::resolve()`):
1. `$LLM_USER_PATH` -> flat layout (both config and data dirs point there)
2. `$XDG_CONFIG_HOME/llm` / `$XDG_DATA_HOME/llm`
3. `$HOME/.config/llm` / `$HOME/.local/share/llm`

**Key resolution chain** (`resolve_key()`):
1. Explicit `--key` CLI flag (literal value, not an alias)
2. `keys.toml` entry matching provider's `needs_key` name
3. Environment variable (e.g. `OPENAI_API_KEY`)
4. `NeedsKey` error with actionable message

`Config` fields use `#[serde(default)]` for graceful degradation: missing file -> defaults, partial file -> defaults for missing fields, extra unknown fields -> ignored. `LLM_DEFAULT_MODEL` env var overrides the config file's `default_model`. Model aliases in `config.toml` resolved via `Config::resolve_model()`.

`keys.toml` gets 0o600 permissions on Unix. `KeyStore::set()` creates parent directories automatically.

### Default subcommand (Step 5)

Clap does not natively support a default subcommand. We use argv rewriting in `main.rs:rewrite_args()`: before clap parsing, if the first real argument is not a known subcommand (`prompt`, `keys`, `models`, `logs`) or global flag (`--help`, `--version`), insert `"prompt"` at position 1. When no args at all and stdin is piped, also insert `"prompt"`. This gives:

- `llm "hello"` -> `llm prompt "hello"`
- `llm -m gpt-4o "hello"` -> `llm prompt -m gpt-4o "hello"`
- `echo "hi" | llm` -> `echo "hi" | llm prompt`
- `llm --help` -> unchanged (shows top-level help)
- `llm keys list` -> unchanged (recognized subcommand)

### Provider registry (Step 5)

`commands/mod.rs::providers()` returns a `Vec<Box<dyn Provider>>` with all compiled-in providers. Each provider is behind a `#[cfg(feature)]` gate (e.g. `feature = "openai"`). The OpenAI provider reads `OPENAI_BASE_URL` env var at construction time, defaulting to `https://api.openai.com`. This supports both OpenAI-compatible APIs (vllm, LiteLLM) and test mocking (wiremock).

### Exit code mapping (Step 5)

| `LlmError` variant | Exit code | Category |
|---------------------|-----------|----------|
| `Io`, `Store` | 1 | Runtime error |
| `Model`, `NeedsKey`, `Config` | 2 | Configuration error |
| `Provider` | 3 | Network/API error |

Matches the design in `metaplan.md`. Errors print to stderr before exiting.

### Interactive key input (Step 5)

`llm keys set <name>` detects whether stdin is a terminal. If so, uses `rpassword` for hidden input (key does not appear on screen or in shell history). If stdin is piped, reads a plain line (for scripting and testing: `echo "sk-..." | llm keys set openai`).

### Config mutation for `models default` (Step 5)

`llm models default <model>` read-modify-writes `config.toml` using `toml::Table` to preserve unknown fields. This avoids adding a `Config::save()` method to `llm-core`, keeping the core crate focused on read-only config loading.

---

## Test strategy

- **`llm-core`** (88 tests): Inline `#[cfg(test)]` modules. `tempfile::TempDir` for filesystem isolation, `temp_env` for safe env var scoping.
- **`llm-openai`** (29 tests): Inline modules. `wiremock::MockServer` for HTTP mocking (SSE streaming + non-streaming + error responses).
- **`llm-store`** (42 tests): Inline modules. `tempfile::TempDir` for isolated filesystem state. JSONL round-trip tests, unicode handling, malformed-line recovery.
- **`llm-cli`** (29 tests): Integration tests in `tests/integration.rs` using `assert_cmd` + `predicates`. Tests run the compiled binary as a subprocess, asserting on stdout/stderr/exit code. API-dependent tests use `wiremock` with `OPENAI_BASE_URL` pointing to the local mock server. All tests use `LLM_USER_PATH` for filesystem isolation. Helper functions (`openai_non_streaming_body`, `openai_streaming_body`, `write_test_conversation`) create mock data.
- TDD was used throughout: tests written before implementation in each cycle.

---

## Dependencies

| Crate | Key deps | Dev deps |
|-------|----------|----------|
| `llm-core` | `serde`, `serde_json`, `thiserror`, `tokio`, `futures`, `async-trait`, `tokio-stream`, `toml` | `temp-env`, `tempfile` |
| `llm-openai` | `llm-core`, `reqwest` (stream + json) | `wiremock` |
| `llm-store` | `llm-core`, `serde_json`, `ulid`, `chrono` | `tempfile` |
| `llm-cli` | `llm-core`, `llm-openai` (optional), `llm-store`, `clap`, `tokio`, `serde_json`, `futures`, `tokio-stream`, `toml`, `ulid`, `chrono`, `rpassword` | `assert_cmd`, `predicates`, `wiremock`, `tempfile`, `temp-env` |

Workspace dependencies declared in root `Cargo.toml`. `llm-openai` is an optional dependency of `llm-cli` behind the `openai` feature flag (enabled by default).

---

## Phase 1 build order

Each step was a self-contained TDD cycle: write failing tests, make them pass, refactor.

| Step | Crate | What was built | Tests added |
|------|-------|----------------|------------:|
| 1 | `llm-core` | `Prompt`, `Chunk`, `Response`, `Usage`, `Provider` trait, `LlmError` | 54 |
| 2 | `llm-openai` | `OpenAiProvider`, SSE parser, message builder | 29 |
| 3 | `llm-store` | `LogStore`, JSONL file I/O, conversation listing | 42 |
| 4 | `llm-core` | `Paths`, `Config`, `KeyStore`, `resolve_key()` | 34 |
| 5 | `llm-cli` | `prompt`, `keys`, `models`, `logs` commands, default subcommand, exit codes, logging | 29 |

Step 5 was further broken into 12 inner TDD cycles (scaffold, keys path, keys set/get/list, models list, models default, logs list, prompt non-streaming, prompt streaming, prompt flags, stdin+default-subcmd, exit codes, logging).
