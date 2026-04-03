# CLAUDE.md

## Project

LLM-RS: Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) (v0.30). CLI tool for interacting with LLMs via a unified interface. See `doc/metaplan.md` for full architecture, `doc/implementation.md` for build history and decisions.

## Commands

```bash
cargo test --workspace           # Run all 188 tests
cargo test -p llm-core           # Core types/traits/config (88 tests)
cargo test -p llm-openai         # OpenAI provider (29 tests)
cargo test -p llm-store          # JSONL storage (42 tests)
cargo test -p llm-cli            # CLI integration tests (29 tests)
cargo clippy --workspace         # Lint
cargo build --release -p llm-cli # Build optimized binary
```

## Architecture

Four crates in a Cargo workspace (Rust 2024 edition):

```
crates/
  llm-core/     # Traits, types, streaming, errors, config, key management
  llm-openai/   # OpenAI Chat API provider (streaming SSE + non-streaming)
  llm-store/    # JSONL file-per-conversation log storage
  llm-cli/      # Binary: prompt, keys, models, logs commands
```

Dependency flow: `llm-cli` -> `llm-openai` (optional, feature-gated) + `llm-store` -> `llm-core`. No cycles. `llm-openai` and `llm-store` are siblings that both depend only on `llm-core`.

### Key types (llm-core)

- **`Provider` trait** (`provider.rs`): async streaming interface. Methods: `id()`, `models()`, `needs_key()`, `key_env_var()`, `execute() -> Result<ResponseStream>`.
- **`Prompt`** (`types.rs`): text + system + attachments + tools + tool_results + schema + options. Builder pattern with `with_*` methods.
- **`Response`** (`types.rs`): materialized post-stream result (16 fields: id, model, prompt, system, response text, options, usage, tool_calls, tool_results, attachments, schema, schema_id, duration_ms, datetime).
- **`Chunk`** (`stream.rs`): streaming enum (`Text`, `ToolCallStart`, `ToolCallDelta`, `Usage`, `Done`).
- **`ResponseStream`**: `Pin<Box<dyn Stream<Item=Result<Chunk>> + Send>>`.
- **`LlmError`** (`error.rs`): six variants (`Model`, `NeedsKey`, `Provider`, `Config`, `Io`, `Store`).
- Stream helpers: `collect_text()`, `collect_tool_calls()`, `collect_usage()`.

### Config system (llm-core/config.rs)

- **`Paths`**: XDG path resolution. `LLM_USER_PATH` -> flat layout; else `$XDG_CONFIG_HOME/llm` + `$XDG_DATA_HOME/llm` with `~/.config` / `~/.local/share` fallbacks.
- **`Config`**: TOML config (`config.toml`). Fields: `default_model` (default: `"gpt-4o-mini"`), `logging` (default: `true`), `aliases`, `options`, `providers`. All `#[serde(default)]`. `effective_default_model()` checks `LLM_DEFAULT_MODEL` env var. `resolve_model()` resolves aliases.
- **`KeyStore`**: TOML key storage (`keys.toml`). `load/get/set/list/path`. `set()` writes 0o600 on Unix, creates parent dirs.
- **`resolve_key()`**: 4-level chain: explicit `--key` -> `keys.toml` -> env var -> `NeedsKey` error.

### Storage (llm-store)

JSONL files, one per conversation, at `$XDG_DATA_HOME/llm/logs/{conversation_id}.jsonl`. Line 1: `ConversationRecord` header (`"type":"conversation"`, `"v":1`). Lines 2+: `ResponseRecord`s (`"type":"response"`) with all data denormalized inline. `LineRecord` is the `#[serde(tag = "type")]` dispatch enum.

Key API: `LogStore::open()`, `log_response(conversation_id, model, &response)`, `read_conversation(id)`, `list_conversations(logs_dir, limit)`, `latest_conversation_id(logs_dir)`.

### CLI (llm-cli)

Binary name: `llm`. Built with `clap` derive macros.

**Default subcommand:** `main.rs::rewrite_args()` inserts `"prompt"` before clap parsing when the first arg is not a known subcommand or global flag. This makes `llm "hello"` and `echo "hello" | llm` work.

**Provider registry:** `commands/mod.rs::providers()` returns `Vec<Box<dyn Provider>>` with `#[cfg(feature)]`-gated providers. `OPENAI_BASE_URL` env var overrides the API endpoint.

**Commands:**
- `llm prompt <text>` --- flags: `-m`, `-s`, `--no-stream`, `-n/--no-log`, `--key`, `-u/--usage`
- `llm keys set/get/list/path` --- `set` uses rpassword for hidden terminal input
- `llm models list` / `llm models default [model]`
- `llm logs list [--json] [-r] [-n count]`

**Exit codes:** 0 success, 1 runtime, 2 config/key/model, 3 provider/network.

## Implementation status

Phase 1 (v0.1) complete --- `echo "Hello" | llm` works end-to-end with streaming + logging.

Next: Phase 2 (conversations, multi-provider, attachments). See `doc/metaplan.md` for the full roadmap.

## Conventions

- Rust 2024 edition, `resolver = "2"` workspace
- TDD throughout: tests written before implementation
- IDs: ULID (26-char lowercase), via `ulid` crate
- Timestamps: RFC 3339 via `chrono`
- Errors: single `LlmError` enum in llm-core, `#[from]` for `io::Error`
- Unit tests: inline `#[cfg(test)]` modules per source file
- Integration tests: `tests/integration.rs` with `assert_cmd` for CLI, `wiremock` for HTTP mocking
- Test isolation: `LLM_USER_PATH` for filesystem, `OPENAI_BASE_URL` for API endpoint, `temp_env` for env vars
- Feature flags: `llm-cli` uses optional `openai` feature (default on) for `llm-openai` dependency
