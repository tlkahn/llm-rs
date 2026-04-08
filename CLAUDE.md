# CLAUDE.md

## Project

LLM-RS: Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) (v0.30). CLI tool for interacting with LLMs via a unified interface. See `doc/metaplan.md` for full architecture, `doc/implementation.md` for build history and decisions.

## Commands

```bash
cargo test --workspace           # Run all 316 tests
cargo test -p llm-core           # Core types/traits/config/schema/chain/messages (119 tests)
cargo test -p llm-openai         # OpenAI provider (42 tests)
cargo test -p llm-anthropic      # Anthropic provider (48 tests)
cargo test -p llm-store          # JSONL storage (49 tests)
cargo test -p llm-cli            # CLI integration tests (58 tests)
cargo clippy --workspace         # Lint
cargo build --release -p llm-cli # Build optimized binary

# Library targets (excluded from workspace, built separately):
wasm-pack build crates/llm-wasm --target web      # WASM for browser/Obsidian
cd crates/llm-python && uv run maturin develop     # Python native module
```

## Architecture

Seven crates in a Cargo workspace (Rust 2024 edition):

```
crates/
  llm-core/      # Traits, types, streaming, errors, config, key management
  llm-openai/    # OpenAI Chat API provider (streaming SSE + non-streaming)
  llm-anthropic/ # Anthropic Messages API provider (streaming SSE + non-streaming)
  llm-store/     # JSONL file-per-conversation log storage
  llm-cli/       # Binary: prompt, keys, models, logs commands
  llm-wasm/      # WASM library for browser/Obsidian (excluded from workspace)
  llm-python/    # Python native module via PyO3 (excluded from workspace)
```

Dependency flow: `llm-cli`, `llm-wasm`, and `llm-python` are top-level entry points -> `llm-openai` + `llm-anthropic` (optional, feature-gated) + `llm-store` -> `llm-core`. No cycles. `llm-openai`, `llm-anthropic`, and `llm-store` are siblings that depend only on `llm-core`. `llm-wasm` and `llm-python` are excluded from `cargo test --workspace` and built with their own toolchains (wasm-pack, maturin).

### Key types (llm-core)

- **`Provider` trait** (`provider.rs`): async streaming interface. Methods: `id()`, `models()`, `needs_key()`, `key_env_var()`, `execute() -> Result<ResponseStream>`.
- **`Role`** (`types.rs`): enum `User`, `Assistant`, `Tool`. Serde as lowercase strings.
- **`Message`** (`types.rs`): role + content + tool_calls + tool_results. Constructors: `user()`, `assistant()`, `assistant_with_tool_calls()`, `tool_results()`.
- **`Prompt`** (`types.rs`): text + system + attachments + tools + tool_calls + tool_results + messages + schema + options. Builder pattern with `with_*` methods.
- **`Response`** (`types.rs`): materialized post-stream result (16 fields: id, model, prompt, system, response text, options, usage, tool_calls, tool_results, attachments, schema, schema_id, duration_ms, datetime).
- **`Chunk`** (`stream.rs`): streaming enum (`Text`, `ToolCallStart`, `ToolCallDelta`, `Usage`, `Done`).
- **`ResponseStream`**: `Pin<Box<dyn Stream<Item=Result<Chunk>> + Send>>` (native); without `Send` on wasm32.
- **`LlmError`** (`error.rs`): six variants (`Model`, `NeedsKey`, `Provider`, `Config`, `Io`, `Store`).
- Stream helpers: `collect_text()`, `collect_tool_calls()`, `collect_usage()`.
- **`ToolExecutor` trait** (`chain.rs`): async interface for executing tool calls. `execute(&ToolCall) -> ToolResult`.
- **`chain()`** (`chain.rs`): chain loop that accumulates `Vec<Message>` across iterations — each provider call sees full conversation history. Executes provider → collects tool calls → executes tools → repeats until no tool calls or limit reached.
- **`parse_schema_dsl()`** (`schema.rs`): parses "name str, age int" DSL into JSON Schema. Types: str, int, float, bool.
- **`multi_schema()`** (`schema.rs`): wraps a schema in `{"items":{"type":"array","items":<schema>}}` for `--schema-multi`.

### Config system (llm-core/config.rs)

- **`Paths`**: XDG path resolution. `LLM_USER_PATH` -> flat layout; else `$XDG_CONFIG_HOME/llm` + `$XDG_DATA_HOME/llm` with `~/.config` / `~/.local/share` fallbacks.
- **`Config`**: TOML config (`config.toml`). Fields: `default_model` (default: `"gpt-4o-mini"`), `logging` (default: `true`), `aliases`, `options`, `providers`. All `#[serde(default)]`. `effective_default_model()` checks `LLM_DEFAULT_MODEL` env var. `resolve_model()` resolves aliases.
- **`KeyStore`**: TOML key storage (`keys.toml`). `load/get/set/list/path`. `set()` writes 0o600 on Unix, creates parent dirs.
- **`resolve_key()`**: 4-level chain: explicit `--key` -> `keys.toml` -> env var -> `NeedsKey` error.

### Providers

**OpenAI** (`llm-openai`): `POST /v1/chat/completions`, `Authorization: Bearer` auth, SSE with `data: [DONE]` sentinel, `stream_options.include_usage` for token counts. Tool calling via `tools` + `tool_calls` in delta/message. Structured output via `response_format: { type: "json_schema" }`. Models: `gpt-4o`, `gpt-4o-mini`.

**Anthropic** (`llm-anthropic`): `POST /v1/messages`, `x-api-key` + `anthropic-version: 2023-06-01` headers, typed SSE events (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`), `max_tokens` required (default 4096), system prompt as top-level field (not in messages). Tool calling via `tools` + `tool_use` content blocks + `input_json_delta` streaming. Structured output via transparent `_schema_output` tool wrapping (tool_use input emitted as Text). Models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`.

### Storage (llm-store)

JSONL files, one per conversation, at `$XDG_DATA_HOME/llm/logs/{conversation_id}.jsonl`. Line 1: `ConversationRecord` header (`"type":"conversation"`, `"v":1`). Lines 2+: `ResponseRecord`s (`"type":"response"`) with all data denormalized inline. `LineRecord` is the `#[serde(tag = "type")]` dispatch enum.

Key API: `LogStore::open()`, `log_response(conversation_id, model, &response)`, `read_conversation(id)`, `list_conversations(logs_dir, limit)`, `list_conversations_filtered(logs_dir, limit, &ListOptions)`, `latest_conversation_id(logs_dir)`, `reconstruct_messages(&[Response]) -> Vec<Message>`.

### CLI (llm-cli)

Binary name: `llm`. Built with `clap` derive macros.

**Default subcommand:** `main.rs::rewrite_args()` inserts `"prompt"` before clap parsing when the first arg is not a known subcommand or global flag. This makes `llm "hello"` and `echo "hello" | llm` work.

**Provider registry:** `commands/mod.rs::providers()` returns `Vec<Box<dyn Provider>>` with `#[cfg(feature)]`-gated providers. `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` env vars override API endpoints. Both `openai` and `anthropic` features are default-on.

**Commands:**
- `llm prompt <text>` --- flags: `-m`, `-s`, `--no-stream`, `-n/--no-log`, `--key`, `-u/--usage`, `-T/--tool`, `--chain-limit`, `--tools-debug`, `--tools-approve`, `--schema`, `--schema-multi`, `-c/--continue`, `--cid`, `--messages`, `--json`
- `llm chat` --- interactive REPL with `rustyline`. Flags: `-m`, `-s`, `-T/--tool`, `--chain-limit`
- `llm keys set/get/list/path` --- `set` uses rpassword for hidden terminal input
- `llm models list` / `llm models default [model]`
- `llm logs list [--json] [-r] [-n count] [-m model] [-q query] [-u]` / `llm logs path` / `llm logs status` / `llm logs on` / `llm logs off`
- `llm tools list` --- list built-in tools (`llm_version`, `llm_time`)
- `llm schemas dsl <input>` --- parse DSL to JSON Schema
- `llm schemas list` --- scan logs for used schemas
- `llm schemas show <id>` --- show schema by ID

**Exit codes:** 0 success, 1 runtime, 2 config/key/model, 3 provider/network.

### WASM + Python multi-provider

Both `llm-wasm` and `llm-python` use an internal `ProviderImpl` enum dispatching to either `OpenAiProvider` or `AnthropicProvider`. Auto-detection from model name: `"claude*"` -> Anthropic, otherwise OpenAI. Explicit constructors available for full control.

## Implementation status

Phase 1 (v0.1) complete --- `echo "Hello" | llm` works end-to-end with streaming + logging for both OpenAI and Anthropic. Core crates compile for `wasm32-unknown-unknown`. WASM library (`llm-wasm`) and Python module (`llm-python`) support both providers.

Phase 2 tools & structured output complete --- Tool calling (both providers), chain loop, built-in tools (`llm_version`, `llm_time`), structured output (OpenAI `response_format`, Anthropic transparent tool wrapping), schema DSL, `--schema`/`--schema-multi` flags, `llm tools list`, `llm schemas dsl/list/show` commands.

Phase 3 conversations & multi-turn complete --- `Message`/`Role` core types, provider multi-turn message building, chain loop accumulates full conversation history, conversation continuation (`-c`/`--cid`), `--messages`/`--json` flags, `llm chat` REPL, `llm logs` full feature set (path/status/on/off, model filter, text search, usage display), `reconstruct_messages()` for conversation reconstruction.

Next: Phase 4 (subprocess extensibility, Ollama provider, aliases, options, attachments). See `doc/metaplan.md` for the full roadmap.

## Conventions

- Rust 2024 edition, `resolver = "2"` workspace
- TDD throughout: tests written before implementation
- IDs: ULID (26-char lowercase), via `ulid` crate
- Timestamps: RFC 3339 via `chrono`
- Errors: single `LlmError` enum in llm-core, `#[from]` for `io::Error`
- Unit tests: inline `#[cfg(test)]` modules per source file
- Integration tests: `tests/integration.rs` with `assert_cmd` for CLI, `wiremock` for HTTP mocking
- Test isolation: `LLM_USER_PATH` for filesystem, `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` for API endpoints, `temp_env` for env vars
- Feature flags: `llm-cli` uses optional `openai` and `anthropic` features (both default on)
