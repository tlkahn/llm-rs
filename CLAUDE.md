# CLAUDE.md

## Project

LLM-RS: Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) (v0.30). CLI tool for interacting with LLMs via a unified interface. See `doc/roadmap.md` for status and roadmap, `doc/design/architecture.md` for design rationale, `doc/implementation.md` for build history and decisions.

## Commands

```bash
cargo test --workspace           # Run all 548 tests
cargo test -p llm-core           # Core types/traits/config/schema/chain/messages/agent/retry (215 tests)
cargo test -p llm-openai         # OpenAI provider (44 tests)
cargo test -p llm-anthropic      # Anthropic provider (50 tests)
cargo test -p llm-store          # JSONL storage (55 tests)
cargo test -p llm-cli            # CLI unit (57) + integration (127) tests
cargo clippy --workspace         # Lint
cargo build --release -p llm-cli # Build optimized binary

# Library targets (excluded from workspace, built separately):
wasm-pack build crates/llm-wasm --target web      # WASM for browser/Obsidian
cd crates/llm-python && make rebuild              # Python native module (see Makefile)
```

> **Python build footgun — use `make` not raw `uv run`.** `uv run` auto-syncs the venv every invocation, and uv's wheel cache silently reinstalls a stale prior build of `llm-rs 0.1.0` over the fresh `.so` from `maturin develop` (the version never bumps, so uv treats it as "already built"). Always use the Makefile targets in `crates/llm-python/Makefile` — they set `UV_NO_SYNC=1` so the maturin-installed extension sticks. For ad-hoc commands, prefix with `UV_NO_SYNC=1 uv run …` or `export UV_NO_SYNC=1` once per shell. Phase A and Phase B both burned time chasing phantom stale binaries from this.

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
- **`LlmError`** (`error.rs`): seven variants (`Model`, `NeedsKey`, `Provider`, `HttpError`, `Config`, `Io`, `Store`). `HttpError { status: u16, message: String }` for HTTP-level errors. `is_retryable()` returns `true` for 429 and 5xx status codes.
- **`RetryConfig`** (`retry.rs`): exponential backoff configuration. Fields: `max_retries` (default 3), `base_delay_ms` (default 1000), `max_delay_ms` (default 30000), `jitter` (default true). `delay_for_attempt(attempt) -> Duration` computes delay with exponential backoff and optional jitter. Serde-compatible (TOML/JSON).
- Stream helpers: `collect_text()`, `collect_tool_calls()`, `collect_usage()`.
- **`ToolExecutor` trait** (`chain.rs`): async interface for executing tool calls. `execute(&ToolCall) -> ToolResult`.
- **`Usage`** (`types.rs`): token usage with `input`, `output`, `details` fields. `add(&other)` combines two Usage values (summing fields). `total()` returns input + output (treating None as 0).
- **`ChainEvent`** (`chain.rs`): enum for chain loop observability. `IterationStart { iteration, limit, messages }` emitted before provider call, `IterationEnd { iteration, usage, cumulative_usage, tool_calls }` after, `BudgetExhausted { cumulative_usage, budget }` when token budget exceeded.
- **`ChainResult`** (`chain.rs`): result of chain loop. Fields: `chunks`, `tool_results`, `total_usage` (accumulated across all iterations), `budget_exhausted` (bool).
- **`chain()`** (`chain.rs`): chain loop that accumulates `Vec<Message>` across iterations — each provider call sees full conversation history. Executes provider → collects tool calls → executes tools → repeats until no tool calls, limit reached, or budget exceeded. Optional `on_event` callback for observability. `budget: Option<u64>` parameter for token budget enforcement. `parallel: ParallelConfig` controls parallel tool dispatch within a single iteration.
- **`ParallelConfig`** (`chain.rs`): parallel tool dispatch config. Fields: `enabled` (default `true`), `max_concurrent` (`Option<usize>`, `None` = unlimited). `dispatch_tools()` takes a sequential fast path when `enabled == false` or `calls.len() <= 1`; otherwise eagerly collects per-call futures into a `Vec` and drives them via `future::join_all` (unlimited) or `stream::iter(futs).buffered(n)` (bounded). Result order always matches input order. Serde-compatible.
- **`parse_schema_dsl()`** (`schema.rs`): parses "name str, age int" DSL into JSON Schema. Types: str, int, float, bool.
- **`multi_schema()`** (`schema.rs`): wraps a schema in `{"items":{"type":"array","items":<schema>}}` for `--schema-multi`.

### Config system (llm-core/config.rs)

- **`Paths`**: XDG path resolution. `LLM_USER_PATH` -> flat layout; else `$XDG_CONFIG_HOME/llm` + `$XDG_DATA_HOME/llm` with `~/.config` / `~/.local/share` fallbacks. `agents_dir()` returns `config_dir/agents`.
- **`Config`**: TOML config (`config.toml`). Fields: `default_model` (default: `"gpt-4o-mini"`), `logging` (default: `true`), `aliases`, `options`, `providers`. All `#[serde(default)]`. `effective_default_model()` checks `LLM_DEFAULT_MODEL` env var. `resolve_model()` resolves aliases. `model_options(model)` returns options HashMap. `set_option(model, key, value)`, `clear_option(model, key)`, `clear_model_options(model)` for CRUD.
- **`parse_option_value(s)`**: smart coercion of string to JSON value (int, float, bool, null, fallback string).
- **`KeyStore`**: TOML key storage (`keys.toml`). `load/get/set/list/path`. `set()` writes 0o600 on Unix, creates parent dirs.
- **`resolve_key()`**: 4-level chain: explicit `--key` -> `keys.toml` -> env var -> `NeedsKey` error.

### Agent system (llm-core/agent.rs)

- **`AgentConfig`**: TOML config loaded from agent files. Fields: `model` (Option), `system_prompt` (Option), `tools` (Vec), `chain_limit` (default 10), `options` (HashMap), `budget` (Option<BudgetConfig>, wired), `retry` (Option<RetryConfig>, wired), `parallel_tools` (bool, default `true`), `max_parallel_tools` (Option<usize>, default `None`). `AgentConfig::load(path)` returns error if file not found (unlike `Config`). Unknown fields in TOML (including legacy `sub_agents` / `[memory]`) are silently ignored by serde.
- **`BudgetConfig`**: `max_tokens` (Option<u64>). Wired to `chain()` budget enforcement in agent run.
- **Sub-agent delegation and memory are permanently parked** — see [doc/research/specialist-tools-vs-sub-agents.md](doc/research/specialist-tools-vs-sub-agents.md). Hierarchical workflows compose via specialist tools (`llm-tool-*`), not an in-process runtime.
- **`AgentSource`**: enum `Global` / `Local`. Display trait for output.
- **`AgentInfo`**: name + path + source. Returned by discovery.
- **`discover_agents(global_dir, local_dir)`**: scans `.toml` files in both dirs. Local shadows global (same name). Sorted alphabetically. Nonexistent dirs silently skipped.
- **`resolve_agent(name, global_dir, local_dir)`**: finds agent by name, returns `(AgentConfig, PathBuf)`. Local wins. Error if not found.

### Providers

**OpenAI** (`llm-openai`): `POST /v1/chat/completions`, `Authorization: Bearer` auth, SSE with `data: [DONE]` sentinel, `stream_options.include_usage` for token counts. Tool calling via `tools` + `tool_calls` in delta/message. Structured output via `response_format: { type: "json_schema" }`. HTTP errors returned as `LlmError::HttpError { status, message }`. Models: `gpt-4o`, `gpt-4o-mini`.

**Anthropic** (`llm-anthropic`): `POST /v1/messages`, `x-api-key` + `anthropic-version: 2023-06-01` headers, typed SSE events (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`), `max_tokens` required (default 4096), system prompt as top-level field (not in messages). Tool calling via `tools` + `tool_use` content blocks + `input_json_delta` streaming. Structured output via transparent `_schema_output` tool wrapping (tool_use input emitted as Text). HTTP errors returned as `LlmError::HttpError { status, message }`. Models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`.

### Storage (llm-store)

JSONL files, one per conversation, at `$XDG_DATA_HOME/llm/logs/{conversation_id}.jsonl`. Line 1: `ConversationRecord` header (`"type":"conversation"`, `"v":1`). Lines 2+: `ResponseRecord`s (`"type":"response"`) with all data denormalized inline. `LineRecord` is the `#[serde(tag = "type")]` dispatch enum.

Key API: `LogStore::open()`, `log_response(conversation_id, model, &response)`, `read_conversation(id)`, `list_conversations(logs_dir, limit)`, `list_conversations_filtered(logs_dir, limit, &ListOptions)`, `latest_conversation_id(logs_dir)`, `reconstruct_messages(&[Response]) -> Vec<Message>`.

### CLI (llm-cli)

Binary name: `llm`. Built with `clap` derive macros.

**Default subcommand:** `main.rs::rewrite_args()` inserts `"prompt"` before clap parsing when the first arg is not a known subcommand or global flag. This makes `llm "hello"` and `echo "hello" | llm` work.

**Provider registry:** `commands/mod.rs::providers()` returns `Vec<Box<dyn Provider>>` with `#[cfg(feature)]`-gated providers. `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` env vars override API endpoints. Both `openai` and `anthropic` features are default-on.

**Commands:**
- `llm prompt <text>` --- flags: `-m`, `-s`, `--no-stream`, `-n/--no-log`, `--key`, `-u/--usage`, `-o/--option`, `-T/--tool`, `--chain-limit`, `--tools-debug`, `--tools-approve`, `--schema`, `--schema-multi`, `-c/--continue`, `--cid`, `--messages`, `--json`, `-v/--verbose` (count: `-v` summary, `-vv` full messages), `--retries`, `--sequential-tools`, `--max-parallel-tools`
- `llm chat` --- interactive REPL with `rustyline`. Flags: `-m`, `-s`, `-o/--option`, `-T/--tool`, `--chain-limit`, `-v/--verbose`, `--retries`, `--sequential-tools`, `--max-parallel-tools`
- `llm keys set/get/list/path` --- `set` uses rpassword for hidden terminal input
- `llm models list` / `llm models default [model]`
- `llm logs list [--json] [-r] [-n count] [-m model] [-q query] [-u]` / `llm logs path` / `llm logs status` / `llm logs on` / `llm logs off`
- `llm tools list` --- list built-in tools (`llm_version`, `llm_time`)
- `llm schemas dsl <input>` --- parse DSL to JSON Schema
- `llm schemas list` --- scan logs for used schemas
- `llm schemas show <id>` --- show schema by ID
- `llm options set/get/list/clear` --- manage per-model options in config.toml
- `llm aliases set/show/list/remove/path` --- manage model aliases in config.toml
- `llm plugins list` --- show compiled providers, external providers, and external tools
- `llm agent run <name> [prompt]` --- run an agent (accepts stdin). Flags: `-m`, `-s`, `--no-stream`, `-n/--no-log`, `--key`, `-u/--usage`, `-v/--verbose`, `--chain-limit`, `--tools-debug`, `--tools-approve`, `--json`, `--retries`, `--dry-run`, `--sequential-tools`, `--max-parallel-tools`
- `llm agent list` --- list discovered agents (name, model, source)
- `llm agent show <name>` --- print agent config details
- `llm agent init <name> [--global]` --- scaffold TOML template (local by default)
- `llm agent path` --- print global and local agent directory paths

**Exit codes:** 0 success, 1 runtime, 2 config/key/model, 3 provider/network.

### Subprocess Extensibility (`llm-cli/src/subprocess/`)

**Tool protocol** (`llm-tool-*`): Any executable on `$PATH` matching `llm-tool-*` can extend LLM-RS with new tools. Discovery via `--schema` flag (returns JSON matching `Tool` struct). Invocation: arguments JSON on stdin, result on stdout. Exit 0 = success, non-zero = error (stderr).

**Provider protocol** (`llm-provider-*`): Any executable on `$PATH` matching `llm-provider-*` can add model providers. Metadata flags: `--id`, `--models` (JSON array of `ModelInfo`), `--needs-key` (JSON `{"needed":bool,"env_var":?}`). Invocation: `ProviderRequest` JSON on stdin. Streaming: JSONL `ProtocolChunk` lines. Non-streaming: single `ProviderResponse` JSON.

**Module structure**: `protocol.rs` (wire types + Chunk conversion), `discovery.rs` (PATH scanning + schema/metadata fetching), `tool.rs` (`ExternalToolExecutor` impl `ToolExecutor`), `provider.rs` (`SubprocessProvider` impl `Provider`).

### WASM + Python multi-provider

Both `llm-wasm` and `llm-python` use an internal `ProviderImpl` enum dispatching to either `OpenAiProvider` or `AnthropicProvider`. Auto-detection from model name: `"claude*"` -> Anthropic, otherwise OpenAI. Explicit constructors available for full control.

### Persistent logs and programmatic agents (bindings)

`llm-store` exposes an abstract `ConversationStore` trait (in `store.rs`, wasm-safe) with four async methods: `log_response`, `read_conversation`, `list_conversations`, `latest_conversation_id`. `LogStore` implements it natively; `logs`/`query` modules and the `ulid`/`chrono` deps are cfg-gated off wasm32. `build_response` is a native-only helper that synthesizes a `Response` with a ULID id + RFC 3339 datetime for use by library consumers that need to log.

**llm-python.** `LlmClient(log_store=LogStore("/path"))` (or the legacy `log_dir=`) enables auto-logging: every `prompt()` call appends a `Response` to a rolling conversation id kept on the client. `Conversation(client)` inherits the client's store; `conv.send()` auto-logs per user turn. `Conversation.load(client, store, cid)` rehydrates via `reconstruct_messages` and re-seeds the system prompt from `responses[0].system`. `Conversation.persist_to(store)` is attach-only — it errors on non-empty history. `AgentConfig(**kwargs)` / `AgentConfig.from_toml(path)` + `client.run_agent(config, prompt)` runs a configured agent with CLI-parity precedence (model: `config.model` > client default; system: arg > `config.system_prompt`; retry: arg > `config.retry` > client default; tools: `config.tools` whitelisted against the registry with the CLI's `unknown tool in agent config: {name}` error).

**llm-wasm.** `client.setConversationStore({ logResponse, readConversation, listConversations, latestConversationId })` attaches a JS-callback-backed store (callbacks may return Promises). Once set, `client.prompt()` / `client.chain()` / `Conversation.send()` auto-log, and `Conversation.load(client, cid)` reloads from the attached store. `new AgentConfig({ model, system_prompt, tools, ... })` + `client.runAgent(config, prompt, { system?, retries? })` mirrors the Python API. Tool resolution and precedence reuse the same pure helpers from `llm_core::agent`.

## Implementation status

Phase 1 (v0.1) complete --- `echo "Hello" | llm` works end-to-end with streaming + logging for both OpenAI and Anthropic. Core crates compile for `wasm32-unknown-unknown`. WASM library (`llm-wasm`) and Python module (`llm-python`) support both providers.

Phase 2 tools & structured output complete --- Tool calling (both providers), chain loop, built-in tools (`llm_version`, `llm_time`), structured output (OpenAI `response_format`, Anthropic transparent tool wrapping), schema DSL, `--schema`/`--schema-multi` flags, `llm tools list`, `llm schemas dsl/list/show` commands.

Phase 3 conversations & multi-turn complete --- `Message`/`Role` core types, provider multi-turn message building, chain loop accumulates full conversation history, conversation continuation (`-c`/`--cid`), `--messages`/`--json` flags, `llm chat` REPL, `llm logs` full feature set (path/status/on/off, model filter, text search, usage display), `reconstruct_messages()` for conversation reconstruction.

Phase 4 subprocess extensibility complete --- External tool protocol (`llm-tool-*` on PATH with `--schema` discovery, stdin/stdout invocation), external provider protocol (`llm-provider-*` with `--id`/`--models`/`--needs-key` metadata, JSON stdin/JSONL stdout streaming), PATH scanning/dedup, `ExternalToolExecutor` implementing `ToolExecutor`, `SubprocessProvider` implementing `Provider`, composite `CliToolExecutor` (builtin + external), `-T` flag resolves external tools in both `prompt` and `chat`, `llm plugins list` command, `providers()` is now async and includes discovered subprocess providers.

Phase 4 verbose observability complete --- `-v`/`--verbose` flag (count) on `prompt` and `chat` commands. `ChainEvent` enum in llm-core with `IterationStart`/`IterationEnd` variants. `chain()` accepts optional `on_event` callback. `-v` shows iteration summary (number, message count, role summary, per-iteration usage, tool call count). `-vv` additionally dumps full message JSON per iteration. `--verbose` implies `--tools-debug`. `format_chain_event()` and `format_message_summary()` in `prompt.rs`, shared by `chat.rs`.

Phase 4 model options complete --- `-o/--option` flag on `prompt` and `chat` commands (repeatable: `-o temperature 0.7 -o max_tokens 200`). `parse_option_value()` for smart string-to-JSON coercion. `Config.model_options/set_option/clear_option/clear_model_options` methods. `build_options()` merges config defaults with CLI overrides (CLI wins). `llm options set/get/list/clear` subcommands for persistent per-model options in `config.toml`. Options flow through to provider request bodies via `Prompt.options`.

Phase 4 aliases complete --- `Config.set_alias/remove_alias` methods. `llm aliases set/show/list/remove/path` subcommands for managing model aliases in `config.toml`. `resolve_model()` (already existed) resolves aliases at runtime in prompt/chat. No transitive resolution (matches simonw/llm behavior).

Phase 4 (v0.4) is complete. Phase 6 (budget tracking) is complete. Phase 7 (retry/backoff) is complete. Phase 8 (dry-run mode) is complete. Phase 9 (parallel tool execution) is complete. Phase 10 (later-stage features in `llm-wasm` + `llm-python`) is complete. See `doc/roadmap.md` for future work.

Phase 5 agent config & discovery complete --- `AgentConfig` struct with TOML parsing (`model`, `system_prompt`, `tools`, `chain_limit`, `options`, `budget` stub). `Paths.agents_dir()`. Discovery: `discover_agents()` scans global (`$XDG_CONFIG_HOME/llm/agents/`) and local (`$CWD/.llm/agents/`) directories, local shadows global. `resolve_agent()` finds agent by name. CLI: `llm agent run <name> [prompt]` resolves config, model (CLI > agent TOML > global default), tools, builds prompt with system_prompt, calls `chain()`. `llm agent list/show/init/path` management commands. Shared helpers (`find_provider`, `resolve_prompt_text`, `format_chain_event`) extracted from `prompt.rs` as `pub(crate)` and reused by `agent.rs` and `chat.rs`.

Phase 6 budget tracking complete --- `Usage::add()` and `Usage::total()` accumulation helpers. `ChainResult.total_usage` accumulates usage across all chain iterations. `ChainEvent::IterationEnd` includes `cumulative_usage`. `chain()` gains `budget: Option<u64>` parameter — exceeding budget triggers `ChainEvent::BudgetExhausted` and graceful stop (like chain_limit). `ChainResult.budget_exhausted` flag. `-u` flag now shows cumulative usage across chain iterations. Verbose output includes cumulative usage per iteration. `BudgetConfig.max_tokens` wired from agent TOML to `chain()`. Agent budget exhaustion prints `[budget]` warning. Chat REPL tracks session-wide cumulative usage across turns, prints summary on exit.

Phase 7 retry/backoff complete --- `LlmError::HttpError { status, message }` variant for HTTP-level errors (replacing opaque `Provider` strings for HTTP failures). `is_retryable()` method returns `true` for 429 and 5xx. `RetryConfig` struct in `llm-core/retry.rs` with exponential backoff + jitter (`delay_for_attempt()`). `RetryProvider` wrapper in `llm-cli/retry.rs` decorates any `Provider` with retry logic (pre-stream only). `--retries` flag on `prompt`, `chat`, and `agent run` commands. Agent TOML `[retry]` section wired — CLI `--retries` overrides agent config. Both OpenAI and Anthropic providers emit `HttpError` for non-success HTTP status codes.

Phase 8 dry-run mode complete --- `--dry-run` flag on `llm agent run` resolves the full agent invocation pipeline (agent file, model + source, provider, system prompt, prompt text, tools with builtin/external classification, merged options, chain limit, budget, retry, logging flag) without calling the LLM, resolving the API key, or writing logs. `DryRunReport` struct in `llm-cli/commands/dry_run.rs` with `render_plain()` and `render_json()` methods. Plain output is a labeled block with stable field order, sorted options, and optional sections omitted when empty. `--dry-run --json` emits the same info as a JSON envelope (`#[serde(skip_serializing_if)]` on optional fields). `-v`/`-vv` under `--dry-run` populate `report.prompt` with the full serialized `Prompt` JSON the provider would have received (both verbosity levels behave the same). External tool discovery still runs so unknown tool names surface as errors. Tool resolution + prompt construction were reordered before `resolve_key` so the dry-run branch can skip key lookup entirely.

Phase 9 parallel tool execution complete --- `ParallelConfig { enabled, max_concurrent }` in `llm-core/chain.rs`; `chain()` gains a trailing `parallel: ParallelConfig` parameter. `dispatch_tools()` helper runs sequentially when `enabled == false` or `calls.len() <= 1`, otherwise eagerly collects per-call futures into a `Vec` and drives them with `future::join_all` (unlimited) or `stream::iter(futs).buffered(n)` (bounded). Result order is guaranteed to match input order. `AgentConfig` adds `parallel_tools` (default `true`) and `max_parallel_tools` (`Option<usize>`) fields. `--sequential-tools` and `--max-parallel-tools` flags on `prompt`, `chat`, and `agent run`. Precedence for agent: CLI > agent TOML > default. `--tools-approve` forces sequential dispatch to avoid interleaved stdin prompts. `DryRunReport` surfaces the resolved `ParallelConfig` (used by integration tests to assert precedence without timing).

Phase 10 wrap later-stage features into `llm-wasm` + `llm-python` complete (v0.10) --- delivered in three sub-phases.

**Phase A (tools, multi-turn, structured output, built-ins).** Both bindings expose tool calling, multi-turn `Conversation`, structured output with the `parse_schema_dsl`/`multi_schema` helpers, and the built-in `llm_version` / `llm_time` tools.

**Phase B (chain loop, retry, budget).** Both bindings expose a `chain()` wrapper that surfaces per-iteration events, a `RetryProvider` decorator backing the `retries` option, token budgets, and `budget_exhausted` on `ChainResult`.

**Phase C (persistent logs + programmatic agents).** `ConversationStore` trait in `llm-store::store` with dual `Send + Sync` / `?Send` cfg blocks mirroring `ToolExecutor`; `LogStore` implements it natively. `llm-store` gains a wasm32-buildable surface (`records`, `store`, `ConversationSummary`) with `logs`/`query` + `ulid`/`chrono` cfg-gated off wasm32. `build_response` free fn for synthesizing a `Response` with ULID id + RFC 3339 datetime (native-only). `llm-core::agent` gains pure resolution helpers (`resolve_agent_model`/`system`/`retry`/`tools`/`budget`) shared by both bindings, with the CLI's byte-identical `unknown tool in agent config: {name}` error asserted by unit tests. `llm-python` adds `LogStore` pyclass, wires `LlmClient.prompt()` / `Conversation.send()` auto-logging, `Conversation.load`/`persist_to`, `AgentConfig`/`AgentConfig.from_toml(path)`, and `LlmClient.run_agent(config, prompt, *, system=None, retries=None)`. `llm-python` gains a `rlib` crate type and an `extension-module` feature (default on) so pure-Rust helper tests compile without libpython (`--no-default-features`). `llm-wasm` adds `JsConversationStore` backed by four JS `Function` callbacks (each may return a Promise), `LlmClient.setConversationStore(spec)`, auto-logging in `prompt`/`chain`/`Conversation.send`, `Conversation.load(client, cid)`, `AgentConfig` class, and `LlmClient.runAgent(config, text, options)`. WASM Response construction uses a parallel `build_response_wasm` that pulls a ULID-ish id via `crypto.randomUUID` and the datetime via `js_sys::Date`, so `llm-store` stays free of `js-sys` deps.

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
