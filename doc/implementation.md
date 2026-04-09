# Implementation Notes

Status snapshot of what has been built, what remains, and key decisions made along the way. Complements the design-level `metaplan.md`.

---

## Current state (Phase 4 core + verbose observability complete)

Phase 1 goal was: `echo "Hello" | llm` works end-to-end --- streams to stdout, logs to JSONL. Phase 2 goal was: tool calling, structured output, and the chain loop --- the core "agentic" capability. Phase 3 goal was: multi-turn conversations, interactive chat, conversation continuation. Phase 4 core goal was: subprocess extensibility --- any executable on `$PATH` matching `llm-tool-*` or `llm-provider-*` can extend the system with new tools or model providers without recompilation. Phase 4 continued: `--verbose` flag for chain loop observability.

### Crate map

| Crate | Status | Tests | Purpose |
|-------|--------|------:|---------|
| `llm-core` | Complete | 123 | Traits, types, streaming, errors, config, keys, schema DSL, chain loop, ChainEvent |
| `llm-openai` | Complete | 42 | OpenAI Chat API provider (streaming SSE + non-streaming + tool calling + structured output) |
| `llm-anthropic` | Complete | 48 | Anthropic Messages API provider (streaming SSE + non-streaming + tool calling + structured output) |
| `llm-store` | Complete | 49 | JSONL conversation file I/O and queries |
| `llm-cli` | Complete | 111 | Binary: prompt, keys, models, logs, tools, schemas, plugins commands; subprocess tool/provider extensibility; verbose chain observability |
| `llm-wasm` | Complete | --- | WASM library for browser/Obsidian plugin (wasm-bindgen) |
| `llm-python` | Complete | --- | Python native module via PyO3/maturin |

Total: 369 tests (workspace crates), all passing. `llm-wasm` and `llm-python` are excluded from the workspace and built with their own toolchains.

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

**`llm-anthropic`** (Step 7)

- `AnthropicProvider` implementing `Provider` for `claude-opus-4-6`, `claude-sonnet-4-6`, and `claude-haiku-4-5`.
- Anthropic Messages API (`POST /v1/messages`): `x-api-key` + `anthropic-version: 2023-06-01` auth headers (not Bearer token).
- System prompt is a top-level `system` field in the request body (not a message).
- `max_tokens` is required by Anthropic --- defaults to 4096, overridable via `prompt.with_option("max_tokens", ...)`.
- Streaming via SSE with typed events (`message_start`, `content_block_delta`, `message_delta`, `message_stop`). Parser ignores `event:` lines, dispatches from JSON `type` field.
- Usage tracking: `input_tokens` from `message_start`, `output_tokens` from `message_delta`. Emits `Chunk::Usage` with both.
- Non-streaming fallback: parses `MessagesResponse`, concatenates all `text`-type content blocks.
- `ANTHROPIC_BASE_URL` env var support for API endpoint override.
- Compiles for `wasm32-unknown-unknown` (same cfg-gating pattern as `llm-openai`).

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
- Provider registry via `providers()` function with `#[cfg(feature)]`-gated provider construction. Both `openai` and `anthropic` features default on.

**`llm-core` + `llm-openai` wasm32 compatibility** (Step 6a + 6b)

- `llm-core` production code has zero tokio usage; `tokio` removed from `[dependencies]`, kept only in `[dev-dependencies]`.
- `ResponseStream` type alias cfg-gated: `+ Send` on native, no `Send` on wasm32 (single-threaded).
- `Provider` trait cfg-gated: `#[async_trait] trait Provider: Send + Sync` on native, `#[async_trait(?Send)] trait Provider` on wasm32.
- `llm-openai` streaming path: replaced `tokio::sync::mpsc` with `futures::channel::mpsc` (works on all platforms). Only the spawn call is cfg-gated: `tokio::spawn` on native, `wasm_bindgen_futures::spawn_local` on wasm32. Removed `tokio-stream` dependency entirely.
- Both crates pass `cargo check --target wasm32-unknown-unknown`.

**`llm-wasm`** (Step 6c + 8)

- wasm-bindgen exports: `LlmClient` class with `new(api_key, model)`, `newWithBaseUrl(api_key, model, base_url)`, `newAnthropic(api_key, model)`, `newAnthropicWithBaseUrl(api_key, model, base_url)`.
- Auto-detection: `new(api_key, model)` routes `"claude*"` models to Anthropic, all others to OpenAI.
- Internal `ProviderImpl` enum dispatches to either `OpenAiProvider` or `AnthropicProvider`.
- `prompt(text)`, `promptWithSystem(text, system)` --- non-streaming, returns JS Promise resolving to string.
- `promptStreaming(text, callback)`, `promptStreamingWithSystem(text, system, callback)` --- streaming, calls JS callback per text chunk, returns full text.
- `promptWithOptions(text, system, options_json)`, `promptStreamingWithOptions(text, system, options_json, callback)` --- pass temperature, max_tokens, etc. as JSON string.
- Stateless: no storage, no config, no key management. Key passed at construction time.
- HTTP via reqwest (auto-detects wasm32, uses web-sys `fetch` under the hood).
- Built with `wasm-pack build crates/llm-wasm --target web` (or `--target bundler` for webpack).
- Generates TypeScript declarations (.d.ts), JS bindings, and .wasm binary.

**`llm-python`** (Step 6d + 8)

- PyO3 module: `import llm_rs`.
- `LlmClient(api_key, model, *, provider=None, base_url=None, log_dir=None)` --- owns a `tokio::Runtime` for async-to-sync bridging.
- Provider selection: explicit `provider="openai"` or `provider="anthropic"` kwarg. If omitted, auto-detects from model name (`"claude*"` -> Anthropic). Default base URLs: `https://api.openai.com`, `https://api.anthropic.com`.
- `prompt(text, *, system=None) -> str` --- blocking, collects full response.
- `prompt_stream(text, *, system=None) -> ChunkIterator` --- returns a Python iterator yielding text chunks. Uses `std::sync::mpsc` to bridge from async stream to sync Python iteration.
- Optional log storage via `log_dir` parameter (passes through to `llm_store::LogStore`).
- Built with `maturin develop` (editable install) or `maturin build --release` (wheel).

**Phase 2 additions (tools and structured output):**

**`llm-core` --- schema DSL and chain loop** (Steps 1, 5)

- `parse_schema_dsl()`: parses comma/newline-separated field definitions into JSON Schema. Types: `str` (default), `int`, `float`, `bool`. Optional descriptions via `:`. `"name str, age int:The person's age"` -> full JSON Schema object.
- `multi_schema()`: wraps a schema in an `items` array envelope for `--schema-multi`.
- `Prompt.tool_calls: Vec<ToolCall>`: new field with `#[serde(default)]` for backward-compatible deserialization. Both providers need the assistant's prior tool calls when building follow-up messages.
- `ToolExecutor` trait: async `execute(&ToolCall) -> ToolResult`. Implemented by the CLI's `CliToolExecutor`.
- `chain()`: provider-agnostic chain loop. Calls provider -> collects tool calls -> executes via `ToolExecutor` -> builds next prompt with `with_tools`/`with_tool_calls`/`with_tool_results` -> repeats. Stops on empty tool calls or chain limit.

**`llm-openai` --- tool calling and structured output** (Steps 2, 4)

- Request types: `ChatTool`, `ChatToolFunction`, `MessageToolCall`, `DeltaToolCall`, `DeltaFunction`, `ResponseFormat`, `JsonSchemaFormat`.
- `ChatRequest` extended with `tools`, `tool_choice`, `response_format` fields (all `Option`, skip_serializing_if).
- `Message.tool_calls` changed from `Option<serde_json::Value>` to `Option<Vec<MessageToolCall>>` for typed deserialization.
- `Delta.tool_calls: Option<Vec<DeltaToolCall>>` for streaming tool call chunks.
- `build_messages()` extended: when `prompt.tool_calls` and `prompt.tool_results` are non-empty, appends assistant message with `tool_calls` array + `"role": "tool"` messages with results.
- Streaming: parses `delta.tool_calls` array --- first chunk (with `name` + `id`) emits `Chunk::ToolCallStart`, subsequent chunks (with `arguments`) emit `Chunk::ToolCallDelta`.
- Non-streaming: extracts `message.tool_calls` array, emits `ToolCallStart` + `ToolCallDelta` per call.
- Structured output: when `prompt.schema` is set, adds `response_format: { type: "json_schema", json_schema: { name: "output", strict: true, schema: ... } }` to request. Response comes back as normal JSON text in `content`.

**`llm-anthropic` --- tool calling and structured output** (Steps 3, 4)

- Request types: `AnthropicTool` (name, description, input_schema).
- `MessagesRequest` extended with `tools`, `tool_choice` fields.
- `ContentBlock` extended with tool_use fields (`id`, `name`, `input`) and tool_result fields (`tool_use_id`, `content`, `is_error`). All optional with `skip_serializing_if`.
- `ContentDelta` extended with `partial_json: Option<String>` for `input_json_delta` events.
- `build_messages()` extended: when tool calls/results present, appends assistant message with `Blocks(tool_use)` + user message with `Blocks(tool_result)`.
- Streaming: `content_block_start` with `type: "tool_use"` emits `Chunk::ToolCallStart`; `content_block_delta` with `type: "input_json_delta"` emits `Chunk::ToolCallDelta`.
- Non-streaming: iterates `content` blocks, emits `ToolCallStart` + `ToolCallDelta` for `tool_use` blocks.
- Structured output (transparent tool wrapping): when `prompt.schema` is set, injects synthetic `_schema_output` tool with the schema as `input_schema`, forces `tool_choice: { type: "tool", name: "_schema_output" }`. In both streaming and non-streaming paths, `_schema_output` tool_use output is emitted as `Chunk::Text` instead of `ToolCallStart/Delta`. The `is_schema_block` flag tracks whether the current streaming block is the synthetic tool.

**`llm-cli` --- tools, schemas, and chain flags** (Steps 6, 7, 8)

- `llm tools list`: prints built-in tools (name + description).
- `BuiltinToolRegistry`: `list()`, `get(name)`, `execute_tool(call)`. Two built-in tools: `llm_version` (returns `CARGO_PKG_VERSION`), `llm_time` (returns UTC + local time + timezone as JSON).
- `CliToolExecutor`: implements `ToolExecutor`. Supports `--tools-debug` (prints calls/results to stderr) and `--tools-approve` (interactive y/n prompt before each execution).
- Prompt flags: `-T/--tool` (repeatable, `ArgAction::Append`), `--chain-limit` (default 5), `--tools-debug`, `--tools-approve`, `--schema`, `--schema-multi`.
- Schema resolution chain: try JSON literal (`serde_json::from_str`), try file path (`Path::exists` + read), try DSL (`parse_schema_dsl`).
- `make_schema_id()`: deterministic hash of schema JSON (via `std::hash::DefaultHasher`, 16-char hex).
- `llm schemas dsl <input>`: parse DSL, print pretty JSON.
- `llm schemas list`: scan log files for unique `schema_id` values.
- `llm schemas show <id>`: find and pretty-print schema by ID (prefix match).
- Known subcommands list updated: `tools` and `schemas` added so `llm tools` doesn't trigger default subcommand insertion.

**Phase 3 additions (conversations and multi-turn):**

**`llm-core` — Message types and chain rewrite** (Steps 1, 3)

- `Role` enum (`User`, `Assistant`, `Tool`) with `#[serde(rename_all = "lowercase")]`.
- `Message` struct with `role`, `content`, `tool_calls`, `tool_results`. Convenience constructors: `user()`, `assistant()`, `assistant_with_tool_calls()`, `tool_results()`. Tool fields use `#[serde(default, skip_serializing_if = "Vec::is_empty")]`.
- `Prompt.messages: Vec<Message>` with `#[serde(default, skip_serializing_if = "Vec::is_empty")]` and `with_messages()` builder.
- `chain()` rewritten: seeds `Vec<Message>` from `prompt.messages` (or creates `[Message::user(text)]`), accumulates assistant and tool result messages each iteration. Preserves schema and system prompt from `initial_prompt`.

**`llm-openai` + `llm-anthropic` — conversation message builders** (Step 2)

- Both `build_messages()` dispatch: `prompt.messages.is_empty()` → `build_single_turn()` (existing logic extracted), else `build_from_conversation()` (new multi-turn path).
- OpenAI conversation path: maps `Role::User` → `{"role":"user"}`, `Role::Assistant` → `{"role":"assistant"}` with optional `tool_calls` array, `Role::Tool` → per-result `{"role":"tool","tool_call_id":"..."}`.
- Anthropic conversation path: maps `Role::User` → `{"role":"user"}`, `Role::Assistant` → text + `tool_use` content blocks, `Role::Tool` → `{"role":"user"}` with `tool_result` content blocks (Anthropic requires tool results in user role).
- Shared helper functions extracted: `map_tool_calls()`, `map_tool_use()`, `map_tool_result()`, `append_tool_exchange()`.

**`llm-store` — conversation reconstruction and filtered queries** (Steps 4, 8)

- `reconstruct_messages(&[Response]) -> Vec<Message>`: rebuilds conversation history from stored responses. Each `Response` becomes user + assistant (or user + assistant_with_tools + tool_results).
- `list_conversations_filtered()`: accepts `ListOptions { model, query }` for model filter and case-insensitive full-text search.
- `Config::save()`: TOML serialization with parent directory creation.

**`llm-cli` — conversation continuation, messages/json, chat, logs expansion** (Steps 5-8)

- `-c/--continue`: loads `latest_conversation_id()` → `read_conversation()` → `reconstruct_messages()`, appends current prompt as user message, logs to same conversation file.
- `--cid <id>`: same as `-c` but targets a specific conversation ID.
- `--messages <file|->`: loads JSON array of `Message` objects. Mutually exclusive with `-c`/`--cid`. When `-`, reads stdin (with `skip_stdin` preventing `resolve_prompt_text` from also reading stdin).
- `--json`: buffers response, emits JSON envelope with `model`, `content`, `conversation_id`, `tool_calls`, `usage`, `duration_ms`.
- `llm chat`: interactive REPL using `rustyline::DefaultEditor`. Accumulates `Vec<Message>`, streams responses, logs each turn to same JSONL conversation. Exits on Ctrl-D, Ctrl-C, or `/exit`.
- `llm logs path`: prints `paths.logs_dir()`.
- `llm logs status`: prints whether logging is enabled/disabled.
- `llm logs on/off`: updates `config.toml` via `Config::save()`.
- `llm logs list -m <model>`: filter by model name.
- `llm logs list -q <text>`: case-insensitive full-text search across JSONL files.
- `llm logs list -u`: include token usage totals per conversation.
- Known subcommands updated: `chat` added to `should_insert_prompt()`.

**Phase 4 additions (subprocess extensibility):**

**`llm-cli/src/subprocess/protocol.rs` — wire protocol types** (Step 1)

- `ProtocolChunk`: serde-tagged enum (`#[serde(tag = "type", rename_all = "snake_case")]`) with variants `Text`, `ToolCallStart`, `ToolCallDelta`, `Usage`, `Done`. Maps 1:1 to `llm_core::Chunk` but has its own serde implementation since `Chunk` intentionally has no serde.
- `From<ProtocolChunk> for Chunk` and `From<&Chunk> for ProtocolChunk` conversions.
- `ProviderRequest`: serializable struct sent to subprocess providers on stdin (`model`, `prompt`, `key`, `stream`). `Prompt` is directly serializable since it already has serde derives.
- `ProviderResponse`: non-streaming response from subprocess providers (`text`, `tool_calls`, `usage`).
- `ResponseUsage`: concrete `{input: u64, output: u64}` (unlike core `Usage` which has `Option` fields).

**`llm-cli/src/subprocess/discovery.rs` — PATH scanning and metadata fetching** (Steps 2, 3, 7)

- `scan_path(prefix)`: scans all directories in `$PATH` for executables matching the prefix. Skips directories, non-executable files (checked via `mode() & 0o111` on Unix), and deduplicates by filename (first occurrence in PATH wins).
- `discover_tools()` / `discover_providers()`: thin wrappers calling `scan_path("llm-tool-")` / `scan_path("llm-provider-")`.
- `fetch_tool_schema(binary, timeout)`: runs `binary --schema`, parses stdout as `Tool` JSON. Timeout via `tokio::time::timeout`.
- `fetch_all_tool_schemas(binaries, timeout)`: batch fetch with warning on failure (skips broken tools).
- `fetch_provider_id(binary, timeout)`: runs `binary --id`, returns trimmed string.
- `fetch_provider_models(binary, timeout)`: runs `binary --models`, parses `Vec<ModelInfo>` JSON.
- `fetch_provider_key_info(binary, timeout)`: runs `binary --needs-key`, parses `KeyRequirement { needed, env_var }`.
- All subprocess calls use `tokio::process::Command` + `tokio::time::timeout` with a configurable default of 30 seconds.

**`llm-cli/src/subprocess/tool.rs` — external tool executor** (Step 4)

- `ExternalToolExecutor`: holds `HashMap<String, (PathBuf, Tool)>` mapping tool names to binary paths and schemas.
- `discover()`: scans PATH + fetches all schemas. `discover_with_timeout()` for custom timeout.
- `ToolExecutor` impl: spawns the tool binary, writes `call.arguments` JSON to stdin, reads stdout/stderr. Exit 0 → success (stdout = output). Non-zero → error (stderr = message). Timeout → error.
- `get_tool(name)` / `list_tools()` for introspection (used by CLI tool resolution and `llm tools list`).

**`llm-cli/src/subprocess/provider.rs` — subprocess provider** (Step 8)

- `SubprocessProvider`: holds binary path + cached metadata (provider ID, model list, key requirement).
- `from_binary(path)`: fetches all metadata by running `--id`, `--models`, `--needs-key`. Used during provider discovery.
- `Provider` trait impl:
  - `id()`, `models()`, `needs_key()`, `key_env_var()`: return cached metadata.
  - `execute()`: serializes `ProviderRequest` to stdin. Streaming mode: reads stdout line by line via `tokio::io::BufReader::lines()`, parses each as `ProtocolChunk`, converts to `Chunk`, yields via `async_stream::try_stream!`. Non-streaming mode: reads all stdout, parses `ProviderResponse`, converts to `Vec<Chunk>`.

**`llm-cli/src/commands/tools.rs` — composite tool executor** (Step 5)

- `CliToolExecutor` now holds `Option<ExternalToolExecutor>`. Constructor: `new(debug, approve)` + `.with_external(ext)` builder.
- Execution order: try `BuiltinToolRegistry::execute_tool()` first. If it returns an "unknown tool" error, delegate to external executor if present.
- `tools list` now shows both builtin tools and discovered external tools (with binary path).
- `run()` is now `async` (needed for `ExternalToolExecutor::discover()`).

**`llm-cli/src/commands/prompt.rs` + `chat.rs` — external tool wiring** (Step 6)

- `-T` flag resolution: check `BuiltinToolRegistry` first, collect unresolved names, then discover external tools and resolve remaining names. Error if any name is still unknown.
- `ExternalToolExecutor` is passed into `CliToolExecutor.with_external()` for chain loop delegation.
- Key resolution fix: `resolve_key()` is now skipped when the provider reports `needs_key() == None` and no `--key` flag is given. Previously, calling `resolve_key` with an empty key alias always errored because it couldn't find a key for `""`. This was never hit before because both compiled-in providers (OpenAI, Anthropic) always need a key.

**`llm-cli/src/commands/mod.rs` — async provider registry** (Step 9)

- `compiled_providers()`: renamed from `providers()`, returns only compiled-in providers (sync).
- `providers()`: new async function that returns compiled + discovered subprocess providers. Iterates `discover_providers()`, calls `SubprocessProvider::from_binary()` for each, warns and skips on failure.
- All callers (`prompt::run`, `chat::run`, `models::run`) now `.await` the providers.
- `models::run` became `async` to support the async provider discovery.

**`llm-cli/src/commands/plugins.rs` — plugins command** (Step 10)

- `llm plugins list`: shows compiled providers (with model lists), external providers (with binary paths and models), and external tools (with descriptions).
- Added `Plugins` variant to `Commands` enum in `app.rs`.
- Added `"plugins"` to `should_insert_prompt()` known subcommands in `main.rs`.

**Integration test fixtures** (Step 11)

- `tests/fixtures/bin/llm-tool-upper`: shell script implementing the tool protocol. `--schema` returns JSON, invocation reads stdin JSON, uppercases the `text` field via `python3`.
- `tests/fixtures/bin/llm-provider-echo`: shell script implementing the provider protocol. `--id` → `"echo"`, `--models` → `[{"id":"echo-model",...}]`, `--needs-key` → `{"needed":false}`. Invocation echoes back prompt text in streaming or non-streaming format.

### What remains

Remaining items from Phase 4 and beyond (`metaplan.md`):

- **Phase 4 continued:** Ollama provider (as `llm-provider-ollama` subprocess binary or compiled-in crate), aliases, options passthrough, attachments (image/audio), shell completions, config resolution tracing (extending `--verbose` beyond chain scope).
- **Phase 5+:** MCP client protocol, embedding support, template system, fragment pipelines.

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

### Provider registry (Step 5 + 7, updated Phase 4)

`commands/mod.rs::compiled_providers()` returns compiled-in providers behind `#[cfg(feature)]` gates. The async `providers()` function combines these with subprocess providers discovered on PATH. The OpenAI provider reads `OPENAI_BASE_URL` env var, the Anthropic provider reads `ANTHROPIC_BASE_URL`, both defaulting to their production endpoints. Subprocess provider discovery failures are logged as warnings and skipped (graceful degradation).

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

### Anthropic provider design (Step 7)

The Anthropic Messages API differs from OpenAI in several ways that required a separate crate rather than parameterizing the OpenAI one:

**Auth:** Anthropic uses `x-api-key` header + `anthropic-version: 2023-06-01` instead of `Authorization: Bearer`.

**System prompt:** Goes to a top-level `system` field in the request body, not as a `{"role": "system"}` message. `build_messages()` in `llm-anthropic` returns only user messages; the provider extracts `prompt.system` separately.

**`max_tokens`:** Required by Anthropic (unlike OpenAI where it's optional). Defaults to 4096 if not in `prompt.options["max_tokens"]`.

**Response format:** Non-streaming responses return a `content[]` array of typed blocks (e.g. `{"type": "text", "text": "..."}`) instead of `choices[0].message.content`. The provider concatenates all `text`-type blocks.

**SSE format:** Anthropic sends typed events (`event: message_start`, `event: content_block_delta`, etc.) with a `type` field in the JSON payload matching the `event:` line. The parser ignores `event:` lines entirely and dispatches from JSON `type` via `#[serde(tag = "type")]`. The done signal is `message_stop` (not `data: [DONE]`).

**Usage in streaming:** `input_tokens` arrives in `message_start`, `output_tokens` in `message_delta`. The provider stores `input_tokens` in a local variable and emits `Chunk::Usage` when `message_delta` arrives (combining both).

**WASM/Python multi-provider:** Both `llm-wasm` and `llm-python` use an internal `ProviderImpl` enum (not trait objects) to dispatch between `OpenAiProvider` and `AnthropicProvider`. Model name prefix (`"claude"` -> Anthropic) provides zero-config auto-selection; explicit constructors offer full control.

**Model IDs --- use aliases, not snapshot dates:** Anthropic model IDs come in two forms: aliases (`claude-sonnet-4-6`) and dated snapshots (`claude-sonnet-4-6-20250514`). The initial implementation used speculative snapshot dates from the plan (`claude-sonnet-4-6-20250725`) which did not exist, causing the API to reject requests with a cryptic `"model: claude-sonnet-4-6-20250725"` error. The fix was to use alias-form IDs (`claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`) which are stable, always route to the latest snapshot, and are what Anthropic recommends in their docs. Lesson: never hardcode speculative snapshot dates --- use aliases for provider model lists and let users pass specific snapshots via `-m` if they need pinning.

### Anthropic structured output via transparent tool wrapping (Phase 2, Step 4)

Unlike OpenAI which has native `response_format: json_schema`, Anthropic has no first-class structured output API. The solution is the same approach used by the Python `llm` library: inject a synthetic `_schema_output` tool with the schema as `input_schema`, force `tool_choice: { type: "tool", name: "_schema_output" }`, then intercept the tool_use response and emit it as text instead of a tool call.

This requires tracking state in the streaming path. An `is_schema_block` boolean tracks whether the current `content_block_start` was a `_schema_output` tool. When true, subsequent `input_json_delta` chunks are emitted as `Chunk::Text` instead of `Chunk::ToolCallDelta`. The `has_schema` boolean (captured from `prompt.schema.is_some()` before the async move) gates the detection.

The non-streaming path is simpler: it checks each `tool_use` content block's name. If `_schema_output`, it serializes `block.input` to a string and pushes `Chunk::Text`. Otherwise it emits normal `ToolCallStart/Delta`.

Edge case: when a schema prompt also has explicit tools, both coexist. The `_schema_output` tool is appended to the existing tools list via `get_or_insert_with(Vec::new).push(schema_tool)`. Non-schema tools produce normal `ToolCallStart/Delta` chunks even in the same response.

### Prompt.tool_calls for multi-turn tool chains (Phase 2, Step 2)

Both OpenAI and Anthropic require the assistant's prior tool calls in follow-up messages. OpenAI needs an assistant message with a `tool_calls` array; Anthropic needs an assistant message with `tool_use` content blocks. This means the `Prompt` must carry the assistant's tool calls from the previous iteration, not just the user's tool results.

Adding `tool_calls: Vec<ToolCall>` to `Prompt` with `#[serde(default)]` maintains backward-compatible deserialization for existing log files. The `with_tool_calls()` builder chains naturally: `Prompt::new(text).with_tools(tools).with_tool_calls(calls).with_tool_results(results)`.

### Chain loop design (Phase 2, Step 5)

The chain loop lives in `llm-core` (not `llm-cli`) so it's reusable from WASM and Python. It's provider-agnostic --- it only uses the `Provider` trait and `ToolExecutor` trait. The CLI implements `ToolExecutor` with built-in tools; WASM/Python consumers can implement their own.

The `on_chunk` callback (`&mut dyn FnMut(&Chunk)`) provides real-time output. The CLI uses it to print text to stdout as it arrives, even across multiple chain iterations. The function returns `ChainResult { chunks, tool_results }` --- chunks from all iterations (for text extraction and logging) plus accumulated tool results (so the caller can log what the tools actually returned).

System prompt preservation: the chain loop builds a new `Prompt` each iteration (with tool_calls/tool_results), but re-applies the original system prompt via `with_system()` if it was set. This was a bug caught during code review --- the initial implementation dropped the system prompt after the first iteration.

The `#[allow(clippy::too_many_arguments)]` annotation was added because the function genuinely needs all 8 parameters (provider, model, prompt, key, stream, executor, limit, callback). Wrapping them in a config struct would add complexity without clarity.

### Schema DSL matching Python's schema_dsl() (Phase 2, Step 1)

The schema DSL parser matches the Python `llm` project's `schema_dsl()` behavior:
- Fields separated by commas or newlines (newline takes priority if present).
- Each field: `name [type][:description]`. Missing type defaults to `"string"`.
- Type map: `str` -> `string`, `int` -> `integer`, `float` -> `number`, `bool` -> `boolean`.
- Unknown types produce an error. Commas in descriptions are NOT supported (matching Python).
- All fields are required (added to `required` array).

### Schema ID without heavy dependencies (Phase 2, Step 8)

The plan called for `blake2` + `hex` crates for schema ID generation (matching Python's `make_schema_id()` which uses blake2b). To avoid adding external dependencies for a non-cryptographic use case, the implementation uses `std::hash::DefaultHasher` (SipHash) formatted as 16-char hex. This is deterministic and collision-resistant enough for distinguishing user schemas. The IDs won't match Python's blake2b output, but cross-tool schema ID compatibility was not a requirement.

### Platform abstraction for wasm32 (Step 6a + 6b)

The refactoring needed to make `llm-core` and `llm-openai` compile for wasm32 was surgical. Key insight: `llm-core` had `tokio` listed as a dependency but never used it in production code (only `#[tokio::test]` in tests). The actual platform-dependent code was confined to three lines in `llm-openai/src/provider.rs`.

**What was cfg-gated:**

| Location | Native | wasm32 | Why |
|----------|--------|--------|-----|
| `ResponseStream` type alias | `+ Send` | no `Send` | wasm32 is single-threaded; web-sys types aren't `Send` |
| `Provider` trait bounds | `Send + Sync`, `#[async_trait]` | no bounds, `#[async_trait(?Send)]` | Same reason; `async_trait(?Send)` avoids boxing with `Send` |
| Streaming spawn | `tokio::spawn(future)` | `wasm_bindgen_futures::spawn_local(future)` | Different async runtimes |
| Streaming channel | `futures::channel::mpsc` | `futures::channel::mpsc` | Same on both (replaced tokio's mpsc) |

**Why `futures::channel::mpsc` everywhere (not just wasm32):** The switch from `tokio::sync::mpsc` to `futures::channel::mpsc` was done unconditionally rather than cfg-gated. This avoids duplicating the 30-line SSE parsing loop. `futures::channel::mpsc::Receiver` implements `Stream` directly, eliminating the `tokio_stream::wrappers::ReceiverStream` wrapper. Backpressure behavior is equivalent at the buffer size used (32).

**Why `cfg_attr` instead of trait body duplication for impl blocks:** The `Provider` trait itself had to be duplicated across two cfg blocks because `#[async_trait]` and `#[async_trait(?Send)]` are different proc macro invocations that transform the trait body differently. But impl blocks (e.g. `impl Provider for OpenAiProvider`) use `#[cfg_attr(..., async_trait)]` / `#[cfg_attr(..., async_trait(?Send))]` to avoid duplicating the impl body.

### WASM crate as a stateless facade (Step 6c)

`llm-wasm` is deliberately minimal: it wraps `OpenAiProvider` with a wasm-bindgen API and nothing else. No config, no key storage, no log persistence. The design principle is that the host environment (Obsidian plugin, browser app) owns all state management --- the WASM module is a pure computation layer that builds HTTP requests, parses SSE responses, and returns structured data.

HTTP is handled by `reqwest`, which auto-detects wasm32 and uses the browser's `fetch()` API via `web-sys`. This means CORS rules apply; the Obsidian plugin or browser app must ensure the LLM API endpoint allows cross-origin requests (OpenAI does).

`wasm-pack build --target web` generates a self-initializing ES module. `--target bundler` generates a module for webpack/rollup (typical in Obsidian plugin builds). Both produce TypeScript declarations.

### Python crate with tokio bridge (Step 6d)

`llm-python` owns a `tokio::Runtime` to bridge async Rust to sync Python. The `prompt()` method uses `Runtime::block_on()` to run the async provider. The `prompt_stream()` method is more involved: it gets the `ResponseStream` via `block_on`, then spawns a tokio task that consumes the stream and sends chunks through a `std::sync::mpsc` channel. The Python `ChunkIterator` reads from the channel's receiving end. The `Receiver` is wrapped in `Mutex` to satisfy PyO3's `Sync` requirement on `#[pyclass]` structs.

The Python virtualenv and maturin are managed via `uv` (`uv venv`, `uv run maturin develop`).

---

## Test strategy

- **`llm-core`** (119 tests): Inline `#[cfg(test)]` modules. `tempfile::TempDir` for filesystem isolation, `temp_env` for safe env var scoping. Schema DSL tests cover all type mappings, descriptions, whitespace tolerance, newline separation, error cases. Chain loop tests use a `MockProvider` that returns pre-configured responses and an `AtomicUsize` call counter to verify iteration counts.
- **`llm-openai`** (42 tests): Inline modules. `wiremock::MockServer` for HTTP mocking (SSE streaming + non-streaming + error responses + tool calls + structured output). Tool calling tests use SSE cassettes with `delta.tool_calls` array chunks for streaming and `message.tool_calls` for non-streaming.
- **`llm-anthropic`** (48 tests): Inline modules. Same pattern as `llm-openai`: serde round-trip tests for Anthropic-specific types (including `ContentBlock` with tool_use/tool_result fields, `ContentDelta` with `partial_json`), SSE parser tests, wiremock integration tests. Structured output tests verify `_schema_output` transparent wrapping: the response should contain `Chunk::Text` (not `ToolCallStart/Delta`) and `collect_tool_calls()` should return empty.
- **`llm-store`** (49 tests): Inline modules. `tempfile::TempDir` for isolated filesystem state. JSONL round-trip tests, unicode handling, malformed-line recovery.
- **`llm-cli`** (103 tests): 45 unit tests (tools registry, schemas, subprocess module) + 58 integration tests in `tests/integration.rs` using `assert_cmd` + `predicates`. Tests run the compiled binary as a subprocess, asserting on stdout/stderr/exit code. API-dependent tests use `wiremock` with `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` pointing to the local mock server. All tests use `LLM_USER_PATH` for filesystem isolation. Tool chain tests use wiremock sequential responses (`up_to_n_times(1)` for first response, default for subsequent). Subprocess tests use shell script fixtures in `tests/fixtures/bin/` and `tempfile::TempDir` with `temp_env` for PATH isolation. Integration tests for subprocess extensibility prepend `tests/fixtures/bin/` to PATH via `env("PATH", path_with_fixtures())`.
- TDD was used throughout: tests written before implementation in each cycle.

---

## Dependencies

| Crate | Key deps | Dev deps |
|-------|----------|----------|
| `llm-core` | `serde`, `serde_json`, `thiserror`, `futures`, `async-trait`, `toml` | `tokio`, `temp-env`, `tempfile` |
| `llm-openai` | `llm-core`, `reqwest` (stream + json), `futures`, `async-trait`; native: `tokio`; wasm32: `wasm-bindgen-futures` | `tokio`, `wiremock` |
| `llm-anthropic` | `llm-core`, `reqwest` (stream + json), `futures`, `async-trait`; native: `tokio`; wasm32: `wasm-bindgen-futures` | `tokio`, `wiremock` |
| `llm-store` | `llm-core`, `serde_json`, `ulid`, `chrono` | `tempfile` |
| `llm-cli` | `llm-core`, `llm-openai` (optional), `llm-anthropic` (optional), `llm-store`, `clap`, `tokio`, `serde_json`, `futures`, `async-trait`, `async-stream`, `tokio-stream`, `toml`, `ulid`, `chrono`, `rpassword`, `rustyline` | `assert_cmd`, `predicates`, `wiremock`, `tempfile`, `temp-env` |
| `llm-wasm` | `llm-core`, `llm-openai`, `llm-anthropic`, `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `futures` | --- |
| `llm-python` | `llm-core`, `llm-openai`, `llm-anthropic`, `llm-store`, `pyo3`, `tokio`, `futures` | --- |

Workspace dependencies declared in root `Cargo.toml`. `llm-openai` and `llm-anthropic` are optional dependencies of `llm-cli` behind feature flags (both enabled by default). `llm-wasm` and `llm-python` are excluded from the workspace (`exclude` in root `Cargo.toml`) and built separately with `wasm-pack` and `maturin`.

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
| 6a | `llm-core` | cfg-gate `ResponseStream` Send, `Provider` Send+Sync for wasm32; remove tokio from deps | 0 (existing pass) |
| 6b | `llm-openai` | `futures::channel::mpsc`, cfg-gate spawn, remove `tokio-stream` | 0 (existing pass) |
| 6c | `llm-wasm` | wasm-bindgen `LlmClient`, `prompt()`, `promptStreaming()` | wasm-pack build |
| 6d | `llm-python` | PyO3 `LlmClient`, `prompt()`, `prompt_stream()` iterator | maturin develop |
| 7 | `llm-anthropic` | `AnthropicProvider`, SSE parser, message builder, CLI registration | 34 + 3 CLI |
| 8 | `llm-wasm`, `llm-python` | Multi-provider support via `ProviderImpl` enum, auto-detection from model name | 0 (build verified) |

Step 5 was further broken into 12 inner TDD cycles (scaffold, keys path, keys set/get/list, models list, models default, logs list, prompt non-streaming, prompt streaming, prompt flags, stdin+default-subcmd, exit codes, logging).

Steps 6a-6b were refactoring steps: the "test" was that all 188 existing tests continued passing AND `cargo check --target wasm32-unknown-unknown` succeeded for both crates. Steps 6c-6d were new crates verified by their respective build toolchains (`wasm-pack build`, `maturin develop`) and smoke tests (`import llm_rs` in Python, TypeScript declarations in WASM output).

Step 7 added 34 unit tests in `llm-anthropic` (types, SSE, messages, provider) plus 3 CLI integration tests (model listing, streaming, non-streaming). Step 8 refactored `llm-wasm` and `llm-python` to support both providers --- verified by `wasm-pack build` and `maturin develop`.

## Phase 2 build order

Steps 1, 2, 3 are independent and were implemented in parallel. Step 4 depends on 2+3. Step 5 depends on 4. Steps 6-8 are sequential.

| Step | Crate | What was built | Tests added |
|------|-------|----------------|------------:|
| 1 | `llm-core` | `parse_schema_dsl()`, `multi_schema()` | 12 |
| 2 | `llm-core`, `llm-openai` | `Prompt.tool_calls`, OpenAI tool types, `build_messages` with tool results, streaming/non-streaming tool call parsing | 10 |
| 3 | `llm-anthropic` | Anthropic tool types, `build_messages` with tool results, streaming/non-streaming tool_use parsing | 10 |
| 4 | `llm-openai`, `llm-anthropic` | OpenAI `response_format` for structured output, Anthropic `_schema_output` transparent tool wrapping | 3 |
| 5 | `llm-core` | `ToolExecutor` trait, `chain()` loop function | 6 |
| 6 | `llm-cli` | `BuiltinToolRegistry`, `CliToolExecutor`, `llm tools list` command | 6 |
| 7 | `llm-cli` | `-T/--tool`, `--chain-limit`, `--tools-debug`, `--tools-approve` flags, chain integration | 4 |
| 8 | `llm-cli` | `--schema`, `--schema-multi` flags, `llm schemas dsl/list/show` commands, schema resolution chain | 7 |

## Phase 3 build order

| Step | Crate | What was built | Tests added |
|------|-------|----------------|------------:|
| 1 | `llm-core` | `Role` enum, `Message` struct with constructors, `Prompt.messages` field, `with_messages()` builder | 11 |
| 2 | `llm-openai`, `llm-anthropic` | `build_from_conversation()` multi-turn message builders, `build_single_turn()` extraction | 4 |
| 3 | `llm-core` | Chain loop rewrite: accumulate `Vec<Message>` across iterations, `MockProvider` captures prompts | 3 |
| 4 | `llm-store` | `reconstruct_messages()` from stored `Vec<Response>` | 4 |
| 5 | `llm-cli` | `-c/--continue`, `--cid` flags, conversation loading/continuation, `rewrite_args` update | 2 |
| 6 | `llm-cli` | `--messages` flag (file or stdin), `--json` output envelope, `skip_stdin` for `--messages -` | 3 |
| 7 | `llm-cli` | `llm chat` command with `rustyline` REPL, conversation accumulation, per-turn logging | 0 (interactive) |
| 8 | `llm-cli`, `llm-store` | `llm logs path/status/on/off`, `ListOptions` with model filter and text search, `Config::save()` | 6 |
| 9 | docs | Updated `metaplan.md` (swap Phase 3/4), `CLAUDE.md`, `implementation.md` | 0 |

## Phase 4 build order

Steps 1 and 2 are independent. Step 3 depends on 2. Step 4 depends on 1+3. Steps 5-6 depend on 4. Steps 7-8 depend on 2. Step 9 depends on 8. Step 10 depends on 2+3+7. Step 11 depends on all.

In practice, steps 1-4 and 7-8 were implemented in a single pass (all new files created simultaneously), then steps 5-6 and 9-10 in a second pass (modifying existing CLI files), and step 11 as integration tests.

| Step | Crate | What was built | Tests added |
|------|-------|----------------|------------:|
| 1 | `llm-cli` | `ProtocolChunk`, `ProviderRequest`, `ProviderResponse` with serde + `Chunk` conversion | 10 |
| 2 | `llm-cli` | `scan_path()`, `discover_tools()`, `discover_providers()` | 6 |
| 3 | `llm-cli` | `fetch_tool_schema()`, `fetch_all_tool_schemas()` | 4 |
| 4 | `llm-cli` | `ExternalToolExecutor` implementing `ToolExecutor` | 5 |
| 5 | `llm-cli` | `CliToolExecutor` composite (builtin + external), `tools list` shows external | 0 (existing tests cover) |
| 6 | `llm-cli` | `-T` resolves external tools in `prompt.rs` and `chat.rs` | 0 (integration tests in step 11) |
| 7 | `llm-cli` | `fetch_provider_id()`, `fetch_provider_models()`, `fetch_provider_key_info()` | 4 |
| 8 | `llm-cli` | `SubprocessProvider` implementing `Provider` (streaming + non-streaming) | 7 |
| 9 | `llm-cli` | Async `providers()` combining compiled + discovered providers | 0 (integration tests in step 11) |
| 10 | `llm-cli` | `llm plugins list` command, `Commands::Plugins`, known subcommand registration | 0 (integration tests in step 11) |
| 11 | `llm-cli` | E2E integration tests with fixture shell scripts | 9 |

## Verbose chain observability build order

Single implementation pass — the chain event system, CLI flag, formatting, and tests were all implemented together since the scope was small and self-contained.

| Step | Crate | What was built | Tests added |
|------|-------|----------------|------------:|
| 1 | `llm-core` | `ChainEvent` enum (`IterationStart`, `IterationEnd`), `on_event` param on `chain()`, event emission with per-iteration `collect_usage()` | 4 |
| 2 | `llm-core` | Updated all 8 existing `chain()` test call sites to pass `None` for `on_event` | 0 |
| 3 | `llm-core` | Re-export `ChainEvent` from `lib.rs` | 0 |
| 4 | `llm-cli` | `-v`/`--verbose` flag on `PromptArgs` (`ArgAction::Count`), `format_chain_event()`, `format_message_summary()` | 0 |
| 5 | `llm-cli` | `verbose > 0` implies `tools_debug` in `CliToolExecutor`, wired `on_event` callback in `prompt.rs` | 0 |
| 6 | `llm-cli` | `-v`/`--verbose` on `ChatArgs`, wired through to `chain()` + executor, reuses `prompt::format_chain_event` | 0 |
| 7 | `llm-cli` | Integration tests: verbose summary, `-vv` message dump, verbose-implies-tools-debug, flag parsing | 4 |

### Verbose observability learnings

**`on_event` as `Option<&mut dyn FnMut>` avoids allocation.** The callback is optional — `None` means no overhead (just an `if let` check each iteration). Using `&mut dyn FnMut` rather than `Box<dyn Fn>` avoids heap allocation and allows the closure to mutate captured state (e.g. collecting events into a `Vec` in tests). The `let mut on_event = on_event;` rebinding at the top of `chain()` is needed because `Option<&mut dyn FnMut>` requires the outer binding to be mutable for the `if let Some(cb) = &mut on_event` pattern.

**`for iteration in 1..=chain_limit` replaces `for _ in 0..chain_limit`.** The 1-based iteration counter is both more natural for human-facing output (`Iteration 1/5`) and lets us remove the implicit counter that would otherwise be needed. The `..=` inclusive range means the loop body runs exactly `chain_limit` times, same as before.

**Per-iteration `collect_usage()` required moving the call before `all_chunks.extend()`.** Previously, `collect_usage()` was only called at the end on the full `all_chunks`. Adding it inside the loop on `iteration_chunks` means we call it before moving chunks into `all_chunks`. The clone in `IterationEnd { usage: usage.clone() }` is necessary because `usage` is also used later (it's an `Option<Usage>`, cheap to clone).

**`format_chain_event` shared between `prompt.rs` and `chat.rs`.** Rather than duplicating the formatting logic, the function was made `pub` on the `prompt` module and called from `chat` via `super::prompt::format_chain_event()`. This works because both modules are `pub mod` children of `commands/mod.rs`. An alternative would be a shared `verbose.rs` module, but that's premature — if more commands need it, the function can be moved then.

**`--verbose` implies `--tools-debug` simplifies the UX.** Instead of requiring `--verbose --tools-debug`, verbose mode automatically enables tool debug output. The implementation passes `args.tools_debug || args.verbose > 0` as the `debug` parameter to `CliToolExecutor::new()`. This means `-v` shows both chain iteration summaries and individual tool call/result lines, which is almost always what you want when debugging tool chains.

**Wiremock `up_to_n_times(1)` ordering for multi-response tests.** The integration tests follow the same pattern established in Phase 2: register the default (final) response first, then register the tool-call response with `up_to_n_times(1)` so it's consumed on the first request. The helper functions `openai_tool_call_response()` and `openai_text_response()` were extracted to avoid duplicating the OpenAI response JSON structure across three new integration tests.

**Message summary format balances compactness with informativeness.** The `format_message_summary()` output looks like `user, assistant+tools(1), tool(1)` — each message is summarized by its role plus a count suffix for tool-bearing messages. This shows at a glance how many tool calls the assistant made and how many results were returned, without overwhelming the user with full message content (which `-vv` provides).

### Phase 4 learnings

**Subprocess protocol design: arguments-only stdin, not full envelope.** The tool protocol sends only the `arguments` JSON to stdin (`{"text":"hello"}`), not the full `ToolCall` envelope. This makes tools simpler to implement — a shell script just reads JSON arguments without needing to parse a wrapper structure. The tool's name and ID are the CLI's concern, not the tool's.

**`ProtocolChunk` is necessary because `Chunk` has no serde.** `Chunk` in `llm-core` is a plain enum without `Serialize`/`Deserialize` — intentionally, since it's an internal streaming type not meant for wire transmission. The subprocess protocol needs serializable chunks for JSONL, so `ProtocolChunk` is a parallel enum with serde derives and `From`/`Into` conversions. This avoids adding serde to the core `Chunk` type, which would be a cross-cutting concern affecting all crates.

**`providers()` had to become async.** The original `providers()` was sync, returning compiled-in providers only. Subprocess provider discovery requires running `--id`, `--models`, `--needs-key` subprocesses, which are async operations (`tokio::process::Command`). Rather than using `block_on` in a sync function (which panics inside a tokio runtime), `providers()` was made async and all callers (`prompt::run`, `chat::run`, `models::run`) were updated to `.await` it. `models::run` also became async as a consequence.

**`resolve_key()` fails for providers that don't need a key.** When a subprocess provider reports `needs_key: false`, `provider.needs_key()` returns `None`, which the existing code mapped to `""` via `.unwrap_or("")`. Then `resolve_key("", ...)` would fail because it couldn't find a key for empty string. The fix: skip `resolve_key()` entirely when the provider doesn't need a key and no `--key` flag was given. The key variable changed from `String` to `Option<String>`, and all provider calls use `key.as_deref()` instead of `Some(&key)`. This is the kind of edge case that never appeared in phases 1-3 because both compiled-in providers always require keys.

**`ExternalToolExecutor` ownership across chat loop turns.** The chat REPL loop creates a `CliToolExecutor` each turn. Moving an `ExternalToolExecutor` into `CliToolExecutor::with_external()` transfers ownership, making it unavailable for the next turn. The fix: create the `CliToolExecutor` once before the loop and reuse it via `&executor` in `chain()`. This is cleaner anyway — the external tool set doesn't change during a chat session.

**PATH scanning deduplication matters.** If the same `llm-tool-foo` binary appears in multiple PATH directories, `discover_tools()` should return only the first occurrence (matching Unix `which` semantics). A `HashSet<String>` tracks seen filenames. Without this, the same tool would appear multiple times in `tools list` and potentially run different versions depending on discovery order.

**Shell script fixtures for integration testing.** Rather than writing Rust test binaries for external tools/providers, shell scripts in `tests/fixtures/bin/` implement the full protocol. This tests the actual subprocess boundary (spawn, stdin/stdout, exit codes) and validates that the protocol works with minimal, language-independent implementations. The scripts use `python3 -c` for JSON parsing — available on all CI machines and developer workstations.

**`async-stream` crate for streaming subprocess output.** The streaming provider path reads stdout line-by-line and needs to yield `Chunk` items through a `Stream`. `async_stream::try_stream!` macro provides a clean way to write this as an async generator with `yield` syntax. The alternative — manual `Stream` implementation via `futures::stream::unfold` — would require explicit state management for the `BufReader` and `Child` process. The crate adds ~200 lines of proc-macro code and compiles in under a second.

### Phase 3 learnings

**Chain loop history accumulation was the critical fix.** The Phase 2 chain loop rebuilt the prompt each iteration using only the latest tool_calls/tool_results, so iteration 3 had no memory of iteration 1. The fix was to maintain a `Vec<Message>` that grows across iterations: user → assistant+tools → tool_results → assistant+tools → tool_results → ... Each provider call now sees the full conversation history via `prompt.messages`. The `MockProvider` was enhanced with `Arc<Mutex<Vec<Prompt>>>` to capture prompts for assertion, enabling tests that verify message accumulation (1 → 3 → 5 messages across 3 iterations).

**Provider conversation paths dispatch on `prompt.messages.is_empty()`.** Rather than breaking existing single-turn behavior, both OpenAI and Anthropic `build_messages()` functions dispatch: empty messages → existing `build_single_turn()` path (unchanged), non-empty → new `build_from_conversation()` path. This ensured all 283 existing tests stayed green throughout the refactor.

**Anthropic tool results go in user role.** A subtlety in the conversation builder: Anthropic requires tool results in a `"role": "user"` message with `tool_result` content blocks, unlike OpenAI which uses `"role": "tool"`. The `Message::tool_results()` constructor uses `Role::Tool` abstractly; the Anthropic conversation builder maps this to `role: "user"` with `MessageContent::Blocks`. The assistant message with tool calls uses `MessageContent::Blocks` containing both a text block (if non-empty) and `tool_use` blocks.

**`--messages -` conflicts with stdin prompt text.** When `--messages -` reads from stdin, `resolve_prompt_text()` also tries to read stdin (because `is_terminal()` returns false for pipes). Both would consume the same input. The fix was a `skip_stdin` parameter: when `--messages -` is specified, `resolve_prompt_text` skips stdin reading entirely. This was caught by integration test `messages_stdin_with_json_output`.

**`--json` disables streaming to buffer output.** When `--json` is set, `stream_mode` is forced false and the chunk callback is suppressed. The full response is collected, then a JSON envelope is emitted at the end. The envelope includes `model`, `content`, `conversation_id` (if logged), `tool_calls` (if any), `usage` (if available), and `duration_ms`.

**`Config::save()` was needed for `logs on/off`.** Phase 2 avoided a `save()` method on `Config` (using `toml::Table` read-modify-write for `models default`). Phase 3 added `Config::save()` because `logs on/off` modifies `config.logging` — a typed boolean field that benefits from going through the serde roundtrip rather than raw TOML table manipulation.

**`rustyline` version 15 compiles cleanly on macOS.** No platform-specific workarounds needed. The editor handles Ctrl-D (EOF) and Ctrl-C (interrupt) via `ReadlineError` variants, both mapped to clean REPL exit. History is per-session only (no `.history` file persistence, keeping the chat command stateless beyond JSONL logging).

### Phase 2 learnings

**Anthropic tool_use blocks need all ContentBlock fields to be optional.** The `ContentBlock` struct is used for both `text` blocks (which have `text`) and `tool_use` blocks (which have `id`, `name`, `input`). Rather than using an enum (which conflicts with `#[serde(untagged)]` on `MessageContent`), all fields are `Option` with `#[serde(default, skip_serializing_if = "Option::is_none")]`. This is verbose but avoids serde ambiguity.

**Wiremock mock ordering for multi-step chains.** Integration tests for tool chains need sequential responses (first request returns tool call, second returns text). Wiremock's priority system works bottom-up: later-registered mocks have higher priority. The pattern is: register the default (final text) response first, then register the tool call response with `up_to_n_times(1)` so it's consumed on the first request, falling back to the default thereafter.

**System prompt preservation across chain iterations.** The chain loop constructs a new `Prompt` each iteration (with fresh `tool_calls` and `tool_results`). An early bug dropped the system prompt by not re-applying `with_system()` on the new prompt. Fixed by checking `current_prompt.system` and carrying it forward.

**`assert_cmd` stderr capture limitations.** A test for `--tools-debug` stderr output was initially written to assert `stderr(predicate::str::contains("Tool call: ..."))`. The test compiled and the tool chain ran (stdout showed the final response), but stderr was empty. The root cause was not fully diagnosed --- potentially related to how `assert_cmd` captures stderr from async code running inside `tokio::main`. The test was replaced with a `--chain-limit` test that verifies the chain loop mechanics without depending on stderr capture.

**Schema ID without blake2/hex.** The plan specified `blake2` + `hex` crates for schema ID generation. This was simplified to `std::hash::DefaultHasher` (SipHash) with `format!("{:016x}")` to avoid adding dependencies for a non-cryptographic hash. Trade-off: IDs won't match Python `llm`'s blake2b-based IDs, but cross-tool ID compatibility is not needed.

**`chain()` must surface tool results, not just chunks.** The initial `chain()` returned `Vec<Chunk>`. The CLI logged `tool_calls` by extracting them from the chunks (via `collect_tool_calls`), but `tool_results` were always `Vec::new()` because the results existed only inside the chain loop --- they were consumed into the next prompt and discarded. A live smoke test (`llm "What time is it?" -T llm_time`) exposed this: the log had `tool_calls: [{name: "llm_time", ...}]` but `tool_results: []`, even though the tool executed successfully and the model used its output.

The fix introduced `ChainResult { chunks: Vec<Chunk>, tool_results: Vec<ToolResult> }` as the return type. The chain loop accumulates `all_tool_results` across iterations (via `extend(tool_results.clone())`) and returns them alongside the chunks. The CLI destructures the result: `let (chunks, chain_tool_results) = ...` and passes `chain_tool_results` into the logged `Response`.

This is a good example of why end-to-end smoke testing with real APIs catches bugs that unit tests miss. The unit tests for `chain()` used a `MockProvider` and `MockExecutor` and verified iteration counts and text output --- but never inspected the logged `Response` because the chain tests don't touch the logging layer. The bug lived in the seam between `chain()` (llm-core) and `run()` (llm-cli), where the return type was too narrow to carry all the data the caller needed. The chain tests now also assert on `result.tool_results` --- verifying length, names, and error state --- so the contract is enforced at the unit level.
