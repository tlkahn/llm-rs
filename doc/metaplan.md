# LLM-RS Metaplan

> **Stack:** Rust 2024 edition + Cargo workspace | tokio + reqwest + clap + serde_json
> **Scope:** Reimplementation of [simonw/llm](https://github.com/simonw/llm) (v0.30) in Rust --- core prompting, conversations, tool calling, structured output, JSONL file logging, multi-provider. Embeddings, templates, and fragments deferred to future work.

---

## Design Philosophy

The Python `llm` is a monolith anchored to the Python packaging ecosystem (pluggy, pip, setuptools entry points). The Rust rewrite does not port that architecture. It decomposes the system into crates that each do one thing well, treats text and JSON as the universal interface, and replaces Python's dynamic plugin loading with subprocess-based extension.

| Unix Rule | How it manifests |
|-----------|-----------------|
| **Composition** | Stdin/stdout as primary I/O. `echo "Hello" \| llm` works. `llm logs list --json \| jq` works. JSONL streaming lets downstream tools process chunks as they arrive. |
| **Silence** | Response text goes to stdout. Errors, usage stats, and diagnostics go to stderr. No progress spinners, no "thinking..." messages. `--verbose` is opt-in. |
| **Modularity** | Six crates, each with one job. Provider crates know nothing about storage. Storage knows nothing about providers. The CLI composes them. |
| **Separation** | Policy lives in TOML config files. Mechanism lives in the binary. The binary doesn't embed defaults that belong in config. |
| **Extensibility** | New providers and tools are executables on `$PATH` speaking a JSON protocol on stdin/stdout. No shared libraries, no WASM runtime, no daemon. |
| **Least Surprise** | CLI mirrors familiar patterns --- `git`-style subcommands, `--json` for machine output, `-` for stdin, `-m` for model, `-s` for system prompt. |
| **Transparency** | `--verbose` logs HTTP requests, resolved config, model selection to stderr. Logs are JSONL files you can inspect with `cat`, `grep`, `jq` --- no special tools needed. |

Key divergences from the Python version:

- **No pluggy/pip plugin system.** Providers are either compiled in (feature flags) or invoked as subprocess executables (`llm-provider-*` convention).
- **No Pydantic.** Option validation uses `serde` with custom deserializers. Providers validate their own options.
- **Async-first.** Python maintains parallel sync and async class hierarchies (`Model`/`AsyncModel`, `Response`/`AsyncResponse`, `Conversation`/`AsyncConversation` --- six base classes total). Rust uses a single async `Provider` trait with `tokio::runtime::Runtime::block_on` for sync contexts.
- **TOML config.** Python scatters state across `keys.json`, `aliases.json`, `default_model.txt`, `options.json`, and YAML templates. Rust consolidates into `config.toml` + `keys.toml`.
- **JSONL log storage.** Python uses SQLite with 12 tables, 21 migrations, and relational joins. Rust uses one JSONL file per conversation --- append-only, human-readable, inspectable with standard Unix tools. The data is hierarchical (conversations containing responses containing tool calls), and JSON represents this directly without the flattening that SQL requires.

---

## Architecture Split

| Crate | Responsibility | Key dependencies |
|-------|---------------|-----------------|
| `llm-core` | Traits, types, streaming contracts, error types | `serde`, `thiserror`, `futures` |
| `llm-store` | JSONL file persistence: conversation log writes, queries, directory management | `serde_json`, `llm-core` |
| `llm-openai` | OpenAI provider (Chat + Completion APIs) | `reqwest`, `llm-core` |
| `llm-anthropic` | Anthropic provider (Messages API) | `reqwest`, `llm-core` |
| `llm-ollama` | Ollama local models (Chat API) | `reqwest`, `llm-core` |
| `llm-cli` | Binary entry point, all clap commands | `clap`, all above via features |
| `llm-wasm` | WASM library for browser/Obsidian plugin; JS-friendly API | `wasm-bindgen`, `llm-core`, `llm-openai` |
| `llm-python` | Python native module via PyO3; sync + streaming API | `pyo3`, `llm-core`, `llm-openai`, `llm-store` |

Dependency flow is strictly downward: `llm-cli`, `llm-wasm`, and `llm-python` are top-level entry points that compose the lower crates. Provider crates depend only on `llm-core`. `llm-store` depends only on `llm-core`. No cycles. `llm-wasm` and `llm-python` are excluded from default workspace builds (built with `wasm-pack` and `maturin` respectively).

---

## Core Skill Areas

| Area | What's needed |
|------|---------------|
| **Trait design** | `Provider` trait with async streaming (`Stream<Item=Result<Chunk>>`), model enumeration, key management |
| **SSE parsing** | Server-Sent Events over HTTP for streaming completions --- each provider uses slightly different conventions |
| **JSONL storage** | One file per conversation, append-only writes, directory-based listing, `grep`/`rg` for search |
| **CLI ergonomics** | `clap` derive macros, stdin detection (`is_terminal`), streaming stdout, subcommand groups |
| **Tool calling** | JSON Schema serialization, tool call response parsing, chain loop with depth limit |
| **Structured output** | JSON Schema generation from DSL, schema validation, multi-result schemas |
| **Subprocess IPC** | Spawn external executables, pipe JSON on stdin, read JSONL from stdout, handle errors |
| **Config management** | TOML parsing, XDG directory layout, key isolation with file permissions |

---

## Major Systems

### 1. Core Type System (`llm-core`)

Maps the Python class hierarchy in `llm/models.py` (~2165 lines) to Rust traits and structs.

**The `Provider` trait** replaces Python's four base classes (`Model`, `AsyncModel`, `KeyModel`, `AsyncKeyModel`):

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn models(&self) -> Vec<ModelInfo>;
    fn needs_key(&self) -> Option<&str> { None }
    fn key_env_var(&self) -> Option<&str> { None }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> Result<ResponseStream>;
}
```

Key management (Python's `_get_key_mixin`, `models.py:1747-1777`) becomes a default method or a standalone function that resolves keys in order: explicit argument, config file, environment variable.

**Core types:**

```
Prompt       { text, system, attachments, schema, tools, tool_results, options }
Chunk        { Text(String) | ToolCallStart{name,id} | ToolCallDelta{content} | Usage{input,output,details} | Done }
Response     { id, chunks, model, usage, tool_calls, duration, json }
Conversation { id, name, model_id, responses }
Attachment   { mime_type, source: Path|Url|Bytes }
Tool         { name, description, input_schema }
ToolCall     { name, arguments, tool_call_id }
ToolResult   { name, output, tool_call_id, error }
Usage        { input, output, details }
ModelInfo    { id, can_stream, supports_tools, supports_schema, attachment_types }
Options      HashMap<String, serde_json::Value>   // provider-validated, not schema-enforced
```

**The chain loop** (Python's `ChainResponse`, `models.py:1621-1675`) becomes a function:

```
fn chain(provider, prompt, tools, limit) -> ResponseStream
    loop:
        response = provider.execute(prompt_with_tool_results)
        tool_calls = response.tool_calls()
        if tool_calls.is_empty() or iteration >= limit: break
        results = execute_tool_calls(tool_calls, tools)
        prompt = prompt.with_tool_results(results)
```

### 2. Provider Implementations (`llm-openai`, `llm-anthropic`, `llm-ollama`)

Each crate implements the `Provider` trait. Reference implementation: Python's `llm/default_plugins/openai_models.py`.

**Critical complexity: SSE delta accumulation.** OpenAI streams tool call arguments as deltas across multiple SSE chunks. The Python code (openai_models.py, lines 798-831) accumulates these by index. Each provider crate must implement its own SSE parser because the wire formats differ:

- **OpenAI:** `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"..."}}]}}]}`
- **Anthropic:** `event: content_block_delta` with `{"type":"input_json_delta","partial_json":"..."}`
- **Ollama:** OpenAI-compatible format (mostly)

Provider crates also handle:
- Model enumeration (hardcoded list + optional user-configured extra models)
- Option validation (temperature, max_tokens, etc. --- provider-specific)
- Error mapping (rate limits, auth failures, server errors)
- Non-streaming fallback (single response instead of SSE)

### 3. Storage Layer (`llm-store`)

The Python storage uses SQLite with 12 tables, 21 migrations, and relational joins to normalize inherently hierarchical data (conversations → responses → tool calls → tool results). The Rust rewrite replaces this with JSONL files --- one per conversation, append-only, with all data denormalized into self-contained JSON records.

**Why JSONL over SQLite:**

- The data is hierarchical, not relational. Tool calls, attachments, and tool results belong to a response and are never queried independently. JSON nests them naturally; SQL flattens them into junction tables.
- JSONL is the tool's lingua franca everywhere else (subprocess IPC, streaming output, `--json` flag). Using it for storage creates one consistent format throughout the system.
- Standard Unix tools (`cat`, `grep`, `jq`, `head`, `tail`, `wc`, `diff`) work directly on the files. No special tooling needed for inspection or debugging.
- No schema migrations. New fields are added freely; old readers ignore unknown keys via `#[serde(default)]`.
- Eliminates the `rusqlite` dependency (bundled SQLite: +3-5s compile time, +1-2MB binary size).

**File layout:**

```
$XDG_DATA_HOME/llm/logs/
    {conversation_id}.jsonl         # one file per conversation
```

**Conversation file format:**

```jsonl
{"v":1,"type":"conversation","id":"01J5A...","model":"gpt-4o","name":null,"created":"2026-04-03T12:00:00Z"}
{"type":"response","id":"01J5B...","prompt":"Hello","system":null,"response":"Hi there!","model":"gpt-4o","options":{},"input_tokens":5,"output_tokens":8,"token_details":null,"duration_ms":230,"datetime":"2026-04-03T12:00:01Z"}
{"type":"response","id":"01J5C...","prompt":"Search for X","response":"Found it.","model":"gpt-4o","tool_calls":[{"name":"search","arguments":{"q":"X"},"tool_call_id":"tc_1"}],"tool_results":[{"name":"search","output":"result...","tool_call_id":"tc_1"}],"attachments":[{"type":"image/png","path":"/tmp/img.png"}],"schema":{"type":"object","properties":{"answer":{"type":"string"}}},"schema_id":"b3a8...","input_tokens":50,"output_tokens":30,"duration_ms":1200,"datetime":"2026-04-03T12:01:00Z"}
```

The first line is conversation metadata (version, ID, initial model). Subsequent lines are responses, each self-contained with all associated data (tool calls, tool results, attachments, schema) denormalized inline.

**Query strategies:**

| Operation | Implementation |
|-----------|---------------|
| List recent N | Read `logs/` directory sorted by mtime, read first line of each for metadata |
| Continue by ID (`--cid`) | Open `{id}.jsonl` by filename --- O(1) lookup |
| Continue last (`-c`) | Most recently modified file in `logs/` |
| Full-text search (`-q`) | `grep` / `rg` across files (fast for typical scale of <10k conversations) |
| Filter by model | Scan first line of each file (model field in conversation metadata) |
| JSON output (`--json`) | Stream lines directly --- they're already JSON |

**Schema storage:** Schemas are stored inline in the response record. The `schema_id` field (BLAKE2b hash of schema content) enables deduplication at the application level. `llm schemas list` scans log files to collect unique schemas by hash.

**`log_response` function:** The Python `log_to_db` (models.py:828-1017, ~190 lines) writes across 3-8 tables in a transaction. The Rust equivalent serializes one JSON line and appends it to the conversation file --- a single `serde_json::to_writer` call followed by a newline and flush.

**Migration from Python `llm`:** Provide `llm import --from-sqlite <path>` to read an existing Python `llm` SQLite database and write JSONL conversation files. This is a one-time conversion, not an ongoing compatibility layer.

### 4. CLI Layer (`llm-cli`)

Built with `clap` derive macros. The Python CLI (`llm/cli.py`, ~4050 lines) uses Click with `click-default-group`. Command structure:

```
llm [prompt]                      # default subcommand (like Python's click-default-group)
llm prompt <text>                 # send a prompt
    -m/--model <id>               # model selection
    -s/--system <text>            # system prompt
    -a/--attachment <path|url>    # file/URL attachment
    --at/--attachment-type <mime> # explicit MIME type
    -T/--tool <name>              # enable a tool
    --td/--tools-debug            # show tool execution details (to stderr)
    --ta/--tools-approve          # prompt before each tool call
    --cl/--chain-limit <n>        # max tool chain depth (default 5)
    -o/--option <key=value>       # model-specific option
    --schema <json|file|dsl>      # structured output schema
    --schema-multi                # schema produces array of results
    -c/--continue                 # continue most recent conversation
    --cid <id>                    # continue specific conversation
    --no-stream                   # disable streaming
    -n/--no-log                   # don't log to file
    --key <key>                   # explicit API key
    -u/--usage                    # show token usage (to stderr)
    -x/--extract                  # extract first fenced code block
    --xl/--extract-last           # extract last fenced code block
    --json                        # JSON output envelope
    --async                       # run asynchronously (for scripting)
llm chat                          # interactive REPL (rustyline)
llm logs list                     # list logged prompts/responses
    --json/--nl                   # output format
    -m/--model <id>               # filter by model
    -q/--query <text>             # full-text search
    --limit/--offset              # pagination
    -u/--usage                    # include token usage
    -r/--response                 # response text only
    -x/--extract                  # extract code blocks
llm logs path                     # print logs directory path
llm logs status                   # show logging on/off
llm logs on|off                   # toggle logging
llm keys list|path|get|set        # API key management
llm models list                   # list available models
    --tools/--schemas             # filter by capability
    -q/--query <text>             # search models
llm models default [model]        # get/set default model
llm schemas list|show             # schema management
llm schemas dsl <input>           # convert DSL to JSON Schema
llm tools list                    # list available tools
llm aliases set|list|remove|path  # model alias management
llm options set|get|list|clear    # per-model option defaults
llm plugins                       # list compiled + subprocess providers
```

**Unix I/O conventions:**
- Response text streams to stdout as chunks arrive. No buffering to "complete" before output.
- Errors, usage stats (`-u`), tool debug (`--td`), and verbose logs (`--verbose`) go to stderr.
- When stdin is a pipe (not a terminal), it becomes the prompt text: `cat file.txt | llm "summarize"` appends stdin to the prompt.
- `--json` wraps output in a JSON envelope with metadata (model, usage, duration).

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error (tool failure, unexpected state) |
| 2 | Configuration error (missing key, bad config file, unknown model) |
| 3 | Provider/network error (API failure, timeout, rate limit) |

### 5. Tool Execution System

The Python tool system (`models.py:133-332`) includes `Tool`, `Toolbox`, `ToolCall`, `ToolResult`, `ToolOutput`, `CancelToolCall`. For the Rust rewrite, tools come from two sources:

**Built-in tools** (compiled into the binary):
- `llm_version` --- returns installed version
- `llm_time` --- returns current UTC and local time

**Subprocess tools** (`llm-tool-*` executables on `$PATH`):

```
# Discovery
llm-tool-search --schema
→ {"name":"search","description":"Web search","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}

# Invocation
echo '{"query":"rust async streams"}' | llm-tool-search
→ {"output":"Results: ..."}

# Error
echo '{"query":""}' | llm-tool-search
→ exit code 1, stderr: "query must not be empty"
```

Convention: tool name is derived from executable name by stripping the `llm-tool-` prefix.

**Tool approval flow** (`--tools-approve`): Before executing each tool call, print the tool name and arguments to stderr and prompt the user for y/n confirmation. This mirrors Python's `before_call` callback.

**Chain loop:** The CLI orchestrates the chain (system 1's chain function). Each iteration: send prompt to provider, collect tool calls from response, execute tools (built-in or subprocess), feed results back. Stop when no tool calls remain or `--chain-limit` is reached.

### 6. Structured Output (Schemas)

The Python schema system spans `llm/utils.py` (schema DSL, ~60 lines), `llm/migrations.py` (migration m014), and `llm/cli.py` (schema resolution, ~40 lines).

**Schema DSL** (Python `utils.py:schema_dsl`, concise shorthand for JSON Schema):

```
"name str, age int:The person's age, active bool"
→
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "description": "The person's age"},
    "active": {"type": "boolean"}
  },
  "required": ["name", "age", "active"]
}
```

Types: `str` (string), `int` (integer), `float` (number), `bool` (boolean). Descriptions follow a `:` after the type.

`--schema-multi` wraps the schema in `{"type":"array","items":<schema>}` for extracting multiple results.

**Schema resolution order** (`--schema` argument):
1. If it parses as JSON, use it as a JSON Schema directly
2. If it's a file path that exists, read and parse as JSON
3. If it matches a known schema ID (BLAKE2b hash), scan logs to find and load it
4. Otherwise, parse as DSL

Schemas are stored inline in response records. The `schema_id` field (BLAKE2b hash of JSON content) enables lookup and deduplication across log files.

### 7. Configuration System

Python scatters config across 6+ files in `click.get_app_dir("io.datasette.llm")`. The Rust rewrite consolidates into XDG-compliant paths:

```
$XDG_CONFIG_HOME/llm/           # defaults to ~/.config/llm/
    config.toml                 # main configuration
    keys.toml                   # API keys (0600 permissions)

$XDG_DATA_HOME/llm/             # defaults to ~/.local/share/llm/
    logs/                       # one JSONL file per conversation
        {conversation_id}.jsonl
```

**`config.toml`:**

```toml
default_model = "gpt-4o-mini"
logging = true

[aliases]
claude = "claude-sonnet-4-20250514"
fast = "gpt-4o-mini"
smart = "gpt-4o"

[options.gpt-4o]
temperature = 0.7

[providers.openai]
# extra_models, base_url overrides, etc.
```

**`keys.toml`** (separate file, restricted permissions):

```toml
openai = "sk-..."
anthropic = "sk-ant-..."
ollama = ""  # no key needed, but entry exists for completeness
```

**Key resolution order** (matching Python's `_get_key_mixin.get_key`, models.py:1751-1777):
1. `--key` CLI flag (explicit)
2. `keys.toml` entry matching provider's `needs_key` value
3. Environment variable (e.g., `OPENAI_API_KEY`)
4. Error with actionable message: `"No key found --- set one with 'llm keys set openai' or export OPENAI_API_KEY"`

**Environment variables:**

| Variable | Purpose |
|----------|---------|
| `LLM_USER_PATH` | Override config/data directory (matches Python for migration) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `LLM_DEFAULT_MODEL` | Override default model |

### 8. Library Targets (WASM + Python)

The core crates (`llm-core`, `llm-openai`) are designed to compile for multiple targets beyond the native CLI. Two additional entry-point crates provide library interfaces:

**Platform abstraction strategy:**

The key insight is that `llm-core` production code has zero tokio usage (only `#[tokio::test]` in tests), and `llm-openai` has exactly 3 tokio-specific lines in its streaming path. The refactoring is surgical:

- **`ResponseStream` Send bound** (`llm-core/src/stream.rs`): cfg-gated for wasm32. On native, the stream requires `Send` (multi-threaded tokio runtime). On wasm32, `Send` is dropped (single-threaded).
  ```rust
  #[cfg(not(target_arch = "wasm32"))]
  pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<Chunk, LlmError>> + Send>>;
  #[cfg(target_arch = "wasm32")]
  pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<Chunk, LlmError>>>>;
  ```

- **`Provider` trait bounds** (`llm-core/src/provider.rs`): cfg-gated. Native uses `#[async_trait] trait Provider: Send + Sync`. WASM uses `#[async_trait(?Send)] trait Provider` (no threading).

- **Streaming channel** (`llm-openai/src/provider.rs`): Replace `tokio::sync::mpsc` with `futures::channel::mpsc` (works on all platforms). `futures::channel::mpsc::Receiver` already implements `Stream`, eliminating the `tokio_stream::wrappers::ReceiverStream` wrapper. Only the spawn call is cfg-gated:
  ```rust
  #[cfg(not(target_arch = "wasm32"))]
  tokio::spawn(parse_future);
  #[cfg(target_arch = "wasm32")]
  wasm_bindgen_futures::spawn_local(parse_future);
  ```

- **Dependency gating**: `llm-core` removes `tokio` from `[dependencies]` (keep in `[dev-dependencies]`). `llm-openai` makes `tokio` a `cfg(not(wasm32))` dependency, adds `wasm-bindgen-futures` as `cfg(wasm32)` dependency, removes `tokio-stream` entirely.

**WASM crate (`llm-wasm`):**

Built with `wasm-pack build --target bundler` (for Obsidian/webpack) or `--target web` (direct browser use). Exports a JS-friendly API via `wasm-bindgen`:

```rust
#[wasm_bindgen]
pub struct LlmClient { /* provider, model, api_key */ }

#[wasm_bindgen]
impl LlmClient {
    #[wasm_bindgen(constructor)]
    pub fn new(api_key: &str, model: &str) -> Self;
    pub fn new_with_base_url(api_key: &str, model: &str, base_url: &str) -> Self;
    pub async fn prompt(&self, text: &str) -> Result<String, JsError>;
    pub async fn prompt_streaming(&self, text: &str, callback: &js_sys::Function) -> Result<String, JsError>;
}
```

TypeScript consumer (Obsidian plugin):
```typescript
import init, { LlmClient } from '@llm-rs/wasm';
await init();
const client = new LlmClient('sk-...', 'gpt-4o');
const response = await client.prompt('Hello');
// Streaming:
await client.promptStreaming('Hello', (chunk) => console.log(chunk));
```

No storage, no config, no key management --- purely stateless. The Obsidian plugin handles persistence via its own vault API. reqwest 0.12 auto-detects wasm32 and uses web-sys `fetch` under the hood.

npm package: `@llm-rs/wasm`.

**Python crate (`llm-python`):**

Built with `maturin build`. Exports a Python-friendly API via PyO3:

```python
import llm_rs

client = llm_rs.LlmClient("sk-...", "gpt-4o-mini")
response = client.prompt("Hello, world!")
print(response)

for chunk in client.prompt_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

The crate owns a `tokio::Runtime` for async-to-sync bridging. Streaming uses `std::sync::mpsc` to bridge from the async stream to a Python iterator (`ChunkIterator` with `__iter__`/`__next__`). Optionally includes `llm-store` for log persistence when `log_dir` is provided.

PyPI package: `llm-rs`, import as `import llm_rs`.

---

## Subprocess Provider Protocol

New LLM providers can be added at runtime without recompilation. Any executable named `llm-provider-<name>` on `$PATH` that implements this protocol is automatically discovered.

### Discovery

```bash
llm-provider-mistral --list-models
```

Stdout (JSON array):

```json
[
  {"id": "mistral-large", "can_stream": true, "supports_tools": true, "supports_schema": false},
  {"id": "mistral-small", "can_stream": true, "supports_tools": false, "supports_schema": false}
]
```

### Execution

```bash
echo '<request json>' | llm-provider-mistral --model mistral-large --stream
```

Stdin (JSON object):

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "tools": [{"name": "calc", "description": "...", "input_schema": {...}}],
  "schema": {"type": "object", "properties": {...}},
  "options": {"temperature": 0.7}
}
```

Stdout (JSONL, one object per line, streamed):

```jsonl
{"type":"text","content":"The answer"}
{"type":"text","content":" is 4."}
{"type":"tool_call","name":"calc","arguments":{"expr":"2+2"},"id":"tc_1"}
{"type":"usage","input":15,"output":8}
{"type":"done"}
```

Without `--stream`, stdout is a single JSON object:

```json
{"type":"response","content":"The answer is 4.","tool_calls":[],"usage":{"input":15,"output":8}}
```

Errors: exit code 1, human-readable message to stderr.

---

## Biggest Risks

1. **SSE delta accumulation.** Each provider streams tool call arguments differently. OpenAI sends argument deltas indexed by tool call position across multiple SSE chunks (Python openai_models.py:798-831). Getting this wrong produces corrupted JSON arguments. Mitigate: write integration tests with recorded HTTP cassettes (the Python project has `tests/cassettes/` as precedent).

2. **Subprocess provider latency.** Python tools execute in-process. Subprocess tools add per-invocation startup cost (~5-50ms). In tight chain loops (5+ tool calls), this compounds. Mitigate: keep commonly used tools built-in; subprocess is for third-party extension. Consider connection pooling for long-running tool processes in future work.

3. **Conversation serialization.** Each provider needs different message array formats. OpenAI uses `messages[].content`, Anthropic uses `messages[].content` as an array of blocks, Ollama is OpenAI-compatible. Building the message array from conversation history (Python openai_models.py `build_messages`, ~75 lines) is easy to get subtly wrong. Mitigate: test with multi-turn conversations that include tool calls and attachments.

4. **JSONL format evolution.** Once conversation files exist in the wild, changing the record format risks breaking existing logs. Mitigate: include a `"v":1` version field in the conversation header line. Future format changes bump the version and the reader handles both. Keep the format minimal --- add fields freely (readers ignore unknowns via `#[serde(default)]`), but avoid renaming or removing fields.

5. **`prompt` command complexity.** The Python `prompt` command is ~580 lines with 30+ flags, complex resolution chains (schema, template, fragment, attachment), and multiple output modes. Rushing this risks a buggy CLI. Mitigate: implement flags incrementally across phases; test each flag in isolation.

6. **wasm32 reqwest streaming.** reqwest 0.12 supports wasm32 via web-sys fetch, but streaming SSE via `ReadableStream` may have edge cases (chunking boundaries, backpressure) that differ from native HTTP. Mitigate: test `llm-wasm` with wasm-pack in a browser environment against a mock SSE server; verify chunk boundaries match native behavior.

---

## Phased Approach

### Phase 1 (v0.1) --- Core Loop

**Goal:** `echo "Hello" | llm` works end-to-end. Streams to stdout, logs to JSONL. Core library also compiles to WASM and Python native module.

- `llm prompt` with `-m`, `-s`, `--no-stream`, `-n/--no-log`, `--key`, `-u/--usage`
- Stdin piping (detect terminal vs pipe)
- OpenAI provider (streaming + non-streaming)
- JSONL log storage (one file per conversation, append-only responses)
- `llm keys set/get/list/path`
- `llm models list/default`
- `llm logs list` (basic: list recent, `--json`, `-r/--response`)
- Config loading (`config.toml` + `keys.toml`)
- Exit codes
- WASM library target (`llm-wasm`) for browser/Obsidian plugin use --- self-contained HTTP via fetch, no storage, JS Promise-based API
- Python native module (`llm-python`) via PyO3/maturin --- sync + streaming API with optional log storage
- Anthropic provider (streaming + non-streaming)

### Phase 2 (v0.2) --- Tools and Structured Output

- OpenAI provider: tool calling in requests (`tools` field, `tool_choice`) + SSE parsing for tool call deltas
- Anthropic provider: tool calling in requests (`tools` field) + SSE parsing for `tool_use` content blocks
- OpenAI + Anthropic: structured output (`response_format` / schema in request)
- Schema DSL parser (`"name str, age int"` → JSON Schema)
- Chain loop function in `llm-core` (iterate provider → tool execution → feed results, up to `--chain-limit`)
- Built-in tools in `llm-cli`: `llm_version`, `llm_time`
- CLI flags: `-T/--tool`, `--chain-limit`, `--tools-debug`, `--tools-approve`
- CLI flags: `--schema`, `--schema-multi`; schema resolution (JSON literal / file / schema ID / DSL)
- `llm tools list`, `llm schemas list/show/dsl` commands
- Tool/tool call/tool result persistence in JSONL response records (store layer already supports this)

### Phase 3 (v0.3) --- Conversations & Multi-turn (COMPLETE)

- `Message`/`Role` core types with conversation history accumulation in chain loop
- Provider conversation paths (OpenAI + Anthropic multi-turn message building)
- `-c/--continue`, `--cid` for conversation continuation from JSONL logs
- `--messages` flag for JSON message input, `--json` for structured output envelope
- `llm chat` (interactive REPL with `rustyline`)
- `llm logs` full feature set (path, status, on/off, `-m` model filter, `-q` text search, `-u` usage)
- Store: `reconstruct_messages()` for conversation reconstruction from stored responses

### Phase 4 (v0.4) --- Subprocess Extensibility & More

- `llm-tool-*` subprocess tool protocol (discovery via `--schema`, invocation via stdin/stdout JSON)
- `llm-provider-*` subprocess protocol (discovery + execution)
- `llm plugins` (list compiled providers + discovered subprocess providers/tools)
- `--verbose` flag (HTTP request logging, config resolution tracing)
- Shell completions (`clap_complete`)
- Ollama provider
- `llm aliases set/list/remove/path`
- `-o/--option`, `llm options set/get/list/clear`
- `-a/--attachment`, `--at/--attachment-type`
- `-x/--extract`, `--xl/--extract-last`

### Iteration Strategy

Each phase is a vertical slice delivering a usable tool. Within each phase, work bottom-up through the crate layers using TDD (red-green-refactor) at each step. The phase boundary is where you dogfood the result before planning the next.

**Phase 1 inner loop (example):**

| Step | Crate | What to build | TDD focus |
|------|-------|---------------|-----------|
| 1 | `llm-core` | `Prompt`, `Chunk`, `Response`, `Usage`, `Provider` trait | Unit tests: type construction, stream collection |
| 2 | `llm-openai` | OpenAI streaming + non-streaming | Tests with recorded HTTP cassettes |
| 3 | `llm-store` | Conversation JSONL file I/O, `log_response` | Tests with tmpdir log files |
| 4 | `llm-core` | `Config`, `KeyStore`, XDG path resolution | Unit tests: key resolution order, config parsing |
| 5 | `llm-cli` | `prompt`, `keys`, `models`, `logs list` commands | Integration tests: assert stdout/stderr/exit code |
| 6a | `llm-core` | cfg-gate `ResponseStream` Send bound, `Provider` Send+Sync bounds for wasm32 | Existing 188 tests pass; `cargo check -p llm-core --target wasm32-unknown-unknown` |
| 6b | `llm-openai` | Switch to `futures::channel::mpsc`, cfg-gate spawn for wasm32 | Existing streaming tests pass; `cargo check -p llm-openai --target wasm32-unknown-unknown` |
| 6c | `llm-wasm` | wasm-bindgen exports: `LlmClient`, `prompt()`, `prompt_streaming()` | `wasm-pack build`; manual test in browser/Node.js |
| 6d | `llm-python` | PyO3 exports: `LlmClient`, `prompt()`, `prompt_stream()` iterator | `maturin develop && python -c "import llm_rs"` |

Each step is a self-contained TDD cycle: write failing tests that describe the contract, make them pass, refactor. Steps 1-4 are unit/component tests. Step 5 is integration tests that exercise the full stack. Steps 6a-6b are refactoring existing crates for wasm32 compatibility (all existing tests must stay green). Steps 6c-6d are new crates with their own build toolchains.

**Phase 2 inner loop:**

Core types (`Tool`, `ToolCall`, `ToolResult`, `Chunk::ToolCallStart/Delta`, `collect_tool_calls()`, `Prompt` builders, `Response` fields) and store persistence already exist from Phase 1. Phase 2 builds on top of them.

| Step | Crate | What to build | TDD focus |
|------|-------|---------------|-----------|
| 1 | `llm-core` | Schema DSL parser (`schema.rs`): `"name str, age int:desc"` → JSON Schema | Unit tests: DSL strings → JSON Schema, types (str/int/float/bool), descriptions, edge cases, parse errors |
| 2 | `llm-openai` | Tool calling: add `tools` + `tool_choice` to request; parse `tool_calls` deltas in SSE (streaming) and full response (non-streaming) | Cassette tests: single tool call, multiple tool calls, streaming delta accumulation, non-streaming extraction |
| 3 | `llm-anthropic` | Tool calling: add `tools` to request; parse `tool_use` content blocks in SSE (streaming) and full response (non-streaming); send `tool_result` content blocks | Cassette tests: same scenarios as step 2 but with Anthropic wire format |
| 4 | `llm-openai` + `llm-anthropic` | Structured output: send schema in request (`response_format` for OpenAI, tool-based for Anthropic); parse structured JSON response | Cassette tests: schema request → JSON response, `--schema-multi` wrapping |
| 5 | `llm-core` | Chain loop function: `chain(provider, prompt, tools, limit) → ResponseStream` — iterate execute → collect tool calls → run tools → feed results back, stop when no calls or limit reached | Unit tests with mock provider: single iteration, multi-iteration, chain limit, no tool calls exits immediately |
| 6 | `llm-cli` | Built-in tools (`llm_version`, `llm_time`); tool registry (enumerate built-ins); `llm tools list` command | Integration tests: `llm tools list` output, tool schema shape |
| 7 | `llm-cli` | `-T/--tool` flag, `--chain-limit`, `--tools-debug` (stderr diagnostics), `--tools-approve` (interactive y/n) | Integration tests with wiremock: tool flag → provider receives tools in request; chain loop executes; debug output on stderr |
| 8 | `llm-cli` | `--schema` flag (JSON literal / file / schema ID / DSL), `--schema-multi`; schema resolution chain; `llm schemas list/show/dsl` commands | Integration tests: each resolution path, DSL → structured response, `schemas dsl` output |

Each step follows strict TDD: write failing tests first (red), implement until they pass (green), then refactor. All 225+ existing tests must stay green at every step. `cargo test --workspace` and `cargo clippy --workspace` gate each commit.

---

## Recommended Starting Point

Begin with Phase 1. Set up the Cargo workspace, get `echo "Hello" | llm` producing a streamed response from OpenAI, and logging it to a JSONL file. This validates the entire vertical slice: CLI parsing, config loading, HTTP streaming, file writes, stdin/stdout I/O.

```
llm-rs/
  Cargo.toml                        # workspace root
  Cargo.lock
  doc/
    metaplan.md                     # this file

  crates/
    llm-core/
      Cargo.toml
      src/
        lib.rs                      # re-exports
        provider.rs                 # Provider trait, ModelInfo
        types.rs                    # Prompt, Response, Conversation, Attachment, Usage
        tools.rs                    # Tool, ToolCall, ToolResult
        stream.rs                   # Chunk enum, ResponseStream type alias
        config.rs                   # Config, KeyStore, XDG paths
        error.rs                    # LlmError (thiserror)
        schema.rs                   # Schema DSL parser, JSON Schema types
        ulid.rs                     # Monotonic ULID generator

    llm-store/
      Cargo.toml
      src/
        lib.rs
        logs.rs                     # write/read conversation JSONL files
        query.rs                    # log listing, search, filtering
        import.rs                   # import from Python llm SQLite database

    llm-openai/
      Cargo.toml
      src/
        lib.rs                      # OpenAiProvider implementing Provider
        chat.rs                     # streaming SSE parser, message builder
        types.rs                    # OpenAI request/response types
        options.rs                  # temperature, max_tokens, etc.

    llm-anthropic/
      Cargo.toml
      src/
        lib.rs                      # AnthropicProvider implementing Provider
        messages.rs                 # Messages API, SSE parser
        types.rs
        options.rs

    llm-ollama/
      Cargo.toml
      src/
        lib.rs                      # OllamaProvider implementing Provider
        chat.rs
        types.rs
        options.rs

    llm-cli/
      Cargo.toml                    # feature-gated provider deps
      src/
        main.rs                     # entry point, tokio::main
        app.rs                      # top-level clap App
        commands/
          mod.rs
          prompt.rs                 # the big one (~30 flags)
          chat.rs                   # interactive REPL
          logs.rs                   # logs {list,path,status,on,off}
          keys.rs                   # keys {list,path,get,set}
          models.rs                 # models {list,default}
          schemas.rs                # schemas {list,show,dsl}
          tools.rs                  # tools {list}
          aliases.rs                # aliases {set,list,remove,path}
          options.rs                # options {set,get,list,clear}
          plugins.rs                # plugins (list compiled + subprocess)
        output.rs                   # formatting: JSON, JSONL, tables, plain text
        resolve.rs                  # attachment + schema resolution
        interactive.rs              # chat REPL (rustyline)
        subprocess.rs               # subprocess provider/tool protocol

    llm-wasm/
      Cargo.toml                    # crate-type = ["cdylib"], wasm-bindgen deps
      src/
        lib.rs                      # wasm-bindgen exports: LlmClient, prompt, streaming

    llm-python/
      Cargo.toml                    # crate-type = ["cdylib"], pyo3 deps
      pyproject.toml                # maturin build config
      src/
        lib.rs                      # PyO3 module: LlmClient, ChunkIterator

  tests/
    cassettes/                      # recorded HTTP responses for providers
```

**Workspace `Cargo.toml`:**

```toml
[workspace]
resolver = "2"
members = ["crates/*"]
exclude = ["crates/llm-wasm", "crates/llm-python"]  # built with wasm-pack / maturin

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
reqwest = { version = "0.12", features = ["stream", "json"] }
rusqlite = { version = "0.32", features = ["bundled"], optional = true }  # only needed for `llm import --from-sqlite`
thiserror = "2"
futures = "0.3"
clap = { version = "4", features = ["derive"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
pyo3 = { version = "0.23", features = ["extension-module"] }
```

**`llm-cli/Cargo.toml` feature flags:**

```toml
[features]
default = ["openai"]
openai = ["dep:llm-openai"]
anthropic = ["dep:llm-anthropic"]
ollama = ["dep:llm-ollama"]
all = ["openai", "anthropic", "ollama"]
```

Users compile with `--features all` for the full suite or `--no-default-features --features ollama` for a minimal local-only binary.

**Library target build commands:**

```bash
# WASM (for Obsidian/browser):
wasm-pack build crates/llm-wasm --target bundler   # npm-ready package
wasm-pack build crates/llm-wasm --target web        # direct browser use

# Python (via maturin):
cd crates/llm-python && maturin develop             # install to current venv
cd crates/llm-python && maturin build --release     # build wheel for distribution
```

---

## Critical Python Source Files

Reference these during implementation:

| File | Lines | What to extract |
|------|-------|----------------|
| `llm/models.py` | ~2165 | Type definitions, streaming contracts, chain loop, tool execution, `log_to_db` |
| `llm/cli.py` | ~4050 | Command structure, flag definitions, prompt orchestration, log formatting |
| `llm/default_plugins/openai_models.py` | ~950 | `build_messages`, SSE parsing, delta tool-call accumulation, option mapping |
| `llm/migrations.py` | ~420 | Reference for `llm import --from-sqlite` (understand source schema to read Python databases) |
| `llm/utils.py` | ~390 | Schema DSL parser, Fragment type, ULID generator, code block extraction |
| `llm/errors.py` | ~15 | Error types (`ModelError`, `NeedsKeyException`) |
| `llm/__init__.py` | ~350 | Public API surface, `get_model`, `get_key`, `user_dir` |
| `tests/conftest.py` | ~488 | MockModel pattern, test fixtures, HTTP mocking approach |
