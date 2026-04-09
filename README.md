# llm-rs

A Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) --- a CLI tool for interacting with Large Language Models from the terminal.

## Usage

```bash
# Send a prompt (streams to stdout)
echo "Hello" | llm

# Positional text works too
llm "Explain monads in one sentence" -m claude-sonnet-4-6

# Specify model and system prompt
llm "What is 2+2?" -m gpt-4o -s "Answer only with the number"

# Use Anthropic models
llm "Hello" -m claude-sonnet-4-6

# Disable streaming
llm "Hello" --no-stream

# Show token usage on stderr
llm "Hello" -u

# Skip logging this prompt
llm "Hello" -n
```

### Tool calling

Built-in tools let the model call functions during a conversation. The CLI manages the chain loop automatically --- it sends tool calls to the executor, feeds results back, and repeats until the model responds with text.

```bash
# Enable a built-in tool
llm "What time is it?" -T llm_time

# Multiple tools
llm "What version are you and what time is it?" -T llm_version -T llm_time

# Limit chain iterations (default: 5)
llm "Do something" -T llm_version --chain-limit 3

# Debug mode: show tool calls/results on stderr
llm "What version?" -T llm_version --tools-debug

# List available built-in tools
llm tools list
```

Available built-in tools:
- `llm_version` --- returns the CLI version
- `llm_time` --- returns current UTC and local time with timezone

### External tools

Any executable on `$PATH` named `llm-tool-*` is automatically discovered and usable with `-T`. External tools can be written in any language.

```bash
# List all tools (built-in + external)
llm tools list

# Use an external tool
llm "Make this loud: hello" -T upper -m gpt-4o

# Mix built-in and external tools
llm "What time is it, and shout it" -T llm_time -T shout
```

Writing an external tool requires two things:

1. **Schema**: respond to `--schema` with JSON describing the tool:
   ```bash
   $ llm-tool-upper --schema
   {"name":"upper","description":"Uppercase text","input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}
   ```

2. **Execution**: read arguments JSON from stdin, write result to stdout:
   ```bash
   $ echo '{"text":"hello"}' | llm-tool-upper
   HELLO
   ```

Exit 0 means success (stdout = output). Non-zero means error (stderr = error message). Default timeout: 30 seconds.

### External providers

Any executable on `$PATH` named `llm-provider-*` extends llm-rs with new model providers. External providers can serve models from Ollama, llama.cpp, or any custom backend.

```bash
# Models from external providers appear alongside built-in ones
llm models list

# Use a model from an external provider
llm "Hello" -m llama3

# See all providers and tools
llm plugins list
```

Writing an external provider requires metadata flags and a JSON stdin/stdout protocol:

- `--id` --- print the provider name (e.g. `ollama`)
- `--models` --- print JSON array of model metadata
- `--needs-key` --- print `{"needed":false}` or `{"needed":true,"env_var":"MY_KEY"}`

On invocation, the provider reads a JSON request from stdin and writes either streaming JSONL lines or a single JSON response to stdout. See `doc/implementation.md` for the full protocol specification.

### Conversations

Continue previous conversations, use multi-turn message input, and chat interactively.

```bash
# Continue the most recent conversation
llm -c "And what about 3+3?"

# Continue a specific conversation by ID
llm --cid 01j5a... "Follow up question"

# Load messages from a JSON file
llm --messages conversation.json "What next?"

# Load messages from stdin
echo '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello!"}]' | llm --messages - "Follow up"

# Get JSON output instead of streaming text
llm --json "What is 2+2?"

# Combine: messages input with JSON output
llm --messages history.json --json "Summarize"
```

### Interactive chat

```bash
# Start an interactive chat session
llm chat

# Chat with a specific model and system prompt
llm chat -m claude-sonnet-4-6 -s "You are a helpful assistant"

# Chat with tools enabled
llm chat -T llm_time -T llm_version
```

### Structured output

Force the model to return JSON conforming to a schema. Works with both OpenAI (native `response_format`) and Anthropic (transparent tool wrapping).

```bash
# Schema DSL: simple field definitions
llm "Extract: John is 30" --schema "name str, age int"

# With field descriptions
llm "Extract: John is 30" --schema "name str:The person's name, age int:Their age"

# JSON Schema literal
llm "Extract name" --schema '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}'

# Schema from a file
llm "Extract data" --schema schema.json

# Multiple items: wrap in array
llm "List the planets" --schema "name str, diameter_km int" --schema-multi

# Preview DSL output
llm schemas dsl "name str, age int"
```

Schema DSL types: `str` (default), `int`, `float`, `bool`.

### Key management

```bash
llm keys set openai          # Prompted for key (hidden input)
llm keys set anthropic       # Set Anthropic API key
llm keys get openai          # Print stored key
llm keys list                # List all stored key names
llm keys path                # Print path to keys.toml
```

Keys are resolved in order: `--key` flag, `keys.toml`, environment variable (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Model management

```bash
llm models list              # List available models (OpenAI + Anthropic)
llm models default           # Show current default model
llm models default gpt-4o    # Set default model
```

Available models:
- **OpenAI:** `gpt-4o`, `gpt-4o-mini`
- **Anthropic:** `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`

### Conversation logs

Every prompt is logged to a JSONL file (one per conversation). Logs are plain text --- inspect them with `cat`, `grep`, `jq`.

```bash
llm logs list                # List recent conversations
llm logs list --json         # JSON output (pipe to jq)
llm logs list -r             # Print the most recent response text
llm logs list -m gpt-4o      # Filter by model
llm logs list -q "rust"      # Full-text search
llm logs list -u             # Show token usage
llm logs path                # Print logs directory path
llm logs status              # Show logging on/off state
llm logs on                  # Enable logging
llm logs off                 # Disable logging
```

Log files live at `~/.local/share/llm/logs/`. Each file is a JSONL conversation:

```jsonl
{"type":"conversation","v":1,"id":"01j5a...","model":"gpt-4o","name":"Hello","created":"2026-04-03T12:00:00Z"}
{"type":"response","id":"01j5b...","model":"gpt-4o","prompt":"Hello","response":"Hi!","usage":{"input":5,"output":3},"duration_ms":230,...}
```

### Schema management

```bash
llm schemas dsl "name str, age int"   # Preview DSL -> JSON Schema
llm schemas list                      # List schemas used in logs
llm schemas show <id>                 # Show schema by ID
```

### Plugins

```bash
llm plugins list    # Show all providers (compiled + external) and external tools
```

Example output:
```
Compiled providers:
  openai (2 models: gpt-4o, gpt-4o-mini)
  anthropic (3 models: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)

External providers:
  ollama (/usr/local/bin/llm-provider-ollama) (3 models: llama3, mistral, phi3)

External tools:
  web_search (/usr/local/bin/llm-tool-web-search) — Search the web
  upper (/usr/local/bin/llm-tool-upper) — Uppercase text
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error (I/O failure, storage error) |
| 2 | Configuration error (missing key, unknown model, bad config) |
| 3 | Provider error (API failure, network timeout) |

## Library usage

In addition to the CLI, llm-rs can be used as a library from JavaScript/TypeScript (via WASM) or Python (via native module). Both support OpenAI and Anthropic.

### WASM (browser / Obsidian plugin)

```typescript
import init, { LlmClient } from '@llm-rs/wasm';

await init();

// Auto-detects provider from model name
const openai = new LlmClient('sk-...', 'gpt-4o');
const claude = new LlmClient('sk-ant-...', 'claude-sonnet-4-6');

// Or use explicit constructors
const client = LlmClient.newAnthropic('sk-ant-...', 'claude-sonnet-4-6');
const custom = LlmClient.newAnthropicWithBaseUrl('sk-ant-...', 'claude-sonnet-4-6', 'https://my-proxy.example.com');

// Non-streaming
const response = await client.prompt('Hello');

// With system prompt
const answer = await client.promptWithSystem('What is 2+2?', 'Answer only with the number');

// Streaming (callback per chunk)
await client.promptStreaming('Tell me a story', (chunk) => {
    process.stdout.write(chunk);
});

// With options
const result = await client.promptWithOptions(
    'Hello',
    null,  // system prompt (optional)
    '{"temperature": 0.7, "max_tokens": 1000}'
);
```

Build from source:

```bash
wasm-pack build crates/llm-wasm --target web       # ES module for browsers
wasm-pack build crates/llm-wasm --target bundler    # For webpack/rollup (Obsidian plugins)
```

The WASM module is stateless --- no config files, no log storage. HTTP goes through the browser's `fetch()` API. The host application manages API keys and persistence.

### Python

```python
import llm_rs

# Auto-detects provider from model name
client = llm_rs.LlmClient("sk-...", "gpt-4o-mini")
claude = llm_rs.LlmClient("sk-ant-...", "claude-sonnet-4-6")

# Or specify provider explicitly
client = llm_rs.LlmClient("sk-ant-...", "claude-sonnet-4-6", provider="anthropic")

# Non-streaming
response = client.prompt("Hello, world!")
print(response)

# With system prompt
answer = client.prompt("What is 2+2?", system="Answer only with the number")

# Streaming (Python iterator)
for chunk in client.prompt_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

Build from source (requires [uv](https://docs.astral.sh/uv/)):

```bash
cd crates/llm-python
uv venv && uv pip install maturin
uv run maturin develop           # Install to current venv
uv run maturin build --release   # Build wheel for distribution
```

Optional parameters: `provider` (`"openai"` or `"anthropic"`), `base_url` for custom API endpoints, `log_dir` to enable JSONL logging.

## Installation

Requires Rust 1.85+ (2024 edition).

```bash
git clone https://github.com/user/llm-rs
cd llm-rs
cargo install --path crates/llm-cli
```

Or build from the workspace:

```bash
cargo build --release -p llm-cli
# Binary is at target/release/llm
```

## Configuration

Config files live in XDG-standard directories:

```
~/.config/llm/config.toml     # Main configuration
~/.config/llm/keys.toml       # API keys (0600 permissions)
~/.local/share/llm/logs/      # Conversation logs (JSONL)
```

Set `LLM_USER_PATH` to put everything in one directory (useful for testing or migrating from Python `llm`).

**config.toml:**

```toml
default_model = "gpt-4o-mini"
logging = true

[aliases]
claude = "claude-sonnet-4-6"
fast = "gpt-4o-mini"
```

**keys.toml:**

```toml
openai = "sk-..."
anthropic = "sk-ant-..."
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (fallback if not in keys.toml) |
| `ANTHROPIC_API_KEY` | Anthropic API key (fallback if not in keys.toml) |
| `OPENAI_BASE_URL` | Override OpenAI API endpoint (for compatible APIs) |
| `ANTHROPIC_BASE_URL` | Override Anthropic API endpoint |
| `LLM_DEFAULT_MODEL` | Override default model |
| `LLM_USER_PATH` | Override config/data directory (flat layout) |

## Architecture

Seven Rust crates in a Cargo workspace:

```
crates/
  llm-core/      Traits, types, streaming, errors, config, keys, schema DSL, chain loop
  llm-openai/    OpenAI Chat API provider (streaming + tools + structured output)
  llm-anthropic/ Anthropic Messages API provider (streaming + tools + structured output)
  llm-store/     JSONL conversation log storage and queries
  llm-cli/       CLI binary (the `llm` command)
  llm-wasm/      WASM library for browser/Obsidian (excluded from workspace)
  llm-python/    Python native module via PyO3 (excluded from workspace)
```

Dependency flow: `llm-cli`, `llm-wasm`, and `llm-python` are top-level entry points -> `llm-openai` + `llm-anthropic` + `llm-store` -> `llm-core`. No cycles.

Key design choices vs the Python original:

- **Subprocess extensibility, not in-process plugins.** Instead of Python's pluggy-based plugin system, external tools (`llm-tool-*`) and providers (`llm-provider-*`) are standalone executables discovered on `$PATH`. Any language can implement the JSON stdin/stdout protocol. Compiled-in providers (OpenAI, Anthropic) are feature-gated for a minimal core binary.
- **JSONL storage.** One file per conversation instead of SQLite. Append-only, human-readable, no migrations.
- **Async-first.** Single `Provider` trait using futures streams, no sync/async class duplication.
- **TOML config.** Two files (`config.toml` + `keys.toml`) instead of six scattered JSON/YAML/text files.
- **Feature-gated providers.** Compile only the providers you need: `--features openai,anthropic` (both default), or `--no-default-features` for a minimal binary.
- **Multi-target.** Core crates compile for both native and `wasm32-unknown-unknown`. The same provider code runs in the CLI, in a browser, and in a Python module.

See [`doc/metaplan.md`](doc/metaplan.md) for the full design rationale and phased roadmap.

## Testing

```bash
cargo test --workspace    # 361 tests (core workspace crates)
```

| Crate | Tests | What's covered |
|-------|------:|----------------|
| `llm-core` | 119 | Types, config, keys, streams, schema DSL, chain loop, messages (mock provider) |
| `llm-openai` | 42 | HTTP mocking (wiremock), SSE parsing, tool calls, structured output, multi-turn |
| `llm-anthropic` | 48 | HTTP mocking (wiremock), typed SSE, tool_use blocks, transparent schema wrapping, multi-turn |
| `llm-store` | 49 | JSONL round-trips, unicode, malformed recovery, listing/queries, message reconstruction |
| `llm-cli` | 103 | Subprocess protocol/discovery/execution (45 unit), CLI integration (58 e2e with assert_cmd) |

Library targets are verified by their build toolchains: `wasm-pack build` for WASM, `maturin develop` for Python.

## Status

Phase 1 (v0.1) complete --- CLI, WASM library, and Python module working with both OpenAI and Anthropic providers.

Phase 2 complete --- tool calling, chain loop, built-in tools, structured output (both providers), schema DSL, CLI commands for tools and schemas.

Phase 3 complete --- multi-turn conversations with full history accumulation, conversation continuation (`-c`/`--cid`), `--messages`/`--json` flags, interactive `llm chat` REPL, expanded `llm logs` (path/status/on/off, model filter, text search, usage).

Phase 4 core complete --- subprocess extensibility via `llm-tool-*` and `llm-provider-*` protocols. PATH-based discovery, JSON stdin/stdout invocation, streaming JSONL for providers, composite tool executor (builtin + external), `llm plugins list`, async provider registry.

Next: Phase 4 continued (Ollama provider, aliases, options, attachments).

## License

TBD
