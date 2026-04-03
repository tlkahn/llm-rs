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
```

Log files live at `~/.local/share/llm/logs/`. Each file is a JSONL conversation:

```jsonl
{"type":"conversation","v":1,"id":"01j5a...","model":"gpt-4o","name":"Hello","created":"2026-04-03T12:00:00Z"}
{"type":"response","id":"01j5b...","model":"gpt-4o","prompt":"Hello","response":"Hi!","usage":{"input":5,"output":3},"duration_ms":230,...}
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
  llm-core/      Traits, types, streaming, errors, config, keys
  llm-openai/    OpenAI Chat API provider (streaming SSE + non-streaming)
  llm-anthropic/ Anthropic Messages API provider (streaming SSE + non-streaming)
  llm-store/     JSONL conversation log storage and queries
  llm-cli/       CLI binary (the `llm` command)
  llm-wasm/      WASM library for browser/Obsidian (excluded from workspace)
  llm-python/    Python native module via PyO3 (excluded from workspace)
```

Dependency flow: `llm-cli`, `llm-wasm`, and `llm-python` are top-level entry points -> `llm-openai` + `llm-anthropic` + `llm-store` -> `llm-core`. No cycles.

Key design choices vs the Python original:

- **No plugin system.** Providers are compiled in (feature flags) or discovered as subprocess executables on `$PATH` (planned).
- **JSONL storage.** One file per conversation instead of SQLite. Append-only, human-readable, no migrations.
- **Async-first.** Single `Provider` trait using futures streams, no sync/async class duplication.
- **TOML config.** Two files (`config.toml` + `keys.toml`) instead of six scattered JSON/YAML/text files.
- **Feature-gated providers.** Compile only the providers you need: `--features openai,anthropic` (both default), or `--no-default-features` for a minimal binary.
- **Multi-target.** Core crates compile for both native and `wasm32-unknown-unknown`. The same provider code runs in the CLI, in a browser, and in a Python module.

See [`doc/metaplan.md`](doc/metaplan.md) for the full design rationale and phased roadmap.

## Testing

```bash
cargo test --workspace    # 225 tests (core workspace crates)
```

| Crate | Tests | What's covered |
|-------|------:|----------------|
| `llm-core` | 88 | Type contracts, config parsing, key resolution, stream collection |
| `llm-openai` | 29 | HTTP mocking (wiremock), SSE parsing, error handling |
| `llm-anthropic` | 34 | HTTP mocking (wiremock), typed SSE parsing, Anthropic auth headers |
| `llm-store` | 42 | JSONL round-trips, unicode, malformed recovery, listing/queries |
| `llm-cli` | 32 | End-to-end CLI: stdout/stderr/exit codes, wiremock API, logging |

Library targets are verified by their build toolchains: `wasm-pack build` for WASM, `maturin develop` for Python.

## Status

Phase 1 (v0.1) is complete --- CLI, WASM library, and Python module all working with both OpenAI and Anthropic providers. Phase 2 will add conversations, Ollama provider, attachments, and interactive chat.

## License

TBD
