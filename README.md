# llm-rs

A Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) --- a CLI tool for interacting with Large Language Models from the terminal.

## Usage

```bash
# Send a prompt (streams to stdout)
echo "Hello" | llm

# Positional text works too
llm "Explain monads in one sentence"

# Specify model and system prompt
llm "What is 2+2?" -m gpt-4o -s "Answer only with the number"

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
llm keys get openai          # Print stored key
llm keys list                # List all stored key names
llm keys path                # Print path to keys.toml
```

Keys are resolved in order: `--key` flag, `keys.toml`, environment variable (e.g. `OPENAI_API_KEY`).

### Model management

```bash
llm models list              # List available models
llm models default           # Show current default model
llm models default gpt-4o    # Set default model
```

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
claude = "claude-sonnet-4-20250514"
fast = "gpt-4o-mini"
```

**keys.toml:**

```toml
openai = "sk-..."
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (fallback if not in keys.toml) |
| `OPENAI_BASE_URL` | Override OpenAI API endpoint (for compatible APIs) |
| `LLM_DEFAULT_MODEL` | Override default model |
| `LLM_USER_PATH` | Override config/data directory (flat layout) |

## Architecture

Four Rust crates in a Cargo workspace:

```
crates/
  llm-core/      Traits, types, streaming, errors, config, keys
  llm-openai/    OpenAI Chat API provider (streaming SSE + non-streaming)
  llm-store/     JSONL conversation log storage and queries
  llm-cli/       CLI binary (the `llm` command)
```

Dependency flow: `llm-cli` -> `llm-openai` + `llm-store` -> `llm-core`. No cycles.

Key design choices vs the Python original:

- **No plugin system.** Providers are compiled in (feature flags) or discovered as subprocess executables on `$PATH` (planned).
- **JSONL storage.** One file per conversation instead of SQLite. Append-only, human-readable, no migrations.
- **Async-first.** Single `Provider` trait using tokio streams, no sync/async class duplication.
- **TOML config.** Two files (`config.toml` + `keys.toml`) instead of six scattered JSON/YAML/text files.
- **Feature-gated providers.** Compile only the providers you need: `--features openai` (default), or `--no-default-features` for a minimal binary.

See [`doc/metaplan.md`](doc/metaplan.md) for the full design rationale and phased roadmap.

## Testing

```bash
cargo test --workspace    # 188 tests
```

| Crate | Tests | What's covered |
|-------|------:|----------------|
| `llm-core` | 88 | Type contracts, config parsing, key resolution, stream collection |
| `llm-openai` | 29 | HTTP mocking (wiremock), SSE parsing, error handling |
| `llm-store` | 42 | JSONL round-trips, unicode, malformed recovery, listing/queries |
| `llm-cli` | 29 | End-to-end CLI: stdout/stderr/exit codes, wiremock API, logging |

## Status

Phase 1 (v0.1) is complete. Phase 2 will add conversations, Anthropic + Ollama providers, attachments, and interactive chat.

## License

TBD
