# llm-rs

A Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) --- a CLI tool for interacting with Large Language Models from the terminal.

> **Status:** Work in progress (Phase 1, Step 5 of 5). Core types, OpenAI provider, storage layer, and configuration system are implemented. The CLI binary is next.

## Goals

- Single static binary, fast startup, low memory
- Unix-native: stdin/stdout streaming, JSONL throughout, composable with pipes
- JSONL log files inspectable with `cat`, `grep`, `jq` --- no database tools required
- Multi-provider: OpenAI, Anthropic, Ollama (compiled in), plus subprocess-based extension for third-party providers

## Project structure

```
crates/
  llm-core/      # Traits, types, streaming contracts, errors, config, key management
  llm-openai/    # OpenAI Chat API provider (streaming SSE + non-streaming)
  llm-store/     # JSONL conversation log storage and queries
  llm-cli/       # CLI binary (planned)
```

## Building

Requires Rust 2024 edition (1.85+).

```bash
cargo build --workspace
```

## Testing

```bash
cargo test --workspace
```

159 tests across three crates, covering type contracts, HTTP mocking, serialization round-trips, filesystem I/O, config parsing, and key resolution.

## Design

See [`doc/metaplan.md`](doc/metaplan.md) for the full architecture and phased implementation plan.

Key divergences from the Python original:

- **No plugin system.** Providers are compiled in (feature flags) or discovered as subprocess executables on `$PATH`.
- **JSONL storage.** One file per conversation instead of SQLite. Append-only, human-readable, no migrations.
- **Async-first.** Single `Provider` trait using tokio streams, no sync/async class duplication.
- **TOML config.** Two files (`config.toml` + `keys.toml`) in XDG directories instead of six scattered JSON/YAML/text files. Pure XDG path resolution on all platforms.

## Configuration

Config files live in XDG-standard directories:

```
~/.config/llm/config.toml    # main configuration
~/.config/llm/keys.toml      # API keys (0600 permissions)
~/.local/share/llm/logs/     # conversation logs (JSONL)
```

Override with `LLM_USER_PATH` to put everything in one directory (e.g. for testing or migration from Python `llm`).

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
anthropic = "sk-ant-..."
```

API keys are resolved in order: `--key` flag, `keys.toml`, environment variable (e.g. `OPENAI_API_KEY`).

## Planned CLI usage

```bash
# Send a prompt
echo "Hello" | llm

# Specify model and system prompt
llm "Explain monads" -m gpt-4o -s "Be concise"

# List recent conversations
llm logs list

# Continue a conversation
llm "Follow up question" -c

# Pipe-friendly JSON output
llm logs list --json | jq '.model'
```

## License

TBD
