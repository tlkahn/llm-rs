# llm-rs

A Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) --- a CLI tool for interacting with Large Language Models from the terminal.

> **Status:** Work in progress. Core types, OpenAI provider, and storage layer are implemented. The CLI binary is not yet usable.

## Goals

- Single static binary, fast startup, low memory
- Unix-native: stdin/stdout streaming, JSONL throughout, composable with pipes
- JSONL log files inspectable with `cat`, `grep`, `jq` --- no database tools required
- Multi-provider: OpenAI, Anthropic, Ollama (compiled in), plus subprocess-based extension for third-party providers

## Project structure

```
crates/
  llm-core/      # Traits, types, streaming contracts, errors
  llm-openai/    # OpenAI Chat API provider
  llm-store/     # JSONL conversation log storage
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

126 tests across three crates, covering type contracts, HTTP mocking, serialization round-trips, and filesystem I/O.

## Design

See [`doc/metaplan.md`](doc/metaplan.md) for the full architecture and phased implementation plan.

Key divergences from the Python original:

- **No plugin system.** Providers are compiled in (feature flags) or discovered as subprocess executables on `$PATH`.
- **JSONL storage.** One file per conversation instead of SQLite. Append-only, human-readable, no migrations.
- **Async-first.** Single `Provider` trait using tokio streams, no sync/async class duplication.
- **TOML config.** Two files (`config.toml` + `keys.toml`) instead of six scattered JSON/YAML/text files.

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
