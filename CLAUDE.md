# CLAUDE.md

## Project

LLM-RS: Rust reimplementation of [simonw/llm](https://github.com/simonw/llm) (v0.30). CLI tool and library for interacting with LLMs via a unified interface. See `doc/metaplan.md` for full architecture and `doc/implementation.md` for current status.

## Commands

```bash
cargo test --workspace           # Run all 126 tests
cargo test -p llm-core           # Core types/traits (55 tests)
cargo test -p llm-openai         # OpenAI provider (29 tests)
cargo test -p llm-store          # JSONL storage (42 tests)
cargo clippy --workspace         # Lint (llm-store should be warning-free)
cargo check --workspace          # Type-check only
```

## Architecture

Cargo workspace with three crates (more planned):

```
crates/
  llm-core/     # Traits, types, streaming, errors. No I/O.
  llm-openai/   # OpenAI Chat API provider (streaming SSE + non-streaming)
  llm-store/    # JSONL file-per-conversation log storage
```

Dependency flow: `llm-openai` and `llm-store` depend on `llm-core`. No cycles. A future `llm-cli` crate will depend on all three.

### Key types (llm-core)

- `Provider` trait: async streaming interface, returns `ResponseStream` (Pin<Box<dyn Stream<Item=Result<Chunk>>>>)
- `Prompt`: text + system + attachments + tools + tool_results + schema + options
- `Response`: materialized post-stream result with all fields (prompt, response text, usage, tool_calls, duration, datetime)
- `Chunk`: streaming enum (Text, ToolCallStart, ToolCallDelta, Usage, Done)
- `LlmError`: Model, NeedsKey, Provider, Config, Io, Store

### Storage (llm-store)

JSONL files, one per conversation, at `$XDG_DATA_HOME/llm/logs/{conversation_id}.jsonl`. Line 1 is a `ConversationRecord` header (`"type":"conversation"`), subsequent lines are `ResponseRecord`s (`"type":"response"`) with all data denormalized inline. `LineRecord` is the `#[serde(tag = "type")]` dispatch enum.

Public API: `LogStore::open()`, `log_response()`, `read_conversation()`, `list_conversations()`, `latest_conversation_id()`.

## Implementation status

Phase 1 (v0.1), Steps 1--3 of 5 complete:
- [x] Step 1: Core types and Provider trait
- [x] Step 2: OpenAI provider (streaming + non-streaming)
- [x] Step 3: JSONL storage layer
- [ ] Step 4: Config and KeyStore (TOML, XDG paths)
- [ ] Step 5: CLI binary (prompt, keys, models, logs commands)

## Conventions

- Rust 2024 edition
- TDD: tests written before implementation
- IDs: ULID (26-char lowercase), via `ulid` crate
- Timestamps: RFC 3339 via `chrono`
- Errors: single `LlmError` enum in llm-core, `#[from]` for io::Error
- Tests: inline `#[cfg(test)]` modules, `tempfile::TempDir` for filesystem tests, `wiremock` for HTTP mocking
