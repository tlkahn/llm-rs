# Architecture & Design Rationale

Stable design document for LLM-RS. For current status and roadmap, see [roadmap.md](../roadmap.md). For implementation reference (types, APIs, conventions), see [CLAUDE.md](../../CLAUDE.md).

---

## Design Philosophy

The Python `llm` is a monolith anchored to the Python packaging ecosystem (pluggy, pip, setuptools entry points). The Rust rewrite does not port that architecture. It decomposes the system into crates that each do one thing well, treats text and JSON as the universal interface, and replaces Python's dynamic plugin loading with subprocess-based extension.

| Unix Rule | How it manifests |
|-----------|-----------------|
| **Composition** | Stdin/stdout as primary I/O. `echo "Hello" \| llm` works. `llm logs list --json \| jq` works. JSONL streaming lets downstream tools process chunks as they arrive. |
| **Silence** | Response text goes to stdout. Errors, usage stats, and diagnostics go to stderr. No progress spinners, no "thinking..." messages. `--verbose` is opt-in. |
| **Modularity** | Seven crates, each with one job. Provider crates know nothing about storage. Storage knows nothing about providers. The CLI composes them. |
| **Separation** | Policy lives in TOML config files. Mechanism lives in the binary. The binary doesn't embed defaults that belong in config. |
| **Extensibility** | New providers and tools are executables on `$PATH` speaking a JSON protocol on stdin/stdout. No shared libraries, no WASM runtime, no daemon. |
| **Least Surprise** | CLI mirrors familiar patterns --- `git`-style subcommands, `--json` for machine output, `-` for stdin, `-m` for model, `-s` for system prompt. |
| **Transparency** | `--verbose` logs HTTP requests, resolved config, model selection to stderr. Logs are JSONL files you can inspect with `cat`, `grep`, `jq` --- no special tools needed. |

---

## Crate Structure

| Crate | Responsibility | Key dependencies |
|-------|---------------|-----------------|
| `llm-core` | Traits, types, streaming contracts, error types | `serde`, `thiserror`, `futures` |
| `llm-store` | JSONL file persistence: conversation log writes, queries, directory management | `serde_json`, `llm-core` |
| `llm-openai` | OpenAI provider (Chat API) | `reqwest`, `llm-core` |
| `llm-anthropic` | Anthropic provider (Messages API) | `reqwest`, `llm-core` |
| `llm-ollama` | Ollama local models (Chat API) | `reqwest`, `llm-core` |
| `llm-cli` | Binary entry point, all clap commands | `clap`, all above via features |
| `llm-wasm` | WASM library for browser/Obsidian plugin | `wasm-bindgen`, `llm-core`, `llm-openai` |
| `llm-python` | Python native module via PyO3 | `pyo3`, `llm-core`, `llm-openai`, `llm-store` |

Dependency flow is strictly downward: `llm-cli`, `llm-wasm`, and `llm-python` are top-level entry points that compose the lower crates. Provider crates depend only on `llm-core`. `llm-store` depends only on `llm-core`. No cycles. `llm-wasm` and `llm-python` are excluded from default workspace builds (built with `wasm-pack` and `maturin` respectively).

---

## Key Design Decisions

### JSONL over SQLite

Python `llm` uses SQLite with 12 tables, 21 migrations, and relational joins to normalize inherently hierarchical data. We use JSONL instead:

- **The data is hierarchical, not relational.** Tool calls, attachments, and tool results belong to a response and are never queried independently. JSON nests them naturally; SQL flattens them into junction tables.
- **JSONL is the lingua franca** everywhere else (subprocess IPC, streaming output, `--json` flag). One consistent format throughout the system.
- **Standard Unix tools work directly** (`cat`, `grep`, `jq`, `head`, `tail`, `wc`, `diff`). No special tooling needed.
- **No schema migrations.** New fields are added freely; old readers ignore unknown keys via `#[serde(default)]`.
- **Eliminates `rusqlite`** (bundled SQLite: +3-5s compile time, +1-2MB binary size).

### Subprocess over Plugins

Python `llm` uses pluggy/pip/setuptools entry points for provider and tool extensibility. We use subprocess executables instead:

- Any executable on `$PATH` matching `llm-tool-*` or `llm-provider-*` extends the system. No shared libraries, no ABI compatibility, no package manager.
- Tools can be written in any language (shell, Python, Go, Rust, etc.).
- Protocol is JSON on stdin/stdout --- the same interface used everywhere else.
- Specs: [external-tools.md](../spec/external-tools.md), [external-providers.md](../spec/external-providers.md).

### Async-first Single Trait

Python maintains parallel sync and async class hierarchies (`Model`/`AsyncModel`, `Response`/`AsyncResponse`, `Conversation`/`AsyncConversation` --- six base classes total). Rust uses a single async `Provider` trait with `tokio::runtime::Runtime::block_on` for sync contexts. This eliminates the duplication.

### TOML Config Consolidation

Python scatters state across `keys.json`, `aliases.json`, `default_model.txt`, `options.json`, and YAML templates. Rust consolidates into two files: `config.toml` (settings, aliases, options) + `keys.toml` (API keys, 0600 permissions). Both under XDG-compliant paths.

### Agent Config as TOML Files

Agents are TOML files in discoverable directories, not database records or code definitions. This matches the Unix philosophy: agents are config, not programs.

- **Global + local directories**: `$XDG_CONFIG_HOME/llm/agents/` for global, `$CWD/.llm/agents/` for project-local. Local shadows global (same name wins), analogous to `.gitignore` or `.env` layering.
- **Name derived from filename**: `reviewer.toml` -> agent name `reviewer`. No separate name field, no registry, no ID generation. The filesystem is the index.
- **All fields optional**: An empty `.toml` file is a valid agent that uses all defaults. This makes `llm agent init` trivial and lets users build up config incrementally.
- **Stub fields for future tiers**: `sub_agents` and `memory` parse and persist but aren't wired. This avoids breaking config files when those features ship — users can start writing configs now. `budget` was a stub through Phase 5 and is now wired (Phase 6).

### Budget Enforcement as Graceful Stop

Budget enforcement mirrors chain_limit semantics: when cumulative token usage exceeds the budget, the chain completes the current iteration fully (collects all chunks, emits the IterationEnd event), then stops before starting the next. It does not error — `ChainResult.budget_exhausted` is a flag, not an exception. This matches the chain_limit pattern and avoids losing partial results.

Budget = input + output tokens combined. The budget is checked only when there are pending tool calls (i.e., the chain would continue). A single-iteration chain that exceeds the budget still completes normally — budget prevents *additional* iterations, not the current one.

### Platform Abstraction (WASM + Python)

`llm-core` production code has zero tokio usage. `llm-openai` has exactly 3 tokio-specific lines in its streaming path. Platform portability is achieved through surgical cfg-gating:

- **`ResponseStream` Send bound**: cfg-gated. Native requires `Send` (multi-threaded tokio). WASM drops `Send` (single-threaded).
- **`Provider` trait bounds**: cfg-gated. Native uses `Send + Sync`. WASM uses `?Send`.
- **Streaming channel**: `futures::channel::mpsc` (works on all platforms) instead of `tokio::sync::mpsc`. Only the spawn call is cfg-gated (`tokio::spawn` vs `wasm_bindgen_futures::spawn_local`).
- **Dependency gating**: `llm-core` has no tokio in `[dependencies]`. `llm-openai` makes tokio a `cfg(not(wasm32))` dependency.

---

## Risk Register

1. **SSE delta accumulation.** Each provider streams tool call arguments differently. OpenAI sends argument deltas indexed by tool call position across multiple SSE chunks. Getting this wrong produces corrupted JSON arguments. Mitigate: integration tests with recorded HTTP cassettes.

2. **Subprocess provider latency.** Subprocess tools add per-invocation startup cost (~5-50ms). In tight chain loops (5+ tool calls), this compounds. Mitigate: keep commonly used tools built-in; subprocess is for third-party extension.

3. **Conversation serialization.** Each provider needs different message array formats. Building the message array from conversation history is easy to get subtly wrong. Mitigate: test with multi-turn conversations that include tool calls and attachments.

4. **JSONL format evolution.** Once conversation files exist in the wild, changing the record format risks breaking existing logs. Mitigate: `"v":1` version field in conversation header. Add fields freely (readers ignore unknowns), but avoid renaming or removing fields.

5. **`prompt` command complexity.** The Python `prompt` command is ~580 lines with 30+ flags. Rushing this risks a buggy CLI. Mitigate: implement flags incrementally across phases; test each flag in isolation.

6. **wasm32 reqwest streaming.** reqwest 0.12 supports wasm32 via web-sys fetch, but streaming SSE via `ReadableStream` may have edge cases. Mitigate: test `llm-wasm` with wasm-pack in a browser environment against a mock SSE server.

---

## Python Source Reference

Reference files from [simonw/llm](https://github.com/simonw/llm) used during implementation:

| File | Lines | What to extract |
|------|-------|----------------|
| `llm/models.py` | ~2165 | Type definitions, streaming contracts, chain loop, tool execution, `log_to_db` |
| `llm/cli.py` | ~4050 | Command structure, flag definitions, prompt orchestration, log formatting |
| `llm/default_plugins/openai_models.py` | ~950 | `build_messages`, SSE parsing, delta tool-call accumulation, option mapping |
| `llm/migrations.py` | ~420 | Reference for `llm import --from-sqlite` (understand source schema) |
| `llm/utils.py` | ~390 | Schema DSL parser, Fragment type, ULID generator, code block extraction |
| `llm/errors.py` | ~15 | Error types (`ModelError`, `NeedsKeyException`) |
| `llm/__init__.py` | ~350 | Public API surface, `get_model`, `get_key`, `user_dir` |
| `tests/conftest.py` | ~488 | MockModel pattern, test fixtures, HTTP mocking approach |
