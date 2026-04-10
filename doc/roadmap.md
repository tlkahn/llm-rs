# LLM-RS Roadmap

Living roadmap and planning document for LLM-RS development.

**Related docs:**
- [Development Process](process.md) --- the plan-build-close cycle
- [Architecture & Design](design/architecture.md) --- philosophy, crate structure, design decisions
- [External Tool Protocol](spec/external-tools.md) --- `llm-tool-*` spec
- [External Provider Protocol](spec/external-providers.md) --- `llm-provider-*` spec
- [Implementation Notes](implementation.md) --- pitfall journal, gotchas, workarounds
- [CLAUDE.md](../CLAUDE.md) --- implementation reference (types, APIs, conventions)

---

## Scope

Unix-philosophy agentic CLI for LLMs, inspired by [simonw/llm](https://github.com/simonw/llm). Core prompting, conversations, tool calling with chain loops, structured output, subprocess-based extensibility, JSONL file logging, multi-provider. Embeddings, templates, and fragments deferred to future work.

---

## Phase Status

| Phase | Version | Status | Summary |
|-------|---------|--------|---------|
| 1 --- Core Loop | v0.1 | Complete | `echo "Hello" \| llm` end-to-end, streaming, logging, OpenAI + Anthropic, WASM + Python |
| 2 --- Tools & Structured Output | v0.2 | Complete | Tool calling (both providers), chain loop, built-in tools, schema DSL, `--schema`/`--schema-multi` |
| 3 --- Conversations & Multi-turn | v0.3 | Complete | `Message`/`Role` types, `-c`/`--cid`, `llm chat` REPL, full `llm logs` feature set |
| 4 --- Extensibility & More | v0.4 | Complete | Subprocess tools + providers, `llm plugins`, `--verbose`, `-o/--option`, aliases |
| 5 --- Agent Config & Discovery | v0.5 | Complete | Agent TOML config, directory discovery (local shadows global), `llm agent run/list/show/init/path` |

---

## Future Work

### Axe: Agent Features (prioritized, see [readiness assessment](research/axe-readiness-assessment.md))

**Tier 1** — zero unresolved deps, highest value:
- Budget tracking (accumulation + display) --- cross-turn token accumulation in `chain()`, surface via `-u`/`ChainEvent`

**Tier 2** — zero or newly-resolved deps:
- Retry/backoff --- exponential backoff + jitter for 429/5xx, wraps provider calls
- Dry-run mode --- `--dry-run` resolves agent config and prints without LLM call
- Parallel tool execution --- `JoinSet` in `chain()` tool dispatch

**Tier 3** — higher complexity:
- Sub-agent delegation --- `call_agent` tool spawning child `llm agent run` (+ exit-code-4, `LLM_BUDGET_REMAINING` env var)
- Memory system --- per-agent JSONL storage; pluggable backends (markdown, SQLite, Redis) deferred

### Other

No fixed ordering:
- Ollama provider (via subprocess or compiled `llm-ollama` crate)
- `-a/--attachment`, `--at/--attachment-type`
- `-x/--extract`, `--xl/--extract-last` (code block extraction)
- Shell completions (`clap_complete`)
- Embeddings support
- Templates and fragments
- `--async` flag for background/scripting use
- Config resolution tracing (`--verbose` showing key/model/alias resolution steps)

---

## Iteration Strategy

Each phase is a vertical slice delivering a usable tool. Within each phase, work bottom-up through the crate layers using TDD (red-green-refactor). The phase boundary is where you dogfood the result before planning the next.

All work follows strict TDD: write failing tests first, implement until they pass, refactor. All existing tests must stay green at every step. `cargo test --workspace` and `cargo clippy --workspace` gate each commit.

---

## Parked

Items explicitly set aside. Not planned for current or next phases.

- `llm import --from-sqlite` --- low demand; users can convert with `jq` scripts if needed
