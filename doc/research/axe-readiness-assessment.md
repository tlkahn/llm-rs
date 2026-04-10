# Axe Feature Readiness Assessment

Assessment of llm-rs architecture against the Unix-philosophy agent features described in `~/Desktop/axe/docs/idea/unix-philosophy-agent-in-rust.md`.

Original assessment written after Phase 2; updated after Phase 4 completion (v0.4).

---

## Phase 2 Assessment (2026-04-09)

The original assessment identified 5 roadblocks and 12 feature gaps. See git history (`9fd2374a`) for the full original text.

---

## Post-Phase 4 Reassessment (2026-04-10)

Phases 3 (Conversations & Multi-turn) and 4 (Extensibility & More) have landed. The original "single biggest blocker" — `Prompt` being a single-turn type — is resolved.

### Resolved

**1. Conversation history is one turn deep** — Phase 3 added `Prompt.messages: Vec<Message>`. `chain()` now accumulates the full `[user, assistant+tools, tool, assistant+tools, tool, ...]` history across iterations. Both providers consume the messages field. The critical architectural gap is closed.

**2. No external CLI tool dispatch** — Phase 4 added `subprocess/tool.rs` with `ExternalToolExecutor` implementing `ToolExecutor`. Fork/exec via `tokio::process::Command`, stdin JSON for arguments, stdout/stderr/exit-code capture. The `llm-tool-*` PATH convention provides discovery. `-T` flag resolves external tools in both `prompt` and `chat`.

**3. No `--messages` input or `--json` output** — Phase 3 added both flags to `llm prompt`. `--messages` loads a JSON conversation history (file or stdin). `--json` emits a structured envelope with model, content, tool_calls, usage, and conversation_id. The binary can now be used as a composable sub-process.

**4. Verbose/observability** — Phase 4 added `ChainEvent` enum (`IterationStart`/`IterationEnd`) with `on_event` callback in `chain()`. `-v` shows per-iteration summary; `-vv` dumps full message JSON. `--verbose` implies `--tools-debug`.

### Partially Resolved

**5. Budget/token tracking across turns** — Per-iteration usage is collected via `ChainEvent::IterationEnd` and displayed with `-u/--usage`. However: no cross-turn budget accumulation, no `LLM_BUDGET_REMAINING` env var for sub-agents, no exit-code-4 on budget exhaustion.

### Still Open

**6. Parallel tool execution** — Tool execution in `chain()` is still sequential (`for call in &tool_calls`). No `join_all()` or `JoinSet`.

**7. Agent config and discovery** — No `llm agent run` subcommand, no `.llm/agents/` directory support, no agent TOML config parsing.

**8. Sub-agent delegation** — No parent/child agent orchestration. The `--messages` + `--json` plumbing now exists, but no dispatch mechanism.

**9. Dry-run mode** — No `--dry-run` flag to resolve config and print without executing.

**10. Retry/backoff** — No retry logic wrapping provider calls.

**11. Memory system** — No cross-conversation memory or RAG abstraction.

### Updated Summary Table

| axe Feature | Phase 2 State | Post-Phase 4 State |
|---|---|---|
| ReAct loop (multi-turn chain) | Single-turn only | **Resolved** — full message accumulation |
| Multi-turn messages in `Prompt` | **Missing** | **Resolved** — `Prompt.messages: Vec<Message>` |
| External CLI tools | Trait only | **Resolved** — `ExternalToolExecutor`, `llm-tool-*` protocol |
| `--messages` input | Not exposed | **Resolved** |
| `--json` output | Not exposed | **Resolved** |
| Chain observability | Not started | **Resolved** — `ChainEvent`, `-v`/`-vv` |
| Budget tracking | Not started | **Partial** — per-iteration only, no budget enforcement |
| Parallel tool exec | Sequential loop | **Still open** |
| Agent TOML config | Not started | **Still open** |
| Sub-agent delegation | Not started | **Still open** (plumbing ready via `--messages`/`--json`) |
| Dry-run mode | Not started | **Still open** |
| Retry/backoff | Not started | **Still open** |
| Memory system | Not started | **Still open** |

## Bottom Line

7 of 12 items from the Phase 2 assessment are resolved. The architecture no longer has fundamental blockers. The remaining gaps — parallel tool exec, agent config/discovery, sub-agent delegation, budget enforcement, dry-run, retry, memory — are all additive features that build on existing seams rather than requiring core type changes.

---

## Prioritized Plan for Axe-on-LLM-RS

Grouping and tiering of remaining work, based on importance and dependency analysis.

### Group A: Agent Core

The feature that turns llm-rs into an agent framework. Everything in Group C blocks on this.

- **Agent config & discovery** — TOML config (`model`, `system_prompt`, `tools`, `budget`), `.llm/agents/` directory with local-shadows-global, `llm agent run <name>` subcommand. Sub-agents and memory fields are inventoried in the TOML schema but not wired up until their implementations land.

### Group B: Loop Hardening

Making `chain()` production-ready for long-running agentic use. All independent of each other and of Group A.

- **Budget tracking (accumulation + display)** — Cross-turn token accumulation in `chain()`, surface via `-u`/`--usage` and `ChainEvent`. Exit-code-4 and `LLM_BUDGET_REMAINING` env var deferred to sub-agent tier.
- **Retry/backoff** — Exponential backoff + jitter for 429/5xx. Never retry 401/403/400. Wraps provider calls.
- **Parallel tool execution** — `JoinSet` / `join_all` in `chain()` tool dispatch loop.

### Group C: Agent Ecosystem

Features that build on a working agent (depend on Group A).

- **Dry-run mode** — `--dry-run` resolves agent config (system prompt + tools + memory + budget) and prints without making an LLM call. Needs agent TOML to exist.
- **Sub-agent delegation** — `call_agent` as a special tool that spawns child `llm agent run`. Needs agent config + budget env var plumbing (exit-code-4 + `LLM_BUDGET_REMAINING` land here).
- **Memory system** — Per-agent JSONL storage (reuses `llm-store` patterns). Pluggable backends (markdown, SQLite, Redis) deferred.

### Tiering

```
Tier 1 ─── zero unresolved deps, highest value
  ├── Agent config & discovery
  └── Budget tracking (accumulation + display)

Tier 2 ─── zero or newly-resolved deps
  ├── Retry/backoff
  ├── Dry-run mode (unblocked by Tier 1)
  └── Parallel tool execution

Tier 3 ─── higher complexity, unblocked by Tier 1
  ├── Sub-agent delegation (+ budget env var plumbing)
  └── Memory system (JSONL first, pluggable backends later)
```
