# Axe Feature Readiness Assessment

Assessment of llm-rs architecture against the Unix-philosophy agent features described in `~/Desktop/axe/docs/idea/unix-philosophy-agent-in-rust.md`.

Original assessment written after Phase 2; updated after Phase 4 (v0.4) and again after Phase 9 (v0.9).

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

## Bottom Line (Post-Phase 4)

7 of 12 items from the Phase 2 assessment are resolved. The architecture no longer has fundamental blockers. The remaining gaps — parallel tool exec, agent config/discovery, sub-agent delegation, budget enforcement, dry-run, retry, memory — are all additive features that build on existing seams rather than requiring core type changes.

---

## Post-Phase 9 Reassessment (v0.9, 2026-04-11)

Phases 5 through 9 have all landed. Five tiered items from the prior plan are now complete: agent config & discovery (v0.5), budget tracking (v0.6), retry/backoff (v0.7), dry-run mode (v0.8), and parallel tool execution (v0.9). Only sub-agent delegation and the memory system remain open from the original axe list.

### Newly Resolved

**5. Budget tracking** (was partial) — Phase 6 added `Usage::add()`/`total()`, `ChainResult.total_usage` accumulation across iterations, `ChainEvent::IterationEnd.cumulative_usage`, and a `budget: Option<u64>` parameter on `chain()`. Exceeding budget triggers `ChainEvent::BudgetExhausted` and a graceful stop (mirroring `chain_limit`). `-u` surfaces cumulative totals; the chat REPL tracks session-wide usage. `BudgetConfig.max_tokens` is wired from agent TOML. `LLM_BUDGET_REMAINING` env var and exit-code-4 are still deferred to the sub-agent tier.

**6. Parallel tool execution** — Phase 9 added `ParallelConfig { enabled, max_concurrent }` and `dispatch_tools()` in `llm-core/chain.rs`. Sequential fast path when disabled or single-call; otherwise `future::join_all` (unlimited) or `stream::iter(futs).buffered(n)` (bounded). Result order is preserved. `--sequential-tools`/`--max-parallel-tools` on `prompt`/`chat`/`agent run`; `parallel_tools`/`max_parallel_tools` in agent TOML. `--tools-approve` forces sequential to avoid interleaved stdin prompts.

**7. Agent config and discovery** — Phase 5 added `AgentConfig` with full TOML parsing (`model`, `system_prompt`, `tools`, `chain_limit`, `options`, plus `budget`/`retry`/`parallel_tools`/`max_parallel_tools` wired; `sub_agents`/`memory` parsed as stubs). `Paths.agents_dir()`, `discover_agents()` scanning global + local with local-shadows-global, `resolve_agent()`, and `llm agent run/list/show/init/path` subcommands all shipped.

**9. Dry-run mode** — Phase 8 added `--dry-run` on `llm agent run`. Resolves the full invocation pipeline (agent file, model + source, provider, system prompt, prompt text, tools classified builtin/external, merged options, chain limit, budget, retry, logging flag, resolved `ParallelConfig`) without calling the LLM, resolving keys, or writing logs. `DryRunReport` with `render_plain()` and `render_json()`. `-v`/`-vv` populate the serialized `Prompt` JSON the provider would have received.

**10. Retry/backoff** — Phase 7 added `LlmError::HttpError { status, message }` (with `is_retryable()` true for 429/5xx), `RetryConfig` with exponential backoff + jitter in `llm-core/retry.rs`, and `RetryProvider` wrapper in `llm-cli/retry.rs` (pre-stream only). `--retries` flag on `prompt`, `chat`, and `agent run`. Agent TOML `[retry]` section wired; CLI overrides agent config. Both OpenAI and Anthropic emit `HttpError` for non-success HTTP status codes.

### Parked (permanent)

**8. Sub-agent delegation** — **Parked.** Not a gap, a deliberate architectural divergence. llm-rs delegates hierarchical workflows to the *specialist tool* pattern (`llm-tool-*` executables that may internally call `llm prompt`), not to an in-process recursive runtime. Research on multi-agent systems informed this decision. See [specialist-tools-vs-sub-agents.md](specialist-tools-vs-sub-agents.md).

**11. Memory system** — **Parked.** Superseded by specialist tools plus JSONL logs; per-agent memory is deferred to user composition rather than built into the runtime. See [specialist-tools-vs-sub-agents.md](specialist-tools-vs-sub-agents.md).

### v0.9 Summary Table

| axe Feature | Phase 2 State | Post-Phase 4 State | Post-Phase 9 State |
|---|---|---|---|
| ReAct loop (multi-turn chain) | Single-turn only | **Resolved** | Resolved |
| Multi-turn messages in `Prompt` | **Missing** | **Resolved** | Resolved |
| External CLI tools | Trait only | **Resolved** | Resolved |
| `--messages` input | Not exposed | **Resolved** | Resolved |
| `--json` output | Not exposed | **Resolved** | Resolved |
| Chain observability | Not started | **Resolved** | Resolved |
| Budget tracking | Not started | Partial | **Resolved** (v0.6) — env var + exit-code deferred |
| Parallel tool exec | Sequential loop | Still open | **Resolved** (v0.9) |
| Agent TOML config | Not started | Still open | **Resolved** (v0.5) |
| Dry-run mode | Not started | Still open | **Resolved** (v0.8) |
| Retry/backoff | Not started | Still open | **Resolved** (v0.7) |
| Sub-agent delegation | Not started | Still open | **Parked** (v0.9+, see [design note](specialist-tools-vs-sub-agents.md)) |
| Memory system | Not started | Still open | **Parked** (v0.9+, see [design note](specialist-tools-vs-sub-agents.md)) |

## Bottom Line (Post-Phase 9)

11 of 13 axe features are resolved; 2 are deliberately parked. The axe-vs-llm-rs feature gap is now closed — the two remaining items are intentional architectural divergences, not gaps. Sub-agent delegation and the memory system are superseded by the specialist tool pattern, which gives users hierarchical composition without the failure modes of recursive multi-agent runtimes. See [specialist-tools-vs-sub-agents.md](specialist-tools-vs-sub-agents.md) for the full rationale.

---

## Design Divergence

llm-rs does not build a recursive sub-agent runtime or an agent-scoped memory system. Hierarchical workflows compose through specialist tools (`llm-tool-*`), and cross-turn memory composes through the existing JSONL log store and user-level scripting. Full rationale and worked examples: [specialist-tools-vs-sub-agents.md](specialist-tools-vs-sub-agents.md). Parked items index: [roadmap.md](../roadmap.md).
