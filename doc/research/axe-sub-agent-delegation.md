# Axe Sub-Agent Delegation: Feature Inventory

> **Status: Archived research.** llm-rs decided not to build sub-agent delegation. This document remains as a record of what was considered. See [specialist-tools-vs-sub-agents.md](specialist-tools-vs-sub-agents.md) for the decision and rationale.

Research reference for the axe sub-agent feature set, captured while evaluating whether to port it into llm-rs. Sourced from the axe Go implementation at `~/Desktop/axe/` (commit snapshot 2026-04-11).

The goal of this doc is to enumerate **what axe ships** so the llm-rs port can design a compatible feature set on top of the existing v0.9 seams (`AgentConfig`, `chain()`, `ChainEvent`, `BudgetConfig`, `ParallelConfig`, `--messages`/`--json`, `DryRunReport`).

---

## 1. The `call_agent` tool

Axe exposes sub-agent dispatch as a single built-in LLM tool named `call_agent`. It is injected into the provider request **only when** the running agent's TOML declares `sub_agents = [...]` **and** the current recursion depth is below the effective max depth.

**Key code:**
- Tool definition: `internal/tool/tool.go:23–74` (`CallAgentTool()`)
- Name constant: `internal/toolname/toolname.go` → `CallAgentToolName = "call_agent"`
- Injection site: `cmd/run.go:421–424`

**Tool schema (LLM-facing):**

```json
{
  "name": "call_agent",
  "description": "Delegate a task to a sub-agent. Available agents: <comma-separated>",
  "parameters": {
    "agent":   { "type": "string", "required": true,
                 "description": "Name of the sub-agent (must be one of: <list>)" },
    "task":    { "type": "string", "required": true,
                 "description": "What you need the sub-agent to do" },
    "context": { "type": "string", "required": false,
                 "description": "Additional context from your conversation to pass along" }
  }
}
```

The description string is parameterized with the actual allowlist, so the model always sees which child agents are available.

---

## 2. Agent TOML schema additions

Axe extends its agent config with two top-level items (`internal/agent/agent.go:58–63`):

```toml
sub_agents = ["test-runner", "lint-checker"]

[sub_agents_config]
max_depth = 3      # 0 = use system default (3). Hard cap: 5.
parallel  = true   # default true; *bool so "unset" is detectable
timeout   = 60     # per sub-agent timeout in seconds; 0 = inherit parent
```

**Validation** (`internal/agent/agent.go:106–114`): `max_depth` must be in `[0, 5]`, `timeout` must be non-negative, otherwise a `ValidationError` is returned at load time.

llm-rs already parses `sub_agents: Vec<String>` as a stub on `AgentConfig` — see `CLAUDE.md`'s Agent system section. The gap is adding `SubAgentsConfig` (depth/parallel/timeout) and wiring it through.

---

## 3. Depth tracking and tool gating

Depth is tracked as a plain integer passed through an `ExecuteOptions` struct (`internal/tool/tool.go:28–45`), not a global.

- Top-level run starts at `depth = 0` (`cmd/run.go:406`).
- Each nested `call_agent` increments: `newDepth = opts.Depth + 1` (`tool/tool.go:255`).
- Depth limit enforced at the tool-executor level: `if opts.Depth >= opts.MaxDepth { return error }` (`tool/tool.go:130–136`).
- **Gate** at the injection site (`cmd/run.go:422–424`):

  ```go
  if len(cfg.SubAgents) > 0 && depth < effectiveMaxDepth {
      req.Tools = append(req.Tools, tool.CallAgentTool(cfg.SubAgents))
  }
  ```

  Crucially, at the depth ceiling the tool is **not injected at all**, so the leaf agent simply never sees `call_agent` and runs as a single-shot agent. The model cannot accidentally exceed the depth by picking the tool.

**Effective max depth** resolution (`cmd/run.go:407–410`): TOML value if in `(0, 5]`, else the system default `3`.

---

## 4. `ExecuteCallAgent()` — the 15-step dispatcher

The heart of delegation is `internal/tool/tool.go:86–360`. It is the handler for the `call_agent` tool call. Flow:

1. Extract `agent`, `task`, `context` from the tool-call arguments (lines 90–92).
2. Validate `agent` non-empty (lines 95–102).
3. Validate `task` non-empty (lines 104–111).
4. **Allowlist check** — the requested name must be in `opts.AllowedAgents` (the parent's `sub_agents` list) (lines 113–127).
5. **Depth check** — `opts.Depth >= opts.MaxDepth` → error (lines 129–136).
6. Load the sub-agent TOML via `agent.Load(agentName, searchDirs)` (lines 149–154).
7. Parse the `provider/model` string (lines 156–160).
8. Resolve workdir, files, skill, system prompt from the sub-agent's own config (lines 164–185).
9. Optionally load memory and append it to the system prompt (lines 187–215).
10. Resolve API key and base URL from global config (lines 217–229).
11. Instantiate the provider (lines 231–235).
12. **Build the sub-agent user message:**

    ```
    Task: <task>

    Context:
    <context>
    ```

    (If `context` is absent, just `Task: <task>`.) (lines 237–243)
13. Build a fresh `Request` with the sub-agent's own tools; **recursively** conditionally inject `call_agent` if the sub-agent itself declares `sub_agents` and `newDepth < maxDepth` (lines 245–307).
14. Run the conversation loop under a timeout context (lines 309–323).
15. Return a `ToolResult` containing the sub-agent's final response text (lines 349–360).

**Error discipline:** every failure path (TOML not found, API error, timeout, depth exceeded, etc.) returns a `ToolResult { is_error: true, content: "Error: sub-agent \"<name>\" failed - <details>. You may retry or proceed without this result." }`. The parent LLM sees the error as a normal tool result and decides whether to retry or proceed. A sub-agent failure never panics the parent.

**Timeout** (`tool/tool.go:309–317`): if `opts.Timeout > 0`, wrap the call in `context.WithTimeout`, else `context.WithCancel` that inherits the parent's context.

---

## 5. Context isolation (important!)

Per `docs/design/sub-agent-pattern.md`, sub-agents are designed for **context isolation**:

- The sub-agent does **not** receive the parent's conversation history.
- Input is only `task` + optional `context`.
- Output is only the final response text, not intermediate turns, tool calls, or reasoning.
- The sub-agent runs with its own system prompt, skill, files, and allowed tools.

This is a deliberate compression boundary — the parent gets a summarized answer, not a transcript.

---

## 6. Conversation loop inside a sub-agent

`internal/tool/tool.go:362–445` mirrors the parent-level loop at `cmd/run.go:522–706`:

```
for turn := 0; turn < 50; turn++ {
    if budget.Exceeded() { break }
    resp := provider.Send(req)
    budget.Add(resp.InputTokens, resp.OutputTokens)
    if len(resp.ToolCalls) == 0 { return resp }
    if budget.Exceeded() { return resp }
    req.Messages = append(req.Messages, assistantMsg(resp))
    results := dispatchTools(resp.ToolCalls, parallel)
    req.Messages = append(req.Messages, toolResultMsg(results))
}
```

**Constants** (`cmd/run.go:35`, `tool/tool.go:26`): `maxConversationTurns = 50`. Exceeding it returns `"sub-agent exceeded maximum conversation turns (50)"`.

llm-rs's `chain()` already implements this loop with `chain_limit` — porting would mean wiring the sub-agent spawn as a `ToolExecutor` whose `execute()` internally calls `chain()` again with a fresh message stack.

---

## 7. Budget tracking across parent + children

**Design:** single shared `BudgetTracker` instance flows from the root invocation into every nested `ExecuteCallAgent`, usually via an `Arc<Mutex<>>`-equivalent in Go (`internal/budget/budget.go`):

```go
type BudgetTracker struct {
    mu         sync.Mutex
    maxTokens  int // 0 = unlimited
    usedTokens int
}
```

- Every LLM response at every depth level calls `tracker.Add(resp.InputTokens, resp.OutputTokens)`.
- Before each provider call, the loop checks `tracker.Exceeded()` and exits gracefully.
- CLI flag `--max-tokens <int>` overrides TOML `budget.max_tokens`.

**Important:** budget is **shared globally across the entire call tree**, not per agent. A runaway child agent burns the parent's budget.

**Exit code planning:** the spec in `docs/idea/unix-philosophy-agent-in-rust.md` reserves **exit code 4 = budget exceeded** and an `LLM_BUDGET_REMAINING` env var for subprocess-style child dispatch. In the current Go implementation, axe uses in-process sharing of the tracker instead of env-var plumbing — the env var is only needed if/when children are run as OS subprocesses. llm-rs can adopt the same in-process approach because `chain()` already takes a `budget` parameter.

---

## 8. Agent discovery for children

`internal/agent/` exposes `BuildSearchDirs(flagAgentsDir, agentsBase)`. Resolution order:

1. `--agents-dir` CLI flag (if passed).
2. The parent agent's workdir (so a parent can ship children co-located with it).
3. Global config dir (`~/.config/axe/agents/`).

Sub-agents inherit the parent's `agentsBase` (`tool/tool.go:150`), meaning a parent agent's relative-to-itself agent references keep working when called recursively.

llm-rs's `discover_agents(global_dir, local_dir)` already covers cases (1) and (3); the "parent workdir" case (2) is the new wrinkle for delegation.

---

## 9. Parallel vs sequential tool dispatch

`cmd/run.go:914–950`:

- Default: `parallel = true` (Requirement 5.1).
- If `parallel == true && len(toolCalls) > 1` → one goroutine per call, results collected via an indexed channel to preserve order, sent back as a **single** tool-result message.
- If `parallel == false || len(toolCalls) == 1` → sequential for-loop.

This maps cleanly onto llm-rs's existing `ParallelConfig` + `dispatch_tools()` (Phase 9). The only new consideration is that `call_agent` results can be large (nested agent output), so unlimited concurrency could multiply memory pressure — `max_concurrent` matters.

---

## 10. Observability

Axe logs to stderr under `--verbose`:

```
[sub-agent] Calling "<name>" (depth <N>) with task: <first 80 chars>...
[sub-agent] "<name>" completed in <ms>ms (<chars> chars returned)
[sub-agent] "<name>" failed: <error>
```

Source: `internal/tool/tool.go:139–145, 326–352`.

Dry-run mode (`cmd/run.go:342–343`) prints a "Sub-Agents" section of the resolved context (max_depth, parallel, timeout, allowlist) with zero tokens consumed. This is a near-perfect fit for extending llm-rs's `DryRunReport` — the `parallel_tools` / `max_parallel_tools` it already surfaces need to be joined by `sub_agents`, `max_depth`, and `sub_agent_timeout`.

JSON output (`llm agent run --json`-equivalent) wraps each tool call with cumulative token tracking and per-call duration, giving the parent visibility into which child consumed what.

---

## 11. Key constants and limits

| Constant | Value | File | Purpose |
|---|---|---|---|
| `maxConversationTurns` | 50 | `cmd/run.go:35`, `tool/tool.go:26` | Safety limit per agent run |
| `CallAgentToolName` | `"call_agent"` | `toolname/toolname.go` | Tool name |
| Default `MaxDepth` | 3 | `cmd/run.go:407` | System default nesting |
| Hard cap `MaxDepth` | 5 | `agent/agent.go:109` | Validation ceiling |
| Default `Parallel` | `true` | `cmd/run.go:509` | Multiple tool calls → goroutines |
| Default `Timeout` | 0 (inherit) | TOML schema | Per sub-agent wall-clock cap |
| `maxToolOutputBytes` | 1024 | `cmd/run.go:37` | Tool result truncation |

---

## 12. Wire formats

**Provider request (parent → LLM) with tool injected:**

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "system": "<parent system prompt>",
  "messages": [{"role": "user", "content": "<parent task>"}],
  "tools": [{"name": "call_agent", "description": "...", "input_schema": {...}}],
  "temperature": 0.3,
  "max_tokens": 4096
}
```

**LLM response requesting delegation:**

```json
{
  "content": "",
  "tool_calls": [{
    "id": "toolu_01ABC",
    "name": "call_agent",
    "arguments": {"agent": "test-runner", "task": "run the unit tests", "context": "changes touched crates/llm-core"}
  }],
  "stop_reason": "tool_use",
  "input_tokens": 100,
  "output_tokens": 50
}
```

**Sub-agent user message (constructed internally):**

```
Task: run the unit tests

Context:
changes touched crates/llm-core
```

**Tool result fed back to parent:**

```json
{"role": "tool", "tool_results": [{
  "call_id": "toolu_01ABC",
  "content": "<sub-agent final response text>",
  "is_error": false
}]}
```

---

## 13. Exit-code contract (planned, from spec)

From `docs/idea/unix-philosophy-agent-in-rust.md`:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Runtime error |
| 2 | Config error |
| 3 | Provider/API error |
| 4 | **Budget exceeded** |

llm-rs already uses 0/1/2/3 per `CLAUDE.md`. Adding 4 for budget exhaustion is a one-line change in `main.rs` plus a `ChainResult.budget_exhausted → exit 4` mapping — it belongs in the same commit as sub-agent delegation per the roadmap.

---

## 14. Mapping to llm-rs

Every axe sub-agent primitive has a corresponding seam already in llm-rs. The port is mostly composition:

| Axe primitive | llm-rs equivalent / gap |
|---|---|
| `CallAgentTool()` | New built-in tool alongside `llm_version` / `llm_time`. Register in the `ToolExecutor` registry. |
| `ExecuteCallAgent()` (15 steps) | New `CallAgentExecutor` implementing `ToolExecutor`. Internally: resolve child via `resolve_agent()`, build `Prompt`, spawn `chain()` recursively. |
| `SubAgentsConfig { max_depth, parallel, timeout }` | Add to `AgentConfig` (currently only `sub_agents: Vec<String>` stub). |
| Depth counter | Thread an explicit `depth: usize` parameter through `chain()` (or a `ChainContext` struct), inject `call_agent` only when `depth < max_depth`. |
| Allowlist check | Use the parent's `AgentConfig.sub_agents` as the allowlist. |
| Shared budget | Already there — `chain()` takes `budget: Option<u64>`. Wrap in `Arc<Mutex<u64>>` (or atomic) and thread through recursive calls. |
| 50-turn safety cap | Already there — `chain_limit`. Keep a hard ceiling so a buggy child can't loop forever. |
| Parallel dispatch | `ParallelConfig` already exists; `call_agent` calls benefit automatically. Respect `--tools-approve → sequential`. |
| Memory injection | `MemoryConfig` stub exists; wiring is Tier 3 anyway. |
| Verbose `[sub-agent]` logs | New `ChainEvent::SubAgent { name, depth, phase }` variant, formatted in `format_chain_event()`. |
| Dry-run surfacing | Extend `DryRunReport` with resolved `sub_agents` + `SubAgentsConfig`. |
| Exit code 4 | Map `ChainResult.budget_exhausted == true` to `process::exit(4)` in `main.rs`. |
| `LLM_BUDGET_REMAINING` env var | **Only needed if** children are ever spawned as subprocesses. In-process recursion via `chain()` does not need it. Defer. |

---

## 15. Minimum MVP for llm-rs

1. `AgentConfig::sub_agents_config: SubAgentsConfig` (depth/parallel/timeout), with the same validation rules as axe.
2. A `CallAgentExecutor` struct implementing `ToolExecutor`, carrying `Arc` references to agent search dirs, global config, provider registry, budget tracker, and current depth.
3. A `call_agent` tool definition, conditionally injected in `build_prompt()` when `sub_agents` is non-empty and `depth < effective_max_depth`.
4. Recursive `chain()` invocation inside the executor's `execute()`, building the child `Prompt` with the `Task: ... / Context: ...` template.
5. Error discipline: every failure becomes a `ToolResult { is_error: true, content: "..." }`, never a panic.
6. Budget continuity: pass the same `Arc`-wrapped budget counter into the recursive `chain()` so child token use decrements the shared pool.
7. Depth tracking: thread an explicit `depth` parameter through `chain()` (or add it to `ChainEvent` for observability).
8. `DryRunReport` extension: surface `sub_agents`, `max_depth`, `parallel`, `sub_agent_timeout`.
9. Exit code 4 wired in `main.rs` from `ChainResult.budget_exhausted`.
10. Integration test: a parent agent that invokes a child agent via wiremock for the LLM side, asserting depth increments, allowlist enforcement, shared budget decrement, and error-result fallback.

**Explicitly deferred:** subprocess-spawned children, `LLM_BUDGET_REMAINING` env var, memory wiring (own Tier 3 item), artifact system (separate axe feature not in llm-rs scope).

---

## 16. Source map (quick index)

```
~/Desktop/axe/
├── cmd/run.go
│   ├── 35          maxConversationTurns = 50
│   ├── 37          maxToolOutputBytes = 1024
│   ├── 342–343     dry-run sub-agent section
│   ├── 406         depth = 0 at top-level
│   ├── 407–410     effectiveMaxDepth resolution
│   ├── 421–424     call_agent tool injection gate
│   ├── 509–511     parallel default = true
│   ├── 522–706     parent-level conversation loop
│   └── 914–950     parallel vs sequential dispatch
├── internal/tool/tool.go
│   ├── 23–74       CallAgentTool() definition
│   ├── 28–45       ExecuteOptions struct (depth, max_depth, budget, timeout, allowlist)
│   ├── 86–360      ExecuteCallAgent() 15-step handler
│   ├── 130–136    depth limit check
│   ├── 139–145    verbose "[sub-agent] Calling..." log
│   ├── 255         newDepth = opts.Depth + 1
│   ├── 309–317    timeout context wrap
│   ├── 326–352    verbose completion/failure logs
│   └── 362–445    sub-agent internal conversation loop
├── internal/agent/agent.go
│   ├── 58–63      SubAgentsConfig struct
│   └── 106–114    validation (max_depth ≤ 5, timeout ≥ 0)
├── internal/budget/budget.go
│   └── *          BudgetTracker with mutex
├── internal/toolname/toolname.go
│   └── *          CallAgentToolName = "call_agent"
├── docs/design/sub-agent-pattern.md    — context isolation + resilience principles
├── docs/design/agent-config-schema.md  — TOML schema
└── docs/idea/unix-philosophy-agent-in-rust.md
    └── *          exit code 4 and LLM_BUDGET_REMAINING env var contract
```
