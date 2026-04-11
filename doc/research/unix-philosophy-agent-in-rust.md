# Unix-Philosophy Agent Architecture: Shell Tools + Rust Core Loop

> **Status: Archived research.** This document explored Unix-philosophy agent design ideas that informed phases 5–9 of llm-rs. The `call_agent` / sub-agent delegation ideas in this document were ultimately not adopted. See [specialist-tools-vs-sub-agents.md](specialist-tools-vs-sub-agents.md) for the current position.

## Origin

This idea grew out of comparing the ReAct loop implementations in two projects:

- **deep_research_from_scratch** — Python/LangGraph declarative state machine
- **axe** — Go imperative for-loop in `cmd/run.go`

Both implement the same pattern (LLM call -> check tool calls -> execute tools -> loop) but differ in how they express it. The question was: can we go further down the Unix philosophy path and build the agent loop around CLI commands as first-class tools, using `llm-rs` (a Rust LLM CLI at `~/Projects/llm-rs`) as the core LLM interface?

## The Idea

Write the core ReAct loop in Rust (extending `llm-rs`), but treat **everything outside the loop** as external processes:

- Tools are CLI commands on `$PATH`
- Sub-agents are child `llm` processes
- Memory is a pluggable backend (filesystem, SQLite, Redis) accessed via CLI
- Agent definitions are TOML config files, not code

The original thought was to write the loop itself in fish shell, but research showed that JSON escaping and state management in shell consume 60%+ of the effort. The revised plan: keep the ~50-line core loop in Rust where JSON handling is reliable, push tool execution to external commands where Unix composability shines.

## Why This Makes Sense (Evidence)

### CLI > MCP (2026 Benchmarks)

- **Scalekit benchmark**: 75 tests across 5 GitHub tasks. CLI achieved 100% reliability vs MCP's 72%. CLI was 4-32x cheaper on tokens. Monthly cost at 10K operations: CLI ~$3.20 vs MCP ~$55.20.
- **Perplexity CTO** announced moving away from MCP internally; benchmarks showed MCP consuming 15-20x more tokens than CLI for equivalent tasks.
- **Anthropic internal research** found writing shell scripts instead of calling MCP tools cut token usage by 98.7%.

Core argument: LLMs trained on millions of shell script examples. The composability grammar is baked into the weights.

### Existing Projects Proving the Pattern

| Project | Approach | Key Insight |
|---------|----------|-------------|
| learn-claude-code | Python loop + bash tool | "One loop & Bash is all you need" — core loop is ~30 lines |
| Ralph | Bash agent loop | State persists via git + append-only files; fresh context each iteration |
| llm-functions (sigoden) | Shell scripts with annotations | Tools as shell scripts with comment-based JSON schema generation |
| Fabric (Miessler) | Unix pipelines | 200+ reusable AI prompt patterns composed via stdin/stdout |
| Simon Willison's llm | Python CLI | Auto-logs to SQLite, composable with pipes |
| Butterfish | Go shell wrapper | Shell history becomes AI context naturally |

## What `llm-rs` Needs First

The `llm-rs` binary has full tool calling infrastructure at the library level (`Tool`, `ToolCall`, `ToolResult` types, streaming tool call deltas via SSE chunks), but **none of it is exposed in the CLI yet**:

1. **No `--tools` flag** — can't pass tool definitions (JSON schema) to the model
2. **No structured JSON output** — can't reliably parse tool call responses
3. **No context/message passing** — each invocation is stateless; no way to send prior messages via stdin or flags
4. **No tool result injection** — can't feed tool results back for the next turn

Minimum viable interface needed:

```bash
echo "$messages_json" | llm --tools tools.json --output-format json
# Or:
llm --messages history.jsonl --tools tools.json --json
```

## Proposed Architecture

```
agent.toml              (declarative config — what the agent IS)
    │
    ▼
llm-rs `agent run`      (Rust binary — owns the loop, JSON, dispatch)
    │
    ├── tool call ──► CLI tools on $PATH (ripgrep, sqlite3, curl, jq, custom scripts)
    │                    └── stdout back to loop
    │
    ├── call_agent ──► child `llm agent run` process (isolated context)
    │                    └── stdout back to parent
    │
    ├── append ──► memory backend (file / sqlite / redis — swappable)
    │
    ├── --json ──► structured trace (tool calls, timing, tokens)
    │
    └── exit 0-4 ──► final output on stdout
```

### What the Rust Binary Owns

- The ReAct loop (LLM call -> tool dispatch -> loop)
- JSON construction and parsing (the #1 failure mode in shell-based agents)
- Tool dispatch (fork/exec CLI commands, capture stdout/stderr/exit code)
- Budget/token tracking (shared across parent + sub-agents via env var)
- Retry logic with exponential backoff + jitter
- Agent TOML config parsing
- Streaming output for human-facing mode, buffered JSON for programmatic mode

### What External Processes Own

- Actual tool execution (ripgrep, sqlite3, curl, custom scripts — anything on `$PATH`)
- Sub-agent execution (child `llm agent run` processes with their own config)
- Memory storage (filesystem, SQLite, Redis — accessed via their native CLIs)

## Features to Borrow from Axe

### Tier 1: Steal Immediately

**Agent-as-TOML.** An agent is a config file, not code. Versionable in git, shareable via `cp`. Different projects get different agents via local agent directories (`<cwd>/.llm/agents/` shadows global `~/.config/llm/agents/`).

```toml
model = "anthropic/claude-sonnet-4-20250514"
system_prompt = "You are a code reviewer."
tools = ["ripgrep", "read_file", "gh"]
sub_agents = ["security-checker"]

[memory]
enabled = true
last_n = 10

[budget]
max_tokens = 50000
```

**Sub-agent delegation with context isolation.** Parent never sees child's internal conversation, only final stdout. Depth limit (default 3, hard max 5). Shared token budget across parent + all children. Parallel sub-agents when LLM makes multiple `call_agent` calls in one turn — just background processes, the OS does this for free.

**Dry-run mode.** `llm agent run reviewer --dry-run` shows resolved context (system prompt + skill + files + memory + tools) without making an LLM call. Zero tokens. Essential for debugging agent configs.

### Tier 2: High-Value, Low-Effort

**Append-only markdown memory + LLM-powered GC.** One file per agent (`~/.local/share/llm/memory/<agent>.md`). Each run appends a timestamped entry. Next run loads last N entries into context. `llm gc <agent>` feeds full log to LLM for pattern detection, then trims. Memory maintenance itself is an LLM task.

**JSON output envelope with tool call trajectory.** `--json` returns structured trace: model, content, tokens, duration, and per-tool-call details (name, input, output, turn, timing). Machine-readable observability for free.

**Budget tracking with exit code 4.** Cumulative tokens (input + output) across all turns and sub-agents. Shared budget passed to children via `LLM_BUDGET_REMAINING` env var. Budget exceeded = exit code 4 (current response completes, no further tool calls).

**Retry with exponential backoff + jitter.** Retry 429s and 5xx, never retry 401/403/400. Belongs in the Rust binary, not the shell layer.

### Tier 3: Nice-to-Have

**Artifact system.** Shared scratch directory for multi-agent pipelines. Sub-agents inherit parent's artifact dir via `LLM_ARTIFACT_DIR` env var. Less critical because Unix already has `/tmp` and pipes.

**Refusal detection.** Pattern-match "I cannot", "I'm unable to" etc. and set a `refused: true` flag. 20 lines, no LLM call, saves you from silently accepting refusals.

**SSRF protection on url_fetch.** Block all private IP ranges regardless of config. DNS resolution before connection to prevent rebinding.

### Skip These

**MCP support.** Opposite direction from Unix philosophy. CLI beats MCP on reliability (100% vs 72%) and cost (4-32x cheaper).

**Compiled-in tools.** Axe bakes tools into the binary. Our whole point is that tools are external commands.

**Golden file testing.** Useful for deterministic compiled binaries, less so for composed shell agents. Test the components instead.

## Swappable Memory Backends

The strongest differentiator of this architecture. The memory interface is always read/append; the backend is pluggable:

| Backend | Read | Append | Tradeoff |
|---------|------|--------|----------|
| Filesystem | `cat ~/.llm/memory/agent.md` | `echo >> file` | Simplest, git-versioned, grep-able |
| SQLite | `sqlite3 db "SELECT ..."` | `sqlite3 db "INSERT ..."` | FTS5 search, ACID, single file |
| Redis | `redis-cli GET` | `redis-cli APPEND` | Fast, TTL expiry, pub/sub for multi-agent |
| JSONL | `jq 'select(...)' file.jsonl` | `jq -c '.' >> file.jsonl` | Structured, streaming-friendly |

AgentFS (Turso) even FUSE-mounts SQLite so standard file tools work against a database transparently — could be an interesting backend.

## Known Hard Problems

### JSON Escaping in Tool Output

LLM outputs contain literal newlines, unescaped quotes, backslashes, and control characters. This is the #1 failure mode for shell-based agents. Solution: the Rust binary handles all JSON construction. External tools return plain text on stdout; the binary wraps it into valid JSON tool result messages.

### Context Window Growth

Each turn adds messages. In Go/Python you accumulate in a data structure. Here, the Rust binary manages a messages array internally during the loop. For long-running agents, need a compaction strategy:
- Truncate tool output (cap at N characters)
- Summarize periodically (separate `llm` call with a compression prompt, like deep_research's `compress_research` node)
- Fresh context each iteration with state externalized to files (Ralph pattern)

### Streaming vs Loop

`llm-rs` streams by default, but the ReAct loop needs the complete response to check for tool calls. Options:
- `--no-stream` for agent mode (simplest)
- Stream text to terminal while buffering internally (best UX, more complex)
- The Rust binary already has chunk types (`Text`, `ToolCallStart`, `ToolCallDelta`, `Done`) — it can stream text and accumulate tool calls simultaneously

### Fish vs Bash

Fish has better syntax, but:
- No `read -r` equivalent (streaming SSE harder)
- Smaller ecosystem of examples for LLMs to draw on
- Different quoting rules from bash (LLMs sometimes generate bash-isms)

For the loop itself being in Rust, this doesn't matter. For helper scripts and custom tools, fish works fine — a tool is just an executable.

## Tool Contract

Every tool (built-in or custom) follows this contract:

**Input:** Command-line arguments and/or stdin (plain text or JSON, tool's choice)

**Output:** Plain text on stdout (the Rust binary wraps it into JSON)

**Errors:** Non-zero exit code + stderr message

**Example custom tool:**

```fish
#!/usr/bin/env fish
# tools/search-codebase
# Usage: search-codebase <pattern> [--type <filetype>]
rg --json $argv | jq -r '.data.lines.text // empty'
```

The Rust binary's tool dispatch:
1. Parse tool call from LLM response: `{name: "search-codebase", arguments: {pattern: "TODO", type: "py"}}`
2. Fork/exec: `search-codebase TODO --type py`
3. Capture stdout, stderr, exit code
4. Format as tool result message for next LLM turn

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error (I/O, tool failure) |
| 2 | Config error (missing key, bad TOML, unknown model) |
| 3 | Provider/network error (API failure, timeout) |
| 4 | Token budget exceeded |

## Next Steps

1. **Add tool calling to `llm-rs` CLI** — `--tools`, `--json` output with tool calls, `--messages` for context passing
2. **Implement the agent loop** — `llm agent run <name>` subcommand, ~50 lines of Rust
3. **Agent TOML parser** — config loading with local directory discovery
4. **Tool dispatch** — fork/exec with stdout/stderr capture, parallel execution
5. **Memory system** — start with filesystem (append-only markdown), abstract the interface
6. **Sub-agent support** — `call_agent` as a special tool that spawns child `llm agent run`
7. **Budget tracking** — shared via env var across parent + children
8. **Dry-run mode** — show resolved context without LLM call

## References

- [Scalekit: MCP vs CLI Benchmark](https://www.scalekit.com/blog/mcp-vs-cli-use)
- [learn-claude-code: "Bash is all you need"](https://github.com/shareAI-lab/learn-claude-code)
- [Ralph: Bash agent loop with filesystem state](https://github.com/snarktank/ralph)
- [llm-functions: Shell scripts as LLM tools](https://github.com/sigoden/llm-functions)
- [Sketch.dev: The Agent Loop in 9 Lines](https://sketch.dev/blog/agent-loop)
- [CLI is the New MCP](https://oneuptime.com/blog/post/2026-02-03-cli-is-the-new-mcp/view)
- [AgentFS: SQLite-backed FUSE filesystem for agents](https://turso.tech/blog/agentfs-fuse)
- [Simon Willison's llm CLI](https://github.com/simonw/llm)
- [Fabric: Unix pipelines for AI](https://github.com/danielmiessler/Fabric)
