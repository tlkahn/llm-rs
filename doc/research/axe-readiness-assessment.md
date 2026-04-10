# Axe Feature Readiness Assessment

Assessment of llm-rs Phase 2 architecture against the Unix-philosophy agent features described in `axe/docs/idea/unix-philosophy-agent-in-rust.md`.

## What Works Well

**The `ToolExecutor` trait is the right seam.** `llm-core/src/chain.rs:11` defines `async fn execute(&self, call: &ToolCall) -> ToolResult` — this is exactly where CLI tool dispatch (fork/exec, capture stdout/stderr/exit code) plugs in. A new `ExternalToolExecutor` alongside the existing `CliToolExecutor` requires no trait changes.

**The `chain()` loop structure is sound.** The iterate -> collect tool calls -> execute -> loop pattern at `chain.rs:31-89` matches the ReAct loop axe needs. The `chain_limit`, `on_chunk` callback for streaming output, and `ChainResult` with full chunk/tool_result history are all directly usable.

**Streaming chunk types already handle the hard case.** `Chunk::ToolCallStart`, `ToolCallDelta`, and the `collect_tool_calls()` assembler at `stream.rs:35-73` solve the "stream text to terminal while buffering tool calls internally" problem. You can stream and accumulate simultaneously — the code already does this.

**Tool types are right.** `Tool { name, description, input_schema }`, `ToolCall { name, arguments, tool_call_id }`, `ToolResult { name, output, tool_call_id, error }` — these map 1:1 to the axe tool contract. External tools return plain text on stdout; wrapping into `ToolResult` is straightforward. The error field already handles non-zero exit codes.

**Both providers construct multi-turn messages** from `Prompt.tool_calls` + `Prompt.tool_results`, each following their API's conventions (OpenAI `tool` role, Anthropic `tool_result` blocks in user messages).

## What Doesn't Work

### 1. Conversation history is one turn deep (the critical gap)

The `chain()` loop at line 76-83 rebuilds the prompt each iteration:

```rust
let mut next_prompt = Prompt::new(&current_prompt.text)
    .with_tools(current_prompt.tools.clone())
    .with_tool_calls(tool_calls)
    .with_tool_results(tool_results);
```

This carries only the **latest** tool calls/results. On turn 3, the LLM has no memory of turn 1's tool interactions. The providers build a mini-conversation from these fields, but it's always: `[user, assistant+tool_calls, tool_results]` — never the full `[user, assistant, tool, assistant, tool, ...]` accumulation that a multi-turn ReAct loop requires.

**The `Prompt` type itself is the bottleneck.** It has a single `text: String`, not a `messages: Vec<Message>`. There's no way to represent "here are 6 prior turns of conversation." The `Provider::execute()` signature takes `&Prompt`, so this constraint propagates everywhere.

**What's needed:** Either extend `Prompt` with a `messages` field (and teach both providers to use it), or introduce a separate `Conversation` / `Messages` type that `chain()` accumulates into.

### 2. No external CLI tool dispatch

`CliToolExecutor` at `tools.rs:104` delegates everything to `BuiltinToolRegistry::execute_tool()`, which is a match on hardcoded tool names. There is no:
- Fork/exec of CLI commands
- stdout/stderr/exit code capture
- Argument serialization (JSON args -> CLI flags/positional args)
- Timeout handling
- Parallel tool execution (the chain loop at `chain.rs:67-71` runs tools sequentially with `for call in &tool_calls`)

The `ToolExecutor` trait supports this — it's just not implemented. An `ExternalToolExecutor` that does `Command::new(name).args(...).output()` is straightforward, but the argument mapping (JSON object -> CLI args) needs a convention.

### 3. No `--messages` input or `--json` output

The axe doc identifies this: `llm --messages history.jsonl --tools tools.json --json`. Currently:

- **Input:** `prompt.rs:219-236` resolves text from arg or stdin. No way to pass a conversation history.
- **Output:** Text streamed to stdout, no structured JSON envelope. Axe wants a trace with model, content, tokens, duration, and per-tool-call details.

Without these, the binary can't be used as a sub-agent (child `llm agent run` needs to receive context and return structured results).

### 4. No budget/token tracking across turns

`collect_usage()` extracts usage from chunks, but `chain()` doesn't accumulate token counts across iterations. There is no budget concept — no `max_tokens` budget that spans the whole loop, no `LLM_BUDGET_REMAINING` env var for sub-agents, no exit-code-4 on budget exhaustion.

### 5. No agent config or discovery

No TOML agent config parsing. No `<cwd>/.llm/agents/` -> `~/.config/llm/agents/` shadowing. No `llm agent run <name>` subcommand. This is new code, not a refactor of existing code — but the `Config` system in `llm-core/config.rs` (TOML + XDG paths) provides the pattern to follow.

## Summary Table

| axe Feature | Current State | Work Required |
|---|---|---|
| ReAct loop | `chain()` exists, works | Minor — fix history accumulation |
| Multi-turn messages | **Missing** — `Prompt` is single-turn | **Major** — core type change, both providers |
| External CLI tools | `ToolExecutor` trait exists | Medium — new executor impl, arg mapping |
| Parallel tool exec | Sequential `for` loop | Small — `tokio::join!` / `JoinSet` |
| `--messages` input | Not exposed | Medium — new CLI flag + parser |
| `--json` output | Not exposed | Medium — structured envelope |
| Agent TOML config | Not started | Medium — new module, follows existing config pattern |
| Sub-agent delegation | Not started | Medium — depends on `--messages` + `--json` |
| Budget tracking | Not started | Small-medium — accumulate usage in `chain()` |
| Dry-run mode | Not started | Small — resolve config, print, exit |
| Retry/backoff | Not started | Small — wrap provider calls |
| Memory system | Not started | Medium — new module |

## Bottom Line

The architecture is pointed in the right direction — `ToolExecutor`, `chain()`, the streaming chunk types, and the provider abstractions are all the right shapes. The single biggest blocker is that `Prompt` is a single-turn type with no conversation history. Fix that (and propagate through the providers), and most of the axe features become straightforward additions on top of what exists.
