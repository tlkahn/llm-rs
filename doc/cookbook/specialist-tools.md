# Cookbook: Specialist Tools

This cookbook shows how to compose narrow, purpose-specific *specialist tools* on top of the `llm-tool-*` subprocess protocol. Unlike a full sub-agent orchestration runtime, each specialist is an opaque function from the parent LLM's perspective — no recursion, no depth, no shared budget.

For the full rationale — why llm-rs chose this pattern over recursive sub-agent delegation — see [specialist-tools-vs-sub-agents.md](../research/specialist-tools-vs-sub-agents.md). For the wire-level protocol, see [external-tools.md](../spec/external-tools.md).

---

## Example: `llm-tool-run-tests`

A specialist tool that runs `cargo test`, captures the output, and asks a cheap LLM to summarize the failures for the parent agent.

### The tool

Save this as `llm-tool-run-tests` somewhere on `$PATH` and make it executable (`chmod +x`):

```bash
#!/bin/bash
# llm-tool-run-tests — runs cargo test and summarizes failures with a cheap LLM
set -euo pipefail

if [[ "${1:-}" == "--schema" ]]; then
  cat <<'JSON'
{
  "name": "run_tests",
  "description": "Run cargo tests and return a short failure summary.",
  "input_schema": {
    "type": "object",
    "properties": {
      "filter": { "type": "string", "description": "Optional cargo test name filter." }
    },
    "required": []
  }
}
JSON
  exit 0
fi

args=$(cat)
filter=$(echo "$args" | jq -r '.filter // ""')

# Run tests; don't let a non-zero exit kill the script — we want to summarize failures.
output=$(cargo test $filter 2>&1 || true)

# Delegate the summarization to a cheap model. The parent agent never sees this call.
echo "$output" | llm -m gpt-4o-mini -s "Summarize this cargo test output in at most 3 bullets. If all tests passed, say 'all tests passed' and stop."
```

### The parent agent

A minimal parent agent TOML that declares the tool:

```toml
# ~/.config/llm/agents/reviewer.toml
model = "claude-opus-4-6"
system_prompt = "You are a senior Rust reviewer. Use run_tests to check the suite."
tools = ["run_tests"]
```

---

## How it behaves end-to-end

1. You run `llm agent run reviewer "check the unit tests"`.
2. The parent LLM (`claude-opus-4-6`) decides to call `run_tests({"filter": "unit"})`.
3. llm-rs dispatches the tool call: it forks `llm-tool-run-tests` and writes `{"filter":"unit"}` to its stdin.
4. The script runs `cargo test unit`, captures stdout+stderr, and pipes the output into a *separate* `llm` invocation with `gpt-4o-mini`. This second call is a completely independent conversation with its own budget and no shared state with the parent.
5. `gpt-4o-mini` produces a 3-bullet summary and writes it to stdout.
6. The script's stdout becomes the tool result that llm-rs feeds back to `claude-opus-4-6`.
7. The parent LLM continues its turn with that summary as a normal tool result.

From the parent's perspective, exactly one tool was called and one result came back. The nested `llm` invocation is invisible.

---

## Why this is *not* sub-agent delegation

A full `call_agent`-style sub-agent runtime (as in axe) would give you depth tracking, a shared token budget threaded through every nested call, and an allowlist of which agents the parent can spawn. Specialist tools intentionally skip all of that:

- **No depth tracking.** Each subprocess is a leaf. Whether the tool internally calls `llm` once, ten times, or not at all is not llm-rs's concern.
- **Independent budget per invocation.** The nested `llm -m gpt-4o-mini ...` call has its own accounting. If you want to enforce a shared budget, you do it at the shell level (wrap the whole pipeline in a script that sums token usage from `--json` output).
- **Independent model choice.** Cheap specialist + expensive generalist composes naturally without any cross-invocation plumbing.
- **Tool author owns discipline.** If a tool author writes a specialist that calls `llm` with an agent that calls the same specialist back, they have built a recursion trap. llm-rs cannot detect or terminate this — it's on the tool author to keep their specialists well-founded.

For the full rationale and research citations, see [specialist-tools-vs-sub-agents.md](../research/specialist-tools-vs-sub-agents.md).

---

## Limitations and honest caveats

- **No context threading.** Each specialist tool invocation is a fresh conversation. Any context the specialist needs must be passed explicitly as tool arguments.
- **No shared token budget across the whole tree.** If you need hierarchical budget enforcement, do it at the shell level. This is out of scope for llm-rs.
- **Opaque inner turns.** The parent cannot introspect what happened inside the subprocess. Authors can mitigate this by writing structured logs to stderr; users can surface them with `--tools-debug` or `-vv`.
- **No dynamic specialist selection from a large pool.** The parent agent must statically declare `tools = [...]` in its TOML. Dispatching by name at call time from a pool of 50+ specialists isn't something this pattern does cleanly.
- **Recursion discipline is externalized.** llm-rs has no cycle detector for tool-calls-tool-calls-tool chains. Tool authors must ensure their specialists terminate.

---

## Cross-references

- [doc/spec/external-tools.md](../spec/external-tools.md) — authoritative protocol specification
- [doc/research/specialist-tools-vs-sub-agents.md](../research/specialist-tools-vs-sub-agents.md) — design rationale and research citations
- `crates/llm-cli/tests/fixtures/bin/llm-tool-upper` — a minimal test-fixture specialist tool that does not call `llm`, useful as a starting template
