# Specialist Tools: Why llm-rs Does Not Build Sub-Agent Delegation

**Status:** Decided. **Date:** 2026-04-11. **Applies to:** llm-rs v0.9 and beyond.

---

## TL;DR

- Recursive sub-agent delegation (an in-process `call_agent` tool, depth tracking, shared token budget, `LLM_BUDGET_REMAINING` env var, exit-code-4) is **permanently parked**. It is not deferred to a later phase — it is a deliberate architectural divergence from axe.
- The **specialist tool pattern** — an `llm-tool-*` executable that may internally invoke `llm prompt` with a narrow, purpose-specific agent — is llm-rs's answer to hierarchical workflows. Each invocation is an opaque leaf function from the parent LLM's perspective.
- llm-rs is a **library (with a CLI), not an orchestration framework.** Hierarchical workflows compose through specialist tools and shell scripts, not through a first-class agent-tree runtime.

---

## What was considered

Axe (`~/Desktop/axe/`, commit snapshot 2026-04-11) ships a full recursive sub-agent runtime: a built-in `call_agent` tool, `sub_agents = [...]` allowlists on agent TOML, depth tracking against a max depth, a shared token budget threaded through child invocations via `LLM_BUDGET_REMAINING`, exit-code-4 on budget exhaustion, and a memory system (`MemoryConfig`) for per-agent storage. A full feature inventory lives in [axe-sub-agent-delegation.md](axe-sub-agent-delegation.md).

llm-rs has carried `sub_agents: Vec<String>` and `memory: Option<MemoryConfig>` as parsed-but-unwired stub fields on `AgentConfig` since Phase 5, held open for a future "Tier 3" port of this axe feature set.

After evaluating the research landscape and the alternatives available inside llm-rs itself, we are choosing not to build this.

---

## Why parked — research summary

**1. Recursive multi-agent trees amplify errors at high rates.** The MAST study (Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, [arXiv 2503.13657](https://arxiv.org/abs/2503.13657), 2025) annotated 1,600+ traces across seven multi-agent frameworks and found failure rates of 41–86.7% on standard benchmarks. 41.8% of those failures were attributable to system-design issues: role ambiguity, poor decomposition, and missing termination conditions. Unstructured multi-agent networks amplified errors by up to **17.2×** compared to single-agent baselines on the same tasks.

**2. Context fragmentation is the core failure mode.** Cognition's ["Don't Build Multi-Agents"](https://cognition.ai/blog/dont-build-multi-agents) argues that every agent boundary is a context boundary, and every context boundary is an opportunity for the child to lose the information it needs to make a coherent decision. Hierarchical trees multiply this problem at every level.

**3. The research consensus is "add complexity only when simpler patterns fail."** Anthropic's ["Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) explicitly recommends single-agent loops with tools as the default, and reaching for multi-agent orchestration only when measurable gains justify the cost.

**4. Production systems use narrow, constrained sub-agents.** Claude Code's own sub-agent feature is deliberately scoped: read-only, non-parallel, single-turn-ish tasks. It is a specialized tool, not a recursive runtime. That matches what the research suggests works — and it matches the specialist tool pattern below.

---

## The specialist tool pattern

**Definition.** A *specialist tool* is an executable on `$PATH` following the `llm-tool-*` naming convention that internally may invoke `llm prompt` with a narrow, purpose-specific agent. From the parent LLM's perspective it is an opaque function — you call it with arguments, you get a result. From the shell's perspective it is a leaf invocation: no depth, no recursion semantics, no shared state with the parent, no allowlist propagation.

The protocol is already specified in [external-tools.md](../spec/external-tools.md). No new runtime, no new config, no new core types are required. A specialist tool is just a conventional use of the existing Phase 4 extensibility seam.

**Properties:**

- **Opaque leaf.** Parent sees one tool call, one tool result. What happened inside the subprocess — whether it called an LLM, ran a database query, or flipped a coin — is invisible.
- **Independent model choice.** A specialist tool can use `gpt-4o-mini` even when the parent uses `claude-opus-4-6`. Cheap specialist + expensive generalist composes naturally at the shell level.
- **Independent budget.** Each `llm prompt` invocation has its own accounting. No shared counter to thread through subprocess boundaries.
- **Independent conversation state.** Each call is a fresh conversation. No context inheritance, no accidental state leakage.
- **Unix-composable.** It's just a binary on `$PATH`. You can pipe to it, test it in isolation, compose it with shell, replace it with a non-LLM implementation without changing the parent agent.

For a runnable worked example, see [cookbook/specialist-tools.md](../cookbook/specialist-tools.md).

---

## Worked example (brief)

A parent agent declares `tools = ["run_tests"]`. The `run_tests` tool is a bash script on `$PATH` named `llm-tool-run-tests` that runs `cargo test`, captures the output, and internally shells out to a cheap model to summarize the failures:

```bash
#!/bin/bash
# llm-tool-run-tests — runs cargo test and summarizes failures with a cheap LLM
if [[ "$1" == "--schema" ]]; then
  echo '{"name":"run_tests","description":"Run cargo tests and return a short failure summary","input_schema":{"type":"object","properties":{"filter":{"type":"string"}},"required":[]}}'
  exit 0
fi
args=$(cat)
filter=$(echo "$args" | jq -r '.filter // ""')
output=$(cargo test $filter 2>&1)
echo "$output" | llm -m gpt-4o-mini -s "Summarize cargo test output in 3 bullets. If all passed, say so."
```

Parent `agent.toml`:

```toml
model = "claude-opus-4-6"
system_prompt = "You are a senior Rust reviewer."
tools = ["run_tests"]
```

The parent LLM emits `run_tests({"filter": "unit"})`, the subprocess runs, the inner `llm` call happens with a cheaper model, the summary returns on stdout, and the parent sees it as a normal tool result. No depth tracking. No shared budget. No `sub_agents` allowlist. It just works because the existing subprocess protocol was already doing the right thing.

Full walkthrough with stderr/debug handling and limitations: [cookbook/specialist-tools.md](../cookbook/specialist-tools.md).

---

## What this gives up honestly

Specialist tools are not a drop-in replacement for a full sub-agent runtime. The pattern explicitly trades the following:

1. **No dynamic specialist selection from a large pool.** A parent agent with 50 possible sub-agents would need to declare 50 tools. A runtime with `sub_agents = [...]` could dispatch by name at call time.
2. **No context threading.** Each specialist tool call is a fresh conversation. Shared context must be passed explicitly as tool arguments.
3. **Opaque inner turns.** The parent cannot see what happened inside the subprocess. Mitigated by the tool author emitting structured stderr and users running with `--tools-debug` or `-vv`.
4. **No hierarchical token budget enforcement.** llm-rs has no shared counter that threads through nested invocations. Users who need this must do shell-level accounting (e.g., wrap calls in a script that sums token usage from `--json` output, or use provider-side rate limiting). **This is out of scope.**
5. **Tool authors can still build hidden recursion trees.** If `llm-tool-foo` internally calls `llm prompt` with an agent that declares `tools = ["foo"]`, the tool author has built a recursive tree that llm-rs cannot see or terminate. This discipline is externalized to tool authors — it's their responsibility to terminate correctly, not the runtime's.

We are accepting these trade-offs deliberately. The alternative is a recursive multi-agent runtime with the failure modes documented in the research above.

---

## Position statement

**llm-rs is a library (with a CLI), not an orchestration framework.** It provides a solid single-agent loop with tools, conversations, structured output, parallel tool dispatch, retries, budgets, and dry-run, packaged as a CLI and exposed as a Rust library, a WASM module, and a Python module. Hierarchical workflows compose through specialist tools and shell scripts layered on top of that core, not through a first-class agent-tree orchestration runtime inside the binary.

---

## Sources

- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* — [arXiv 2503.13657](https://arxiv.org/abs/2503.13657)
- Cognition, *Don't Build Multi-Agents* — https://cognition.ai/blog/dont-build-multi-agents
- Anthropic, *Building Effective Agents* — https://www.anthropic.com/research/building-effective-agents
- Claude Code sub-agent documentation — https://docs.claude.com/en/docs/claude-code/sub-agents

## Related docs

- [axe-sub-agent-delegation.md](axe-sub-agent-delegation.md) — feature inventory of what axe ships (archived research)
- [unix-philosophy-agent-in-rust.md](unix-philosophy-agent-in-rust.md) — earlier design thinking (archived research)
- [external-tools.md](../spec/external-tools.md) — the protocol specialist tools use
- [cookbook/specialist-tools.md](../cookbook/specialist-tools.md) — runnable worked example
- [roadmap.md](../roadmap.md) — parked items index
