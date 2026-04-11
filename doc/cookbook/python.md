# Cookbook: Python (`llm-rs` native module)

`llm-python` exposes `llm-core` to Python as a native PyO3 extension built with [maturin](https://www.maturin.rs/). You get a single `LlmClient` class that talks to OpenAI, Anthropic, or any OpenAI-compatible endpoint, with a real Python iterator for streaming. No `httpx`, no `openai` package, no async event loop in your code — the Rust side runs Tokio internally and hands plain strings back across the FFI boundary.

Phase A of the wrapper extension is in: you also get **tool calling** (Python functions exposed to the model via a decorator), **multi-turn `Conversation`s** with shared message history, **structured output** via either a JSON-Schema dict or LLM-RS's terse schema DSL, and the two built-in tools (`llm_version`, `llm_time`). Retries, budgets, persistent logs, and chain observability events are still CLI-only — see the bottom of this file for what's still missing.

For the underlying class definitions, see [`crates/llm-python/src/lib.rs`](../../crates/llm-python/src/lib.rs), [`tools.rs`](../../crates/llm-python/src/tools.rs), and [`conversation.rs`](../../crates/llm-python/src/conversation.rs).

---

## Build & install

```bash
cd crates/llm-python
uv run maturin develop          # builds and installs into the local .venv
uv run python -c "import llm_rs; print(llm_rs.LlmClient)"
```

`maturin develop` produces an editable install — re-run it after every Rust change. For a release wheel, use `uv run maturin build --release`.

> **Python 3.13 is the current target.** The pinned `pyo3 = "0.23"` does not yet support Python 3.14; either use the project's `.venv` (which `uv` creates at 3.13) or pass `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` and accept the warning.

---

## The whole API in 30 seconds

```python
import llm_rs

# Constructor — provider auto-detected from model name unless overridden.
client = llm_rs.LlmClient(
    api_key="sk-…",
    model="gpt-4o-mini",       # "claude-*" routes to Anthropic
    provider=None,             # "openai" | "anthropic" to force
    base_url=None,             # point at any OpenAI-compatible host
    log_dir=None,              # currently a no-op placeholder
    chain_limit=5,             # max chain iterations when tools are registered
)

# Blocking call, returns the full text.
text = client.prompt("Write a haiku about Python.", system=None)

# Streaming — returns a normal Python iterator yielding text chunks.
for chunk in client.prompt_stream("Tell me a story.", system=None):
    print(chunk, end="", flush=True)

# Register a Python function as a tool — schema inferred from type hints.
@client.tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

client.enable_builtin_tools()                   # llm_version, llm_time
print(client.prompt("What is 17 + 25? Use the add tool."))

# Multi-turn conversation, sharing the client's tool registry.
conv = llm_rs.Conversation(client)
conv.send("My name is Ada.")
print(conv.send("What did I just tell you?"))

# Structured output via the schema DSL.
print(client.prompt("Marcus, 34, engineer.", schema="name str, age int"))
```

Every recipe below is a remix of those calls.

---

## Recipe 1: One-liner CLI summarizer

Pipe anything into a 5-line script and get a TL;DR back. Drop this in `~/.local/bin/tldr` and `chmod +x` it.

```python
#!/usr/bin/env python3
import os, sys, llm_rs

c = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini")
print(c.prompt(sys.stdin.read(), system="Summarize in 5 bullet points."))
```

```bash
$ curl -s https://en.wikipedia.org/wiki/Webassembly | tldr
- WebAssembly (Wasm) is a binary instruction format …
```

---

## Recipe 2: Streaming progress with `rich`

`prompt_stream` is a real iterator, so it composes with any progress library. This snippet shows the response growing live inside a [Rich](https://rich.readthedocs.io/) panel.

```python
import os, llm_rs
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

client = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini")
console = Console()
buf = ""

with Live(Panel("", title="thinking…"), console=console, refresh_per_second=24) as live:
    for chunk in client.prompt_stream("Explain async/await like I'm five."):
        buf += chunk
        live.update(Panel(buf, title="gpt-4o-mini"))
```

The Rust side spawns a Tokio task that fills an `mpsc` channel; `__next__` blocks on `recv()`. So you can also drop the iterator into `concurrent.futures` if you want background streaming.

---

## Recipe 3: Compare providers head-to-head

Two clients, same prompt, parallel HTTP. Threads are fine here — the underlying Tokio runtime is per-client and blocking calls release the GIL.

```python
import os, concurrent.futures as cf, llm_rs

oa = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"],    "gpt-4o-mini")
an = llm_rs.LlmClient(os.environ["ANTHROPIC_API_KEY"], "claude-haiku-4-5")

q = "In one sentence: what is the actor model?"

with cf.ThreadPoolExecutor() as pool:
    a, b = pool.map(lambda c: c.prompt(q), [oa, an])

print(f"OpenAI    → {a}")
print(f"Anthropic → {b}")
```

Add a third client pointed at a local llama.cpp server with `base_url="http://localhost:8080"` and you have a 3-way bench.

---

## Recipe 4: Batch-summarize a folder of markdown

A practical "process my notes" script. Walks a directory, summarizes each file, and writes the summary alongside it as `<file>.tldr.md`.

```python
import os, sys, pathlib, llm_rs

ROOT = pathlib.Path(sys.argv[1])
client = llm_rs.LlmClient(
    os.environ["OPENAI_API_KEY"], "gpt-4o-mini",
)
SYSTEM = (
    "You write 3-bullet summaries of markdown notes. "
    "Keep technical terms verbatim. No preamble."
)

for md in ROOT.rglob("*.md"):
    if md.name.endswith(".tldr.md"):
        continue
    out = md.with_suffix(".tldr.md")
    if out.exists() and out.stat().st_mtime > md.stat().st_mtime:
        continue                                # already up to date
    print(f"… {md}")
    out.write_text(client.prompt(md.read_text(), system=SYSTEM))
```

Skip-when-fresh logic + a single client instance keeps a 500-file vault under a couple of dollars on `gpt-4o-mini`.

---

## Recipe 5: "Explain this stack trace"

Pipe the *previous* command's output through an LLM. Two lines of fish/zsh + one Python helper:

```python
# ~/.local/bin/whatfailed
#!/usr/bin/env python3
import os, sys, llm_rs

c = llm_rs.LlmClient(os.environ["ANTHROPIC_API_KEY"], "claude-sonnet-4-6")
print(c.prompt(
    sys.stdin.read(),
    system=(
        "You are a senior engineer. Given a stack trace or error log, "
        "give: (1) one-line root cause guess, (2) the smallest fix to try, "
        "(3) a follow-up question if you're not sure."
    ),
))
```

```fish
$ cargo test 2>&1 | whatfailed
```

Pairs nicely with `set -o pipefail` and a shell alias like `alias huh='fc -ln -1 | sh 2>&1 | whatfailed'`.

---

## Recipe 6: Jupyter cell magic

Turn any notebook cell into a prompt. Drop this into a setup cell and you get `%%ask`:

```python
import os, llm_rs
from IPython.core.magic import register_cell_magic
from IPython.display import Markdown, display

_client = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini")

@register_cell_magic
def ask(line, cell):
    """Usage:  %%ask  [optional system prompt]"""
    text = ""
    for chunk in _client.prompt_stream(cell, system=line or None):
        text += chunk
    display(Markdown(text))
```

```
%%ask You are a SQL tutor. Explain the query, do not rewrite it.
SELECT user_id, COUNT(*)
FROM events
WHERE ts > now() - interval '7 days'
GROUP BY 1
HAVING COUNT(*) > 10;
```

Because `prompt_stream` accumulates fully before display, you also get a final, well-rendered Markdown cell — no half-formatted intermediate states.

---

## Recipe 7: Async generator wrapper for FastAPI

The native `prompt_stream` is *blocking* (it `recv()`s on a Rust channel). To stream from an async web framework, push the iterator to a thread and bridge it with `asyncio.to_thread` per chunk:

```python
import asyncio, os, llm_rs
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
client = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini")

async def stream(question: str):
    it = iter(client.prompt_stream(question))

    def _next():
        try:
            return next(it)
        except StopIteration:
            return None

    while True:
        chunk = await asyncio.to_thread(_next)
        if chunk is None:
            return
        yield chunk

@app.get("/ask")
async def ask(q: str):
    return StreamingResponse(stream(q), media_type="text/plain")
```

Each chunk hop goes thread → asyncio → SSE without ever holding the GIL during the network read. Good enough for a personal project; for serious throughput, run the CLI behind a real worker pool.

---

## Recipe 8: Self-hosted / proxy endpoints

`base_url` accepts any OpenAI-compatible host. Useful for Azure OpenAI, [Together](https://www.together.ai/), [Groq](https://groq.com/), [Ollama](https://ollama.com/) (`/v1` mode), or [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `--api`.

```python
# Local llama.cpp on port 8080
local = llm_rs.LlmClient(
    api_key="not-used",
    model="local-model",
    base_url="http://localhost:8080",
)

# Groq's OpenAI-compatible gateway
groq = llm_rs.LlmClient(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    provider="openai",
    base_url="https://api.groq.com/openai",
)
```

Provider auto-detection only looks at the model name (`claude*` → Anthropic, else OpenAI), so for non-`claude` models routed through an Anthropic-compatible host, pass `provider="anthropic"` explicitly.

---

## Recipe 9: Cost gate

A defensive wrapper that aborts before sending if a prompt looks suspiciously huge. Crude `len(text) // 4` token estimate, but it catches the "I accidentally pasted my whole codebase" mistake.

```python
import llm_rs

class Gated:
    def __init__(self, client, max_chars):
        self.c, self.max = client, max_chars

    def prompt(self, text, **kw):
        if len(text) > self.max:
            raise ValueError(
                f"refusing prompt of {len(text)} chars (limit {self.max})"
            )
        return self.c.prompt(text, **kw)

gpt = Gated(
    llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini"),
    max_chars=40_000,    # ≈10k tokens
)
```

For hard token budgets, use the CLI's `agent run` with a `[budget]` block — it tracks cumulative usage across chain iterations, which a simple wrapper can't.

---

## Recipe 10: Persona swarm — one client, many voices

Reuse a single `LlmClient` and vary only the `system` argument to spin up a "swarm" of personas. Cheaper than constructing a client per persona because the Tokio runtime + HTTP connection pool are shared.

```python
import os, llm_rs

c = llm_rs.LlmClient(os.environ["ANTHROPIC_API_KEY"], "claude-haiku-4-5")

PERSONAS = {
    "skeptic":  "You are a rigorous skeptic. Find the weakest claim and attack it.",
    "champion": "You are an enthusiastic champion. Make the strongest possible case for the idea.",
    "judge":    "You are a calm judge. Weigh both sides and give a verdict in 2 sentences.",
}

idea = "We should rewrite the build system in Bazel."

skeptic  = c.prompt(idea, system=PERSONAS["skeptic"])
champion = c.prompt(idea, system=PERSONAS["champion"])
verdict  = c.prompt(
    f"SKEPTIC:\n{skeptic}\n\nCHAMPION:\n{champion}",
    system=PERSONAS["judge"],
)

print(verdict)
```

The judge step is just another `prompt()` call with the previous outputs concatenated — no chain loop, no shared state, completely deterministic to debug.

---

## Recipe 11: Tool-using calculator

`@client.tool(...)` is the whole API. The model decides when to call your function; the wrapper executes it and feeds the result back into the chain.

```python
import os, math, llm_rs

client = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini")

@client.tool(description="Compute a single arithmetic expression and return the result")
def calc(expression: str) -> str:
    # Tiny expression evaluator — locked-down namespace.
    return str(eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt}))

print(client.prompt("What's the square root of (8^4 + 1234)? Use the calc tool."))
```

Type hints become a JSON Schema (`str` → `string`, `int` → `integer`, `float` → `number`, `bool` → `boolean`, `list` → `array`, `dict` → `object`). For anything richer — `Optional`, `Union`, dataclasses, Pydantic — pass `schema=` explicitly:

```python
@client.tool(
    description="Look up a city's weather",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["city"],
    },
)
def weather(city, units="c"):
    return f"Sunny and 72°{units.upper()} in {city}"
```

---

## Recipe 12: Multi-turn `Conversation`

The `Conversation` class keeps the message history in memory and reuses your client's tool registry. Same code path as the chain loop, just seeded with the running history.

```python
import os, llm_rs

client = llm_rs.LlmClient(os.environ["ANTHROPIC_API_KEY"], "claude-sonnet-4-6")
client.enable_builtin_tools()

conv = llm_rs.Conversation(client, system="You are a precise note-taker.")
conv.send("My name is Ada Lovelace and I work on the Analytical Engine.")
conv.send("My favourite mathematician is Boole.")
print(conv.send("Summarize what you know about me in one line."))

print(f"\n{len(conv)} messages in history")
for m in conv.messages:
    print(f"  {m['role']}: {m['content'][:60]}")
```

`conv.clear()` resets the history without throwing away the registered tools. `conv.messages` returns plain Python dicts you can pickle, JSON-encode, or stuff into Redis.

---

## Recipe 13: Schema-validated structured output

The same DSL the CLI's `--schema` flag uses is exposed as `llm_rs.parse_schema_dsl`, and the `prompt(...)` method takes either a DSL string or a JSON-Schema-shaped dict via `schema=`. The Rust side wires it through to the provider's structured-output mode (OpenAI's `response_format`, Anthropic's transparent tool wrapping).

```python
import json, os, llm_rs

client = llm_rs.LlmClient(os.environ["OPENAI_API_KEY"], "gpt-4o-mini")

# DSL form — terse, ideal for one-shot extractions.
raw = client.prompt(
    "Hi I'm Marcus, 34, software engineer.",
    schema="name str, age int, profession str",
)
print(json.loads(raw))   # → {'name': 'Marcus', 'age': 34, 'profession': 'software engineer'}

# Dict form — pass any JSON Schema you like.
raw = client.prompt(
    "Three Roman emperors who reigned more than 20 years.",
    schema={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "years": {"type": "integer"},
                    },
                    "required": ["name", "years"],
                },
            }
        },
        "required": ["items"],
    },
)
print(json.loads(raw)["items"])
```

Pass `schema_multi=True` to wrap a single-item DSL in the canonical `{"items": [...]}` envelope without writing the boilerplate yourself.

---

## What's intentionally missing (and what to use instead)

| You want                       | Use this instead                                                       |
|--------------------------------|------------------------------------------------------------------------|
| Chain observability events     | Phase B of the wrapper plan — exposed in the CLI as `-v`/`-vv`.        |
| Persistent conversation logs   | `llm` CLI writes JSONL to `$XDG_DATA_HOME/llm/logs/`. Read it back with `llm logs list`. |
| Retry with backoff             | `llm prompt --retries 3` or set `[retry]` in an agent TOML.            |
| Token budgets                  | `llm agent run` with `[budget] max_tokens = N`.                        |
| External `llm-tool-*` subprocesses | Stays CLI-only — the browser/Python sandbox model doesn't grant `$PATH` access. |

The Python wrapper still doesn't expose retries, budgets, or persistent log writes — those are scoped to Phase B and Phase C. For now, shell out to the CLI (`subprocess.run(["llm", "prompt", ...])`) when you need them; every Phase 1–9 feature is available there, with stable JSON output via `--json` for parsing.
