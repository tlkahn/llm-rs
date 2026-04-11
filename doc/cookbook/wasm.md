# Cookbook: WASM (browser & Obsidian)

`llm-wasm` compiles `llm-core` + the OpenAI / Anthropic providers to a single `.wasm` file you can drop into any browser page, web extension, or Obsidian plugin. There is no Node-only API, no fetch polyfill — the host's `fetch()` is used directly, so streaming "just works" anywhere a modern browser runs.

Phases A and B of the wrapper extension are in: alongside the bare `prompt`/`promptStreaming` calls, you get **tool callbacks** (JS functions exposed to the model), **multi-turn `Conversation`s** with shared message history, **structured output** via `promptWithSchema`, the two built-in tools (`llm_version`, `llm_time`), a **`chain()` wrapper with per-iteration events**, **retries with exponential backoff**, and **token budgets**. Logs are still off the table — see the bottom of this file for what's still CLI-only.

For the underlying type definitions, see [`crates/llm-wasm/pkg/llm_wasm.d.ts`](../../crates/llm-wasm/pkg/llm_wasm.d.ts) after running `wasm-pack build --target web crates/llm-wasm`.

---

## Build

```bash
wasm-pack build --target web crates/llm-wasm
# → crates/llm-wasm/pkg/{llm_wasm.js, llm_wasm_bg.wasm, llm_wasm.d.ts, ...}
```

The `pkg/` directory is a complete ES module. Serve it from any static file host (`python -m http.server`, Vite, Cloudflare Pages, etc.).

> **Keys never leave the browser.** Every example here uses bring-your-own-key — the request goes straight from the user's browser to `api.openai.com` / `api.anthropic.com`. There is no llm-rs server in the loop.

---

## Recipe 1: Hello world page

The smallest thing that works. Save as `index.html` next to the `pkg/` directory.

```html
<!doctype html>
<html>
  <body>
    <pre id="out">loading…</pre>
    <script type="module">
      import init, { LlmClient } from "./pkg/llm_wasm.js";

      await init();
      const client = new LlmClient(prompt("OpenAI key:"), "gpt-4o-mini");
      document.getElementById("out").textContent =
        await client.prompt("Write a haiku about WebAssembly.");
    </script>
  </body>
</html>
```

`new LlmClient(key, model)` auto-detects Anthropic when the model starts with `claude`, otherwise OpenAI. Use `LlmClient.newAnthropic(key, model)` or `LlmClient.newWithBaseUrl(...)` to be explicit.

---

## Recipe 2: Streaming chat widget (vanilla JS)

`promptStreaming` invokes a JS callback for each text delta as it arrives over SSE. No frameworks needed.

```html
<input id="q" placeholder="Ask anything" style="width: 80%">
<button id="go">Send</button>
<pre id="out"></pre>

<script type="module">
  import init, { LlmClient } from "./pkg/llm_wasm.js";
  await init();

  const key = localStorage.getItem("openai-key") ?? prompt("OpenAI key:");
  localStorage.setItem("openai-key", key);
  const client = new LlmClient(key, "gpt-4o-mini");

  document.getElementById("go").onclick = async () => {
    const out = document.getElementById("out");
    out.textContent = "";
    await client.promptStreaming(
      document.getElementById("q").value,
      (chunk) => { out.textContent += chunk; },
    );
  };
</script>
```

The callback runs on the main thread between event-loop ticks, so DOM updates are smooth without `requestAnimationFrame`.

---

## Recipe 3: Two providers, side by side

Spin up two clients and race them. Useful for "which model gives the better answer to *my* prompt" demos.

```js
import init, { LlmClient } from "./pkg/llm_wasm.js";
await init();

const openai    = new LlmClient(OPENAI_KEY, "gpt-4o-mini");
const anthropic = LlmClient.newAnthropic(ANTHROPIC_KEY, "claude-haiku-4-5");

const question = "Explain monads in one sentence.";

const [a, b] = await Promise.all([
  openai.prompt(question),
  anthropic.prompt(question),
]);

console.log("OpenAI    :", a);
console.log("Anthropic :", b);
```

Because each client owns its own provider, you can mix and match freely without rebuilding the WASM.

---

## Recipe 4: Persona playground with system prompts

`promptWithSystem` lets you steer the model without prepending the system text to every user message yourself.

```js
const tutor = new LlmClient(KEY, "gpt-4o-mini");

const persona =
  "You are a pedantic Latin teacher. Always reply in two parts: " +
  "(1) the corrected sentence in classical Latin, (2) a one-line gloss.";

const reply = await tutor.promptWithSystem(
  "How do I say 'I came, I saw, I conquered'?",
  persona,
);
```

The system prompt is sent as the provider's first-class system field (top-level for Anthropic, role=system message for OpenAI), not folded into the user turn — so safety/instruction-following heuristics still apply.

---

## Recipe 5: JSON-mode structured output

`promptWithOptions` takes a JSON string of provider options. Pair it with OpenAI's `response_format` to get parseable JSON straight out of the browser.

```js
const client = new LlmClient(KEY, "gpt-4o-mini");

const opts = JSON.stringify({
  temperature: 0,
  response_format: { type: "json_object" },
});

const raw = await client.promptWithOptions(
  "Extract name and age. Reply as JSON with keys 'name' and 'age'.\n\n" +
    "Hi I'm Marcus, 34, software engineer.",
  null,                       // no system prompt
  opts,
);

const { name, age } = JSON.parse(raw);
```

For full JSON-Schema validation, run the CLI instead — `--schema` is not exposed through the WASM surface today.

---

## Recipe 6: Streaming token-cost meter

Combine `promptStreamingWithOptions` with a running character count to show users the response growing live, plus a rough cost estimate. (Cheap, character-based; for billable usage, the CLI's `-u` flag is the source of truth.)

```js
const client = new LlmClient(KEY, "gpt-4o-mini");

let chars = 0;
const meter = document.getElementById("meter");
const out   = document.getElementById("out");

await client.promptStreamingWithOptions(
  "Summarize the plot of Moby-Dick in 5 bullet points.",
  "You write tight, punchy summaries.",
  JSON.stringify({ temperature: 0.4, max_tokens: 400 }),
  (chunk) => {
    out.textContent += chunk;
    chars += chunk.length;
    meter.textContent = `${chars} chars  ~$${(chars * 0.0000005).toFixed(4)}`;
  },
);
```

---

## Recipe 7: Translate-on-select browser bookmarklet

Drop this into a bookmark; clicking it translates the current text selection to the language you pick. No backend, no extension, no install.

```js
javascript:(async () => {
  const mod = await import("https://YOUR-HOST/pkg/llm_wasm.js");
  await mod.default();
  const key  = localStorage.getItem("openai-key")
            ?? localStorage.setItem("openai-key", prompt("OpenAI key:"));
  const lang = prompt("Translate to:", "Japanese");
  const text = window.getSelection().toString();
  const c    = new mod.LlmClient(key, "gpt-4o-mini");
  const out  = await c.promptWithSystem(
    text,
    `Translate the user's text to ${lang}. Reply with only the translation.`,
  );
  alert(out);
})();
```

Host `pkg/` on any CDN or your own static site. Because the request goes browser → OpenAI directly, you do not need CORS proxying.

---

## Recipe 8: Obsidian plugin — "Explain selection"

Obsidian plugins run in an Electron renderer, so the WASM module loads exactly like in a browser. This is the minimum viable plugin.

```ts
// main.ts
import { Plugin, MarkdownView, Notice } from "obsidian";
import init, { LlmClient } from "./pkg/llm_wasm.js";

let client: LlmClient | null = null;

export default class LlmExplain extends Plugin {
  async onload() {
    await init();
    const key = (this.app as any).loadLocalStorage("openai-key")
             ?? prompt("OpenAI key:");
    client = new LlmClient(key, "gpt-4o-mini");

    this.addCommand({
      id: "explain-selection",
      name: "Explain selection (LLM)",
      editorCallback: async (editor) => {
        const sel = editor.getSelection();
        if (!sel) { new Notice("Select some text first."); return; }
        new Notice("Thinking…");
        const out = await client!.promptWithSystem(
          sel,
          "Explain the selected text to a curious generalist in 3 sentences.",
        );
        editor.replaceSelection(`${sel}\n\n> ${out.replace(/\n/g, "\n> ")}\n`);
      },
    });
  }
}
```

Copy `crates/llm-wasm/pkg/` into the plugin directory next to `main.ts` and Obsidian's bundler will pick it up. The same pattern works for VS Code extensions running in the renderer.

---

## Recipe 9: Cancellable streaming with `AbortController`

The `LlmClient` doesn't expose an abort API directly, but because each call returns a `Promise`, you can wrap the streaming callback in an `AbortController` check and let the underlying `fetch` cancel itself when the page navigates away.

```js
const ctrl = new AbortController();
document.getElementById("stop").onclick = () => ctrl.abort();

try {
  await client.promptStreaming(question, (chunk) => {
    if (ctrl.signal.aborted) throw new Error("user-cancelled");
    out.textContent += chunk;
  });
} catch (e) {
  if (e.message !== "user-cancelled") throw e;
}
```

Throwing from the callback unwinds back through the WASM layer and rejects the outer `Promise` — the underlying SSE `fetch` is then GC'd along with its reader.

---

## Recipe 10: Local-first model picker UI

A tiny pattern for "let the user pick a model and remember it." Combines `localStorage`, `LlmClient.newWithBaseUrl` (for self-hosted OpenAI-compatible endpoints), and lazy reconstruction.

```js
const MODELS = [
  { id: "gpt-4o-mini",       provider: "openai",    label: "OpenAI · 4o-mini (cheap)"  },
  { id: "gpt-4o",            provider: "openai",    label: "OpenAI · 4o"               },
  { id: "claude-haiku-4-5",  provider: "anthropic", label: "Anthropic · Haiku 4.5"     },
  { id: "claude-sonnet-4-6", provider: "anthropic", label: "Anthropic · Sonnet 4.6"    },
];

function makeClient({ id, provider }) {
  const key = localStorage.getItem(`${provider}-key`)
           ?? prompt(`${provider} key:`);
  localStorage.setItem(`${provider}-key`, key);
  return provider === "anthropic"
    ? LlmClient.newAnthropic(key, id)
    : new LlmClient(key, id);
}

let current = MODELS.find(m => m.id === localStorage.getItem("model")) ?? MODELS[0];
let client  = makeClient(current);

picker.onchange = (e) => {
  current = MODELS.find(m => m.id === e.target.value);
  localStorage.setItem("model", current.id);
  client = makeClient(current);
};
```

Use `LlmClient.newWithBaseUrl(key, model, "https://my-proxy.example.com")` to point at any OpenAI-compatible gateway (Azure, Together, Groq, llama.cpp's `--api`).

---

## Recipe 11: Tool callbacks from JavaScript

`registerTool` accepts a plain object with a `name`, `description`, `inputSchema` (any JSON Schema), and an `execute` function. The function may be sync *or* `async` — both are supported. The wrapper drives the chain loop on the Rust side and feeds your function's return value back into the conversation.

```js
import init, { LlmClient } from "./pkg/llm_wasm.js";
await init();

const client = new LlmClient(KEY, "gpt-4o-mini");

client.registerTool({
  name: "get_weather",
  description: "Fetch the current weather for a city",
  inputSchema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: async ({ city }) => {
    const r = await fetch(`https://wttr.in/${encodeURIComponent(city)}?format=3`);
    return await r.text();
  },
});

client.enableBuiltinTools();          // llm_version, llm_time
console.log(await client.prompt("What's the weather in Reykjavík?"));
```

`setChainLimit(n)` raises the iteration cap if your tools cascade. Default is 5.

The chain loop happens entirely on the Rust side — no JS-level async/await spaghetti to manage tool turns. Each call out to your `execute` function is awaited via `JsFuture`, the result is serialized back into a tool message, and the next iteration runs automatically until the model emits a final answer (or the chain limit hits).

---

## Recipe 12: `Conversation` for multi-turn chat

`client.conversation()` (or `new Conversation(client, system?)`) hands back a stateful object that holds the running message history and shares the client's tool registry. Same chain loop under the hood, just seeded with the conversation so far.

```html
<input id="q" placeholder="Ask anything">
<button id="go">Send</button>
<button id="reset">Clear</button>
<pre id="out"></pre>

<script type="module">
  import init, { LlmClient } from "./pkg/llm_wasm.js";
  await init();

  const client = new LlmClient(localStorage.getItem("openai-key"), "gpt-4o-mini");
  const conv   = client.conversation("You are a helpful assistant.");

  document.getElementById("go").onclick = async () => {
    const q = document.getElementById("q").value;
    const out = document.getElementById("out");
    out.textContent += `> ${q}\n`;
    const reply = await conv.sendStreaming(q, (chunk) => {
      out.textContent += chunk;
    });
    out.textContent += "\n\n";
  };

  document.getElementById("reset").onclick = () => conv.clear();
</script>
```

`conv.messages` (getter) returns a JSON-serializable array of `{role, content, ...}` you can stash in IndexedDB or `localStorage` between page loads. `conv.length` is the message count.

---

## Recipe 13: Structured output with `promptWithSchema`

`promptWithSchema` accepts either a JSON-Schema object directly or LLM-RS's terse schema DSL string. The Rust side wires it through to the provider's structured-output mode (`response_format` for OpenAI, transparent tool wrapping for Anthropic).

```js
import init, { LlmClient, parseSchemaDsl } from "./pkg/llm_wasm.js";
await init();

const client = new LlmClient(KEY, "gpt-4o-mini");

// DSL form — terse, perfect for one-shot extraction.
const raw = await client.promptWithSchema(
  "Hi I'm Marcus, 34, software engineer.",
  null,                                // no system prompt
  "name str, age int, profession str", // DSL string
  false,                               // multi=false
);
const { name, age, profession } = JSON.parse(raw);

// Or pass a JSON-Schema object directly.
const schema = parseSchemaDsl("title str, year int");
const films = await client.promptWithSchema(
  "Three Akira Kurosawa films.",
  null,
  schema,
  true,                                // multi=true → wraps in {items:[…]}
);
console.log(JSON.parse(films).items);
```

`parseSchemaDsl(...)` is also exposed as a free function so you can build schemas at startup time and reuse them across requests.

---

## Recipe: Chain loop with events, budget, and retry

`client.chain(text, options)` runs the full chain loop and hands back a structured result. Pass `onEvent` to observe each iteration:

```js
import init, { LlmClient } from "./pkg/llm_wasm.js";
await init();

const client = new LlmClient(openaiKey, "gpt-4o-mini");
client.registerTool({
  name: "add",
  description: "Add two numbers",
  inputSchema: {
    type: "object",
    properties: { a: { type: "number" }, b: { type: "number" } },
    required: ["a", "b"],
  },
  execute: ({ a, b }) => a + b,
});

const result = await client.chain(
  "What is (17 + 25) + (100 + 1)? Use the add tool twice.",
  {
    chainLimit: 10,
    budget: 2000,                      // cumulative tokens cap
    onEvent: (evt) => {
      if (evt.type === "iteration_start") {
        console.log(`→ iteration ${evt.iteration}/${evt.limit}`);
      } else if (evt.type === "iteration_end") {
        const calls = evt.tool_calls.map((c) => c.name).join(", ") || "none";
        console.log(`← tool_calls=[${calls}]`);
      } else if (evt.type === "budget_exhausted") {
        console.warn("budget exhausted", evt.cumulative_usage);
      }
    },
  },
);

console.log("answer:", result.text);
console.log("usage: ", result.totalUsage);
console.log("stopped on budget?", result.budgetExhausted);
```

`result` is a plain JS object: `{ text, toolCalls, totalUsage, budgetExhausted }`. Events are type-tagged dicts: `iteration_start`, `iteration_end`, `budget_exhausted`.

Streaming variant — `chainStreaming(text, callback, options)` fires the same callback for each text chunk (`{type: "text", content}`) *and* each event, interleaved:

```js
await client.chainStreaming(
  "Tell me a story.",
  (evt) => {
    if (evt.type === "text") process.stdout.write(evt.content);
    else console.log("\n[" + evt.type + "]");
  },
  { chainLimit: 5 },
);
```

### Retry with exponential backoff

Transient HTTP errors (429, 5xx) are retryable. Call `setRetryConfig` on the client and every subsequent provider call — `prompt`, `promptStreaming`, `chain`, `conversation.send` — wraps itself in an exponential-backoff retry loop:

```js
client.setRetryConfig(
  3,        // max_retries
  1000,     // base_delay_ms → 1s, 2s, 4s schedule
  30_000,   // max_delay_ms  → clamp
  true,     // jitter
);
```

Non-retryable errors (4xx other than 429, malformed responses) surface immediately without retrying. Call with `max_retries = 0` to disable. The default is `max_retries = 0` — opt in when you need it.

---

## What's intentionally missing

| Feature                          | Why                                                                            |
|----------------------------------|--------------------------------------------------------------------------------|
| External `llm-tool-*` subprocesses | Tools-by-`$PATH` is a CLI-only concept. JS callbacks are the substitute.       |
| Logs / persistent conversation store | No filesystem in the browser. A JS-callback `ConversationStore` backend (IndexedDB-friendly) is scoped to Phase C. |
| Programmatic agent runs          | `AgentConfig` + `runAgent` is scoped to Phase C.                               |

If your use case starts to need any of these, you have two good options:

1. **Run the CLI behind a tiny HTTP shim** and call it from the browser via `fetch`.
2. **Embed `llm-core` + a provider crate directly** in a WASM target of your own — every Phase 1–9 feature compiles to `wasm32-unknown-unknown` already; `llm-wasm` is just the thinnest possible wrapper.
