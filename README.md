# llm-rs

A Unix-philosophy agentic CLI for Large Language Models. Inspired by [simonw/llm](https://github.com/simonw/llm), built for composability --- stdin/stdout pipelines, subprocess-based tool and provider extensibility (`llm-tool-*`, `llm-provider-*`), and multi-target output (native CLI, WASM, Python).

## Usage

```bash
# Send a prompt (streams to stdout)
echo "Hello" | llm

# Positional text works too
llm "Explain monads in one sentence" -m claude-sonnet-4-6

# Specify model and system prompt
llm "What is 2+2?" -m gpt-4o -s "Answer only with the number"

# Use Anthropic models
llm "Hello" -m claude-sonnet-4-6

# Disable streaming
llm "Hello" --no-stream

# Show token usage on stderr
llm "Hello" -u

# Skip logging this prompt
llm "Hello" -n
```

### Tool calling

Built-in tools let the model call functions during a conversation. The CLI manages the chain loop automatically --- it sends tool calls to the executor, feeds results back, and repeats until the model responds with text.

```bash
# Enable a built-in tool
llm "What time is it?" -T llm_time

# Multiple tools
llm "What version are you and what time is it?" -T llm_version -T llm_time

# Limit chain iterations (default: 5 for prompt/chat, 10 for agents)
llm "Do something" -T llm_version --chain-limit 3

# Debug mode: show tool calls/results on stderr
llm "What version?" -T llm_version --tools-debug

# Verbose mode: see chain loop iterations (-v summary, -vv full messages)
llm "What time is it?" -T llm_time --verbose
llm "What time is it?" -T llm_time -vv

# List available built-in tools
llm tools list
```

Available built-in tools:
- `llm_version` --- returns the CLI version
- `llm_time` --- returns current UTC and local time with timezone

### Verbose chain observability

When using tools, the `-v`/`--verbose` flag reveals what happens inside the chain loop --- which iteration you're on, what messages are being sent, per-iteration token usage, and tool call/result details.

```bash
# Level 1 (-v): iteration summary + tool debug
llm "What time is it?" -T llm_time -v
# stderr output:
#   [chain] Iteration 1/5 | 1 message [user]
#   [chain] Iteration 1 complete | usage: 10 input, 5 output | 1 tool call(s)
#   Tool call: llm_time (id: call_1)
#   Arguments: {}
#   Tool result: {"utc_time":"...","local_time":"...","timezone":"..."}
#   [chain] Iteration 2/5 | 3 messages [user, assistant+tools(1), tool(1)]
#   [chain] Iteration 2 complete | usage: 20 input, 10 output | 0 tool call(s)

# Level 2 (-vv): also dumps full message JSON per iteration
llm "What time is it?" -T llm_time -vv
# stderr additionally includes:
#   [chain] Messages:
#   [
#     {"role": "user", "content": "What time is it?"}
#   ]
```

`--verbose` implies `--tools-debug` --- no need for both flags. Works on both `prompt` and `chat` commands.

### External tools

Any executable on `$PATH` named `llm-tool-*` is automatically discovered and usable with `-T`. External tools can be written in any language.

```bash
# List all tools (built-in + external)
llm tools list

# Use an external tool
llm "Make this loud: hello" -T upper -m gpt-4o

# Mix built-in and external tools
llm "What time is it, and shout it" -T llm_time -T shout
```

Writing an external tool requires two things:

1. **Schema**: respond to `--schema` with JSON describing the tool:
   ```bash
   $ llm-tool-upper --schema
   {"name":"upper","description":"Uppercase text","input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}
   ```

2. **Execution**: read arguments JSON from stdin, write result to stdout:
   ```bash
   $ echo '{"text":"hello"}' | llm-tool-upper
   HELLO
   ```

Exit 0 means success (stdout = output). Non-zero means error (stderr = error message). Default timeout: 30 seconds.

### External providers

Any executable on `$PATH` named `llm-provider-*` extends llm-rs with new model providers. External providers can serve models from Ollama, llama.cpp, or any custom backend.

```bash
# Models from external providers appear alongside built-in ones
llm models list

# Use a model from an external provider
llm "Hello" -m llama3

# See all providers and tools
llm plugins list
```

Writing an external provider requires metadata flags and a JSON stdin/stdout protocol:

- `--id` --- print the provider name (e.g. `ollama`)
- `--models` --- print JSON array of model metadata
- `--needs-key` --- print `{"needed":false}` or `{"needed":true,"env_var":"MY_KEY"}`

On invocation, the provider reads a JSON request from stdin and writes either streaming JSONL lines or a single JSON response to stdout. See `doc/implementation.md` for the full protocol specification.

### Conversations

Continue previous conversations, use multi-turn message input, and chat interactively.

```bash
# Continue the most recent conversation
llm -c "And what about 3+3?"

# Continue a specific conversation by ID
llm --cid 01j5a... "Follow up question"

# Load messages from a JSON file
llm --messages conversation.json "What next?"

# Load messages from stdin
echo '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello!"}]' | llm --messages - "Follow up"

# Get JSON output instead of streaming text
llm --json "What is 2+2?"

# Combine: messages input with JSON output
llm --messages history.json --json "Summarize"
```

### Interactive chat

```bash
# Start an interactive chat session
llm chat

# Chat with a specific model and system prompt
llm chat -m claude-sonnet-4-6 -s "You are a helpful assistant"

# Chat with tools enabled
llm chat -T llm_time -T llm_version

# Chat with verbose tool chain output
llm chat -T llm_time -v
```

### Parallel tool execution

When the model requests multiple tool calls in a single turn, llm-rs dispatches them concurrently by default. Results are returned in the same order the model asked for them.

```bash
# Default: parallel dispatch, unlimited concurrency within a single iteration
llm "Check version and time" -T llm_version -T llm_time

# Cap concurrency
llm "Run N tools" -T tool_a -T tool_b --max-parallel-tools 2

# Force sequential dispatch (e.g. to inspect tools one at a time)
llm "Run N tools" -T tool_a -T tool_b --sequential-tools
```

`--tools-approve` forces sequential dispatch automatically so approval prompts don't interleave on stdin. Flags apply to `prompt`, `chat`, and `agent run`. Agents can set `parallel_tools` / `max_parallel_tools` in TOML; CLI flags override.

### Agents

Agents are TOML files that bundle a system prompt, model, tools, chain limit, options, budget, retry, and parallel-tool config. Global agents live in `~/.config/llm/agents/`; project-local agents in `./.llm/agents/` (local shadows global).

```bash
llm agent init researcher              # Scaffold a local agent template
llm agent init planner --global        # Scaffold a global agent
llm agent list                         # List discovered agents (name, model, source)
llm agent show researcher              # Print resolved agent config
llm agent path                         # Print global and local agent directory paths

# Run an agent
llm agent run researcher "summarize recent changes"
echo "some input" | llm agent run researcher

# CLI flags override agent TOML
llm agent run researcher "hi" -m claude-sonnet-4-6 --chain-limit 3 -v

# Dry-run: resolve model, provider, tools, options, budget, retry, and parallel config without calling the LLM
llm agent run researcher "hi" --dry-run
llm agent run researcher "hi" --dry-run --json
llm agent run researcher "hi" --dry-run -vv   # also includes the serialized Prompt payload
```

Example `~/.config/llm/agents/researcher.toml`:

```toml
model = "claude-sonnet-4-6"
system_prompt = "You are a careful research assistant."
tools = ["llm_time", "llm_version"]
chain_limit = 10
parallel_tools = true
max_parallel_tools = 4

[options]
temperature = 0.2

[budget]
max_tokens = 50000

[retry]
max_retries = 3
base_delay_ms = 1000
```

### Budget tracking

Token usage accumulates across chain iterations. Pass `-u` to print cumulative totals; set `[budget] max_tokens` in an agent file to stop the chain when the total exceeds the cap. The chain finishes the current turn, emits a `[budget]` warning, and returns the partial result.

```bash
# Show cumulative usage across all chain iterations
llm "Plan a trip" -T llm_time -u

# llm chat prints a session-wide usage summary on exit
llm chat -u
```

### Retry and backoff

Transient HTTP errors (429, 5xx) are retried with exponential backoff and jitter before any response bytes are streamed. Configure per-invocation with `--retries` or per-agent via `[retry]`.

```bash
llm "Hello" --retries 5
llm chat --retries 3
llm agent run researcher "hi" --retries 5   # overrides agent TOML
```

### Options and aliases

Set persistent per-model options and model-name aliases in `config.toml`. CLI `-o` flags override config defaults per invocation.

```bash
# Options
llm options set gpt-4o temperature 0.7
llm options set gpt-4o max_tokens 1000
llm options get gpt-4o
llm options list
llm options clear gpt-4o temperature

# Aliases
llm aliases set fast gpt-4o-mini
llm aliases set claude claude-sonnet-4-6
llm aliases list
llm aliases show fast
llm aliases remove fast
llm aliases path

llm "Hello" -m claude                       # Uses the alias
llm "Hello" -o temperature 0.9              # Overrides config default
```

### Structured output

Force the model to return JSON conforming to a schema. Works with both OpenAI (native `response_format`) and Anthropic (transparent tool wrapping).

```bash
# Schema DSL: simple field definitions
llm "Extract: John is 30" --schema "name str, age int"

# With field descriptions
llm "Extract: John is 30" --schema "name str:The person's name, age int:Their age"

# JSON Schema literal
llm "Extract name" --schema '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}'

# Schema from a file
llm "Extract data" --schema schema.json

# Multiple items: wrap in array
llm "List the planets" --schema "name str, diameter_km int" --schema-multi

# Preview DSL output
llm schemas dsl "name str, age int"
```

Schema DSL types: `str` (default), `int`, `float`, `bool`.

### Key management

```bash
llm keys set openai          # Prompted for key (hidden input)
llm keys set anthropic       # Set Anthropic API key
llm keys get openai          # Print stored key
llm keys list                # List all stored key names
llm keys path                # Print path to keys.toml
```

Keys are resolved in order: `--key` flag, `keys.toml`, environment variable (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Model management

```bash
llm models list              # List available models (OpenAI + Anthropic)
llm models default           # Show current default model
llm models default gpt-4o    # Set default model
```

Available models:
- **OpenAI:** `gpt-4o`, `gpt-4o-mini`
- **Anthropic:** `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`

### Conversation logs

Every prompt is logged to a JSONL file (one per conversation). Logs are plain text --- inspect them with `cat`, `grep`, `jq`.

```bash
llm logs list                # List recent conversations
llm logs list --json         # JSON output (pipe to jq)
llm logs list -r             # Print the most recent response text
llm logs list -m gpt-4o      # Filter by model
llm logs list -q "rust"      # Full-text search
llm logs list -u             # Show token usage
llm logs path                # Print logs directory path
llm logs status              # Show logging on/off state
llm logs on                  # Enable logging
llm logs off                 # Disable logging
```

Log files live at `~/.local/share/llm/logs/`. Each file is a JSONL conversation:

```jsonl
{"type":"conversation","v":1,"id":"01j5a...","model":"gpt-4o","name":"Hello","created":"2026-04-03T12:00:00Z"}
{"type":"response","id":"01j5b...","model":"gpt-4o","prompt":"Hello","response":"Hi!","usage":{"input":5,"output":3},"duration_ms":230,...}
```

### Schema management

```bash
llm schemas dsl "name str, age int"   # Preview DSL -> JSON Schema
llm schemas list                      # List schemas used in logs
llm schemas show <id>                 # Show schema by ID
```

### Plugins

```bash
llm plugins list    # Show all providers (compiled + external) and external tools
```

Example output:
```
Compiled providers:
  openai (2 models: gpt-4o, gpt-4o-mini)
  anthropic (3 models: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)

External providers:
  ollama (/usr/local/bin/llm-provider-ollama) (3 models: llama3, mistral, phi3)

External tools:
  web_search (/usr/local/bin/llm-tool-web-search) — Search the web
  upper (/usr/local/bin/llm-tool-upper) — Uppercase text
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error (I/O failure, storage error) |
| 2 | Configuration error (missing key, unknown model, bad config) |
| 3 | Provider error (API failure, network timeout) |

## Library usage

In addition to the CLI, llm-rs can be used as a library from JavaScript/TypeScript (via WASM) or Python (via native module). Both support OpenAI and Anthropic.

### WASM (browser / Obsidian plugin)

```typescript
import init, { LlmClient } from '@llm-rs/wasm';

await init();

// Auto-detects provider from model name
const openai = new LlmClient('sk-...', 'gpt-4o');
const claude = new LlmClient('sk-ant-...', 'claude-sonnet-4-6');

// Or use explicit constructors
const client = LlmClient.newAnthropic('sk-ant-...', 'claude-sonnet-4-6');
const custom = LlmClient.newAnthropicWithBaseUrl('sk-ant-...', 'claude-sonnet-4-6', 'https://my-proxy.example.com');

// Non-streaming
const response = await client.prompt('Hello');

// With system prompt
const answer = await client.promptWithSystem('What is 2+2?', 'Answer only with the number');

// Streaming (callback per chunk)
await client.promptStreaming('Tell me a story', (chunk) => {
    process.stdout.write(chunk);
});

// With options
const result = await client.promptWithOptions(
    'Hello',
    null,  // system prompt (optional)
    '{"temperature": 0.7, "max_tokens": 1000}'
);
```

Build from source:

```bash
wasm-pack build crates/llm-wasm --target web       # ES module for browsers
wasm-pack build crates/llm-wasm --target bundler    # For webpack/rollup (Obsidian plugins)
```

The WASM module is stateless --- no config files, no log storage. HTTP goes through the browser's `fetch()` API. The host application manages API keys and persistence.

### Python

```python
import llm_rs

# Auto-detects provider from model name
client = llm_rs.LlmClient("sk-...", "gpt-4o-mini")
claude = llm_rs.LlmClient("sk-ant-...", "claude-sonnet-4-6")

# Or specify provider explicitly
client = llm_rs.LlmClient("sk-ant-...", "claude-sonnet-4-6", provider="anthropic")

# Non-streaming
response = client.prompt("Hello, world!")
print(response)

# With system prompt
answer = client.prompt("What is 2+2?", system="Answer only with the number")

# Streaming (Python iterator)
for chunk in client.prompt_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

Build from source (requires [uv](https://docs.astral.sh/uv/)):

```bash
cd crates/llm-python
uv venv && uv pip install maturin
uv run maturin develop           # Install to current venv
uv run maturin build --release   # Build wheel for distribution
```

Optional parameters: `provider` (`"openai"` or `"anthropic"`), `base_url` for custom API endpoints, `log_dir` to enable JSONL logging.

## Installation

Requires Rust 1.85+ (2024 edition).

```bash
git clone https://github.com/user/llm-rs
cd llm-rs
cargo install --path crates/llm-cli
```

Or build from the workspace:

```bash
cargo build --release -p llm-cli
# Binary is at target/release/llm
```

## Configuration

Config files live in XDG-standard directories:

```
~/.config/llm/config.toml     # Main configuration
~/.config/llm/keys.toml       # API keys (0600 permissions)
~/.local/share/llm/logs/      # Conversation logs (JSONL)
```

Set `LLM_USER_PATH` to put everything in one directory (useful for testing or migrating from Python `llm`).

**config.toml:**

```toml
default_model = "gpt-4o-mini"
logging = true

[aliases]
claude = "claude-sonnet-4-6"
fast = "gpt-4o-mini"
```

**keys.toml:**

```toml
openai = "sk-..."
anthropic = "sk-ant-..."
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (fallback if not in keys.toml) |
| `ANTHROPIC_API_KEY` | Anthropic API key (fallback if not in keys.toml) |
| `OPENAI_BASE_URL` | Override OpenAI API endpoint (for compatible APIs) |
| `ANTHROPIC_BASE_URL` | Override Anthropic API endpoint |
| `LLM_DEFAULT_MODEL` | Override default model |
| `LLM_USER_PATH` | Override config/data directory (flat layout) |

## Architecture

Seven Rust crates in a Cargo workspace:

```
crates/
  llm-core/      Traits, types, streaming, errors, config, keys, schema DSL, chain loop
  llm-openai/    OpenAI Chat API provider (streaming + tools + structured output)
  llm-anthropic/ Anthropic Messages API provider (streaming + tools + structured output)
  llm-store/     JSONL conversation log storage and queries
  llm-cli/       CLI binary (the `llm` command)
  llm-wasm/      WASM library for browser/Obsidian (excluded from workspace)
  llm-python/    Python native module via PyO3 (excluded from workspace)
```

Dependency flow: `llm-cli`, `llm-wasm`, and `llm-python` are top-level entry points -> `llm-openai` + `llm-anthropic` + `llm-store` -> `llm-core`. No cycles.

Key design choices vs the Python original:

- **Subprocess extensibility, not in-process plugins.** Instead of Python's pluggy-based plugin system, external tools (`llm-tool-*`) and providers (`llm-provider-*`) are standalone executables discovered on `$PATH`. Any language can implement the JSON stdin/stdout protocol. Compiled-in providers (OpenAI, Anthropic) are feature-gated for a minimal core binary.
- **JSONL storage.** One file per conversation instead of SQLite. Append-only, human-readable, no migrations.
- **Async-first.** Single `Provider` trait using futures streams, no sync/async class duplication.
- **TOML config.** Two files (`config.toml` + `keys.toml`) instead of six scattered JSON/YAML/text files.
- **Feature-gated providers.** Compile only the providers you need: `--features openai,anthropic` (both default), or `--no-default-features` for a minimal binary.
- **Multi-target.** Core crates compile for both native and `wasm32-unknown-unknown`. The same provider code runs in the CLI, in a browser, and in a Python module.

See [`doc/design/architecture.md`](doc/design/architecture.md) for design rationale, [`doc/roadmap.md`](doc/roadmap.md) for the phased roadmap.

## Testing

```bash
cargo test --workspace    # 530 tests across core workspace crates
```

| Crate | Tests | What's covered |
|-------|------:|----------------|
| `llm-core` | 198 | Types, config, keys, streams, schema DSL, chain loop, ChainEvent, ParallelConfig dispatch, messages, agent config, retry, budget (mock provider) |
| `llm-openai` | 44 | HTTP mocking (wiremock), SSE parsing, tool calls, structured output, multi-turn, HttpError mapping |
| `llm-anthropic` | 50 | HTTP mocking (wiremock), typed SSE, tool_use blocks, transparent schema wrapping, multi-turn, HttpError mapping |
| `llm-store` | 49 | JSONL round-trips, unicode, malformed recovery, listing/queries, message reconstruction |
| `llm-cli` | 189 | Subprocess protocol/discovery/execution, retry wrapper, dry-run rendering (62 unit), CLI integration (127 e2e with assert_cmd) |

Library targets are verified by their build toolchains: `wasm-pack build` for WASM, `maturin develop` for Python.

## Status

Current version: **v0.9**. Phases 1–9 complete. See [`doc/roadmap.md`](doc/roadmap.md) for the full status table and remaining work.

- **v0.1** --- CLI, WASM library, Python module; OpenAI + Anthropic providers end-to-end.
- **v0.2** --- Tool calling, chain loop, built-in tools, structured output, schema DSL.
- **v0.3** --- Multi-turn conversations, `-c`/`--cid`, `llm chat` REPL, full `llm logs`.
- **v0.4** --- Subprocess extensibility (`llm-tool-*`, `llm-provider-*`), `llm plugins`, `-v/--verbose`, `-o/--option`, aliases.
- **v0.5** --- Agent config & discovery (`llm agent run/list/show/init/path`).
- **v0.6** --- Budget tracking with cumulative usage and per-chain enforcement.
- **v0.7** --- Retry/backoff with exponential delay and jitter for transient HTTP errors.
- **v0.8** --- `--dry-run` for `llm agent run` (plain or `--json`).
- **v0.9** --- Parallel tool execution within a chain iteration, order-preserving, opt-out with `--sequential-tools`.

Next up: sub-agent delegation, memory system, Ollama provider, attachments, extract flags. See the Future Work section of the roadmap.

## License

GPLv3
