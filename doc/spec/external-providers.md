# External Provider Protocol Specification

LLM-RS discovers and runs external LLM providers as subprocess executables. Any program on `$PATH` whose name starts with `llm-provider-` becomes available as a model provider, adding new models without recompilation.

This document specifies the protocol. For the external tool protocol, see [external-tools.md](external-tools.md).

---

## Overview

An external provider is a standalone executable that implements three operations:

1. **Identity** (`--id` flag) --- return the provider's unique identifier.
2. **Model enumeration** (`--models` flag) --- list available models with capabilities.
3. **Key requirements** (`--needs-key` flag) --- declare whether an API key is needed.
4. **Execution** (stdin/stdout) --- receive a request as JSON on stdin, return response chunks on stdout.

The executable can be written in any language. The only requirement is that it lives on `$PATH`, has executable permissions, and follows the naming convention `llm-provider-<suffix>`.

### Lifecycle

```
Discovery                        Invocation (per prompt)
---------                        ----------------------
llm starts                       User runs llm -m <model>
    |                                 |
    v                                 v
scan $PATH for                   spawn llm-provider-*
llm-provider-* binaries              |
    |                                 v
    v                            write ProviderRequest JSON to stdin
run each with --id,                   |
--models, --needs-key                 v
    |                            read stdout:
    v                              streaming: JSONL ProtocolChunk lines
register provider                  non-streaming: single ProviderResponse JSON
with its models                       |
    |                                 v
    v                            check exit code
available for                    return response to user
model selection
```

Discovery happens once when `llm` starts. Each prompt spawns a fresh process.

---

## Naming Convention

The binary name must match `llm-provider-*`. The suffix is arbitrary --- the provider's logical ID comes from the `--id` flag, not the binary name.

```
Binary name              Provider ID (from --id)    Models available via
-----------              -----------------------    -------------------
llm-provider-mistral     mistral                    llm -m mistral-large
llm-provider-my-local    local-gpu                  llm -m llama-70b
```

### Discovery rules

- All directories in `$PATH` are scanned for files matching the `llm-provider-` prefix.
- Directories named `llm-provider-*` are skipped (only regular files and symlinks).
- On Unix, files without any execute bit are skipped.
- If the same filename appears in multiple `$PATH` directories, the first occurrence wins.

---

## Metadata Flags

### `--id`

Returns the provider's unique identifier as a plain string on stdout.

```bash
$ llm-provider-mistral --id
mistral
```

### `--models`

Returns a JSON array of `ModelInfo` objects on stdout.

```bash
$ llm-provider-mistral --models
```

```json
[
  {"id": "mistral-large", "can_stream": true, "supports_tools": true, "supports_schema": false},
  {"id": "mistral-small", "can_stream": true, "supports_tools": false, "supports_schema": false}
]
```

Each object has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Model identifier, used with `-m` on the command line |
| `can_stream` | bool | no | Whether the model supports streaming (default: true) |
| `supports_tools` | bool | no | Whether the model supports tool calling (default: false) |
| `supports_schema` | bool | no | Whether the model supports structured output (default: false) |

### `--needs-key`

Returns a JSON object indicating whether the provider requires an API key.

```bash
$ llm-provider-mistral --needs-key
{"needed": true, "env_var": "MISTRAL_API_KEY"}
```

| Field | Type | Description |
|-------|------|-------------|
| `needed` | bool | Whether an API key is required |
| `env_var` | string or null | Environment variable name to check for the key |

If `needed` is false, the key resolution chain is skipped for this provider.

---

## Execution

### Request format

When a user prompts a model from this provider, llm-rs spawns the provider binary and writes a `ProviderRequest` JSON object to stdin:

```bash
echo '<request json>' | llm-provider-mistral --model mistral-large --stream
```

Command-line arguments:
- `--model <id>` --- which model to use (from the `--models` list)
- `--stream` --- request streaming output (omitted for non-streaming)

Stdin JSON:

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "tools": [{"name": "calc", "description": "...", "input_schema": {"type": "object"}}],
  "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
  "options": {"temperature": 0.7}
}
```

| Field | Type | Present when |
|-------|------|-------------|
| `messages` | array | Always. Conversation history as role/content pairs. |
| `tools` | array | When `-T` tools are enabled. Tool definitions for the LLM. |
| `schema` | object | When `--schema` structured output is requested. |
| `options` | object | When `-o` options are set. Provider-specific key-value pairs. |

If an API key was resolved (via `keys.toml`, env var, or `--key`), it is passed via the `LLM_PROVIDER_KEY` environment variable --- not in the JSON.

### Streaming response (with `--stream`)

Stdout is JSONL --- one `ProtocolChunk` JSON object per line, streamed as chunks arrive:

```jsonl
{"type":"text","content":"The answer"}
{"type":"text","content":" is 4."}
{"type":"tool_call","name":"calc","arguments":{"expr":"2+2"},"id":"tc_1"}
{"type":"usage","input":15,"output":8}
{"type":"done"}
```

Chunk types:

| Type | Fields | Description |
|------|--------|-------------|
| `text` | `content` | A text chunk of the response |
| `tool_call` | `name`, `arguments`, `id` | A complete tool call (arguments fully assembled) |
| `usage` | `input`, `output` | Token usage counts |
| `done` | (none) | Signals end of response |

The `done` chunk must be the last line. llm-rs reads until it sees `done` or the process exits.

### Non-streaming response (without `--stream`)

Stdout is a single `ProviderResponse` JSON object:

```json
{
  "type": "response",
  "content": "The answer is 4.",
  "tool_calls": [],
  "usage": {"input": 15, "output": 8}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | The full response text |
| `tool_calls` | array | Tool calls (empty if none). Each: `{name, arguments, id}` |
| `usage` | object or null | Token counts `{input, output}` |

### Errors

Exit code non-zero with a human-readable error message on stderr. llm-rs captures stderr and reports it as a provider error.

```
$ echo '...' | llm-provider-mistral --model bad-model --stream
(stderr) unknown model: bad-model
(exit 1)
```

---

## Complete Example: Shell Script Provider

A minimal provider wrapping a hypothetical API:

```bash
#!/bin/sh
# File: llm-provider-echo
# A toy provider that echoes the last user message (for testing)

case "$1" in
    --id)
        echo "echo"
        exit 0
        ;;
    --models)
        echo '[{"id":"echo-1","can_stream":true,"supports_tools":false,"supports_schema":false}]'
        exit 0
        ;;
    --needs-key)
        echo '{"needed":false,"env_var":null}'
        exit 0
        ;;
esac

# Parse --model and --stream from args
STREAM=false
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --stream) STREAM=true; shift ;;
        *) shift ;;
    esac
done

# Read request JSON from stdin
REQUEST=$(cat)
LAST_MSG=$(echo "$REQUEST" | python3 -c "
import sys, json
msgs = json.load(sys.stdin)['messages']
user_msgs = [m for m in msgs if m['role'] == 'user']
print(user_msgs[-1]['content'] if user_msgs else '(no message)')
")

if [ "$STREAM" = "true" ]; then
    echo "{\"type\":\"text\",\"content\":\"Echo: $LAST_MSG\"}"
    echo '{"type":"usage","input":1,"output":1}'
    echo '{"type":"done"}'
else
    echo "{\"type\":\"response\",\"content\":\"Echo: $LAST_MSG\",\"tool_calls\":[],\"usage\":{\"input\":1,\"output\":1}}"
fi
```

Usage:

```bash
chmod +x llm-provider-echo
cp llm-provider-echo ~/.local/bin/

llm -m echo-1 "Hello world"
# Echo: Hello world

llm plugins list
# ...
# External providers:
#   echo (1 model: echo-1)
```

---

## Viewing in plugins list

```bash
$ llm plugins list
Compiled providers:
  openai (2 models: gpt-4o, gpt-4o-mini)
  anthropic (3 models: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)

External providers:
  mistral (2 models: mistral-large, mistral-small) — /usr/local/bin/llm-provider-mistral

External tools:
  upper (/usr/local/bin/llm-tool-upper) — Convert text to uppercase
```
