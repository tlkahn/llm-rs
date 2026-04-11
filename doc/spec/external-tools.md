# External Tool Protocol Specification

LLM-RS discovers and runs external tools as subprocess executables. Any program on `$PATH` whose name starts with `llm-tool-` becomes available as a tool that LLMs can call during a conversation.

This document specifies the protocol and provides runnable examples. For worked examples of *specialist tools* — external tools that internally call `llm` with a narrow purpose-specific agent — see [doc/cookbook/specialist-tools.md](../cookbook/specialist-tools.md) and the design note [doc/research/specialist-tools-vs-sub-agents.md](../research/specialist-tools-vs-sub-agents.md).

---

## Overview

An external tool is a standalone executable that implements two operations:

1. **Schema declaration** (`--schema` flag) --- tell llm-rs what the tool does and what arguments it accepts.
2. **Execution** (stdin/stdout) --- receive arguments as JSON on stdin, return the result on stdout.

The executable can be written in any language: shell script, Python, Go, Rust, a compiled binary, etc. The only requirement is that it lives on `$PATH`, has executable permissions, and follows the naming convention `llm-tool-<suffix>`.

### Lifecycle

```
Discovery                Invocation (per tool call)
---------                --------------------------
llm starts               LLM requests tool call
    |                         |
    v                         v
scan $PATH for           spawn llm-tool-*
llm-tool-* binaries          |
    |                         v
    v                    write arguments JSON to stdin
run each with --schema       |
    |                         v
    v                    read stdout (result)
parse Tool JSON          read stderr (on error)
    |                    check exit code
    v                         |
register tools                v
for this session         return ToolResult to LLM
```

Discovery happens once when `llm` starts (or when `llm tools list` / `llm plugins list` runs). Each tool call during a conversation spawns a fresh process.

---

## Naming convention

The binary name must match `llm-tool-*`. The suffix after `llm-tool-` is arbitrary and does not need to match the tool's logical name --- the logical name comes from the `name` field in the schema JSON.

```
Binary name          Tool name (from schema)    Used as
-----------          -----------------------    -------
llm-tool-upper       upper                      llm -T upper
llm-tool-web-search  web_search                 llm -T web_search
llm-tool-my-calc     calculator                 llm -T calculator
```

The binary name is for PATH discovery. The tool name in the schema is what the LLM sees and what you pass to `-T`.

### Discovery rules

- All directories in `$PATH` are scanned for files matching the `llm-tool-` prefix.
- Directories named `llm-tool-*` are skipped (only regular files and symlinks).
- On Unix, files without any execute bit (`chmod` mode `& 0o111 == 0`) are skipped.
- If the same filename appears in multiple `$PATH` directories, the first occurrence wins (standard Unix behavior, matching `which`).

---

## Schema declaration

When invoked with the `--schema` flag, the tool must print a single JSON object to stdout and exit 0.

### Schema format

```json
{
  "name": "<tool_name>",
  "description": "<human-readable description>",
  "input_schema": {
    "type": "object",
    "properties": {
      "<param_name>": { "type": "<json_schema_type>", ... },
      ...
    },
    "required": ["<param_name>", ...]
  }
}
```

The three fields are:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Logical tool name. The LLM sees this name and uses it to request the tool. Used with `-T` on the command line. |
| `description` | string | yes | Human-readable description shown to the LLM and in `llm tools list`. Should clearly explain what the tool does so the LLM knows when to use it. |
| `input_schema` | object | yes | JSON Schema describing the arguments. Must have `"type": "object"` at the top level. |

The `input_schema` follows standard [JSON Schema](https://json-schema.org/) syntax. The LLM provider (OpenAI, Anthropic) uses it to generate valid arguments.

### Schema example

```bash
$ llm-tool-weather --schema
{"name":"weather","description":"Get current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string","description":"City name"},"units":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature units"}},"required":["city"]}}
```

Pretty-printed:

```json
{
  "name": "weather",
  "description": "Get current weather for a city",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature units"
      }
    },
    "required": ["city"]
  }
}
```

### Schema errors

If `--schema` exits with a non-zero status, or if stdout is not valid JSON matching the expected shape, the tool is skipped during discovery. A warning is printed to stderr:

```
warning: skipping tool /usr/local/bin/llm-tool-broken: invalid schema JSON from /usr/local/bin/llm-tool-broken: ...
```

If `--schema` does not complete within the timeout (default 30 seconds), the tool is also skipped.

---

## Execution

When the LLM calls a tool, llm-rs spawns the tool's binary as a subprocess:

1. The tool's **arguments** (a JSON object matching `input_schema`) are written to **stdin**.
2. The tool writes its **result** (plain text) to **stdout**.
3. The tool's **exit code** determines success or failure.

### What the tool receives on stdin

The tool receives only the arguments JSON object --- not a wrapper, not the full tool call envelope. The arguments are exactly what the LLM generated based on `input_schema`.

For a tool with schema `{"properties":{"text":{"type":"string"}},"required":["text"]}`, the stdin will be:

```json
{"text":"hello world"}
```

If the schema defines no required properties and the LLM passes no arguments, stdin will be:

```json
{}
```

Stdin is closed (EOF) after the JSON is written. Use `read` in shell or `sys.stdin.read()` in Python.

### Success (exit 0)

The tool's stdout is captured as the tool result and sent back to the LLM. The output is treated as plain text (not parsed as JSON by llm-rs). The LLM receives it verbatim.

```
stdin:  {"text":"hello"}
stdout: HELLO
exit:   0
```

An empty stdout with exit 0 is valid --- it means the tool succeeded but produced no output.

### Failure (exit non-zero)

The tool's stderr is captured as the error message and sent back to the LLM (so it can retry or explain the failure). Stdout is ignored.

```
stdin:  {"city":"Atlantis"}
stderr: city not found: Atlantis
exit:   1
```

If stderr is empty on a non-zero exit, the error message defaults to `"tool exited with exit status: <code>"`.

### Timeout

If the tool does not exit within 30 seconds, it is killed and the LLM receives a timeout error: `"tool <name> timed out"`.

### Summary table

| Exit code | stdout | stderr | Result |
|-----------|--------|--------|--------|
| 0 | captured as output | ignored | success |
| 0 | empty | ignored | success (empty output) |
| non-zero | ignored | captured as error | error |
| non-zero | ignored | empty | error with generic message |
| (timeout) | --- | --- | error: "tool \<name\> timed out" |

---

## Complete examples

### Example 1: Shell script (uppercase)

A minimal tool that uppercases text. No dependencies beyond `tr`.

```bash
#!/bin/sh
# File: llm-tool-upper
# Install: chmod +x llm-tool-upper && cp llm-tool-upper /usr/local/bin/

if [ "$1" = "--schema" ]; then
    cat <<'EOF'
{"name":"upper","description":"Convert text to uppercase","input_schema":{"type":"object","properties":{"text":{"type":"string","description":"Text to uppercase"}},"required":["text"]}}
EOF
    exit 0
fi

# Read arguments JSON from stdin, extract "text", uppercase it
read input
echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'].upper())"
```

Usage:

```bash
# Install
chmod +x llm-tool-upper
cp llm-tool-upper ~/.local/bin/   # or anywhere on $PATH

# Verify discovery
llm tools list
# llm_version: Returns the current LLM CLI version
# llm_time: Returns the current date and time
# upper: Convert text to uppercase (~/.local/bin/llm-tool-upper)

# Test manually
echo '{"text":"hello world"}' | llm-tool-upper
# HELLO WORLD

# Use with LLM
llm "Convert 'good morning' to uppercase" -T upper
```

### Example 2: Shell script with no dependencies (word count)

Using only POSIX shell builtins and standard utilities:

```bash
#!/bin/sh
# File: llm-tool-wc

if [ "$1" = "--schema" ]; then
    echo '{"name":"word_count","description":"Count words, lines, and characters in text","input_schema":{"type":"object","properties":{"text":{"type":"string","description":"Text to analyze"}},"required":["text"]}}'
    exit 0
fi

# Extract text field (requires python3 or jq for JSON parsing)
text=$(cat | python3 -c "import sys,json; print(json.load(sys.stdin)['text'], end='')")

words=$(echo "$text" | wc -w | tr -d ' ')
lines=$(echo "$text" | wc -l | tr -d ' ')
chars=$(echo "$text" | wc -c | tr -d ' ')

echo "Words: $words, Lines: $lines, Characters: $chars"
```

### Example 3: Python script (web search)

A more realistic tool that calls an external API:

```python
#!/usr/bin/env python3
"""File: llm-tool-search"""

import json
import sys
import urllib.request

SCHEMA = {
    "name": "web_search",
    "description": "Search the web and return top results",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 3)"
            }
        },
        "required": ["query"]
    }
}

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--schema":
        json.dump(SCHEMA, sys.stdout)
        sys.exit(0)

    args = json.load(sys.stdin)
    query = args["query"]
    num = args.get("num_results", 3)

    # Replace with your actual search API
    try:
        url = f"https://api.example.com/search?q={query}&n={num}"
        # ... perform search ...
        print(f"Results for '{query}': (search API call would go here)")
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Example 4: Python script with error handling

A tool that demonstrates the error path:

```python
#!/usr/bin/env python3
"""File: llm-tool-divide"""

import json
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--schema":
    json.dump({
        "name": "divide",
        "description": "Divide two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "numerator": {"type": "number"},
                "denominator": {"type": "number"}
            },
            "required": ["numerator", "denominator"]
        }
    }, sys.stdout)
    sys.exit(0)

args = json.load(sys.stdin)
a = args["numerator"]
b = args["denominator"]

if b == 0:
    print("division by zero", file=sys.stderr)
    sys.exit(1)

print(a / b)
```

When the LLM calls `divide(numerator=10, denominator=0)`, it receives the error `"division by zero"` and can explain or retry.

### Example 5: Compiled Go binary

```go
// File: main.go
// Build: go build -o llm-tool-uuid . && cp llm-tool-uuid /usr/local/bin/
package main

import (
    "encoding/json"
    "fmt"
    "os"

    "github.com/google/uuid"
)

type Schema struct {
    Name        string      `json:"name"`
    Description string      `json:"description"`
    InputSchema interface{} `json:"input_schema"`
}

func main() {
    if len(os.Args) > 1 && os.Args[1] == "--schema" {
        schema := Schema{
            Name:        "uuid",
            Description: "Generate a random UUID",
            InputSchema: map[string]interface{}{
                "type":       "object",
                "properties": map[string]interface{}{},
            },
        }
        json.NewEncoder(os.Stdout).Encode(schema)
        return
    }

    // No arguments needed --- just generate a UUID
    fmt.Println(uuid.New().String())
}
```

### Example 6: Tool with no arguments

Tools do not need to accept arguments. The schema can declare an empty properties object:

```bash
#!/bin/sh
# File: llm-tool-flip-coin

if [ "$1" = "--schema" ]; then
    echo '{"name":"flip_coin","description":"Flip a coin, returns heads or tails","input_schema":{"type":"object","properties":{}}}'
    exit 0
fi

if [ $((RANDOM % 2)) -eq 0 ]; then
    echo "heads"
else
    echo "tails"
fi
```

The LLM will call this with `{}` on stdin. The tool ignores stdin and returns a result.

---

## CLI usage

### Listing tools

```bash
$ llm tools list
llm_version: Returns the current LLM CLI version
llm_time: Returns the current date and time
upper: Convert text to uppercase (/usr/local/bin/llm-tool-upper)
word_count: Count words, lines, and characters in text (/usr/local/bin/llm-tool-wc)
```

Built-in tools have no path shown. External tools show their binary path in parentheses.

### Enabling tools in a prompt

Use `-T <tool_name>` (repeatable) to make tools available to the LLM. The tool name is the `name` field from the schema, not the binary name.

```bash
# Single tool
llm "What is 'hello world' in uppercase?" -T upper

# Multiple tools
llm "What time is it? Also uppercase 'hello'" -T llm_time -T upper

# Mix built-in and external
llm "What version of llm is this, and count the words in 'four score and seven'" \
    -T llm_version -T word_count
```

### Enabling tools in chat

The same `-T` flag works with `llm chat`:

```bash
llm chat -T upper -T llm_time
> Make "hello" uppercase
[LLM calls upper tool, returns HELLO]
The uppercase version is: HELLO
> What time is it?
[LLM calls llm_time tool]
...
```

### Chain loop

When a tool is enabled, llm-rs runs a chain loop:

1. Send the prompt (with tool definitions) to the LLM.
2. If the LLM responds with tool calls, execute each tool.
3. Send the tool results back to the LLM.
4. Repeat until the LLM responds with text (no tool calls) or the chain limit is reached.

The default chain limit is 5 iterations. Override with `--chain-limit`:

```bash
llm "Do a complex multi-step task" -T tool1 -T tool2 --chain-limit 10
```

### Observing chain loop steps

There are three ways to see what happens during a chain loop:

**1. `--tools-debug` (real-time, stderr)**

Prints every tool call and its result to stderr as it happens. LLM response text streams to stdout in parallel.

```bash
$ llm "Make 'hello' uppercase, then tell me the result" -T upper --tools-debug
Tool call: upper (id: call_abc123)
Arguments: {"text":"hello"}
Tool result: HELLO
The uppercase version of "hello" is HELLO.
```

In a multi-iteration chain, you see each tool call in order:

```bash
$ llm "What version is this? Also what time is it?" -T llm_version -T llm_time --tools-debug
Tool call: llm_version (id: call_1)
Arguments: {}
Tool result: 0.1.0
Tool call: llm_time (id: call_2)
Arguments: {}
Tool result: {"utc_time":"2026-04-09T12:00:00Z","local_time":"2026-04-09T20:00:00+08:00","timezone":"CST"}
This is llm version 0.1.0, and the current time is ...
```

Note: `--tools-debug` shows the tool execution side (steps 2-3) but not the LLM request/response boundaries (step 1). There is no iteration counter or "calling LLM..." marker in the current implementation.

**2. `--tools-approve` (interactive, stderr)**

Prompts for confirmation before each tool execution. This lets you see what the LLM is about to call and decide whether to proceed:

```bash
$ llm "Make 'hello' uppercase" -T upper --tools-approve
Execute tool upper? [y/N] y
HELLO
```

Declining a tool call sends an error (`"user declined"`) back to the LLM, which may retry or explain.

**3. `--json` (after completion, stdout)**

The JSON output envelope includes `tool_calls` from the final chain iteration. This shows which tools were called but not the intermediate steps or tool results:

```bash
$ llm "Make 'hello' uppercase" -T upper --json --no-stream
{
  "model": "gpt-4o-mini",
  "content": "The uppercase version is HELLO.",
  "tool_calls": [
    {"name": "upper", "arguments": {"text": "hello"}, "tool_call_id": "call_abc123"}
  ],
  "duration_ms": 1234
}
```

**4. JSONL conversation logs (after completion, filesystem)**

Every prompt is logged to `~/.local/share/llm/logs/` (unless `-n` is used). The log record includes both `tool_calls` and `tool_results` accumulated across all chain iterations:

```bash
# Find the latest log
llm logs list -r

# Inspect the raw JSONL
cat $(llm logs path)/*.jsonl | python3 -m json.tool
```

The `tool_calls` and `tool_results` arrays in the log capture the full set from all iterations, but they are flattened into a single response record --- there is no per-iteration breakdown in the log format.

**What is not currently observable:**

- Which chain iteration a tool call belongs to (no iteration counter).
- The exact prompt sent to the LLM at each iteration (message history growth).
- Token usage per iteration (only the final usage is captured).

These would require a `--verbose` flag on the chain loop (planned for future work).

### Viewing in plugins list

```bash
$ llm plugins list
Compiled providers:
  openai (2 models: gpt-4o, gpt-4o-mini)
  anthropic (3 models: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)

External tools:
  upper (/usr/local/bin/llm-tool-upper) — Convert text to uppercase
  word_count (/usr/local/bin/llm-tool-wc) — Count words, lines, and characters in text
```

---

## Resolution order

When `-T <name>` is given, llm-rs resolves the name in this order:

1. **Built-in tools** --- checked first. Built-in names (`llm_version`, `llm_time`) always take precedence.
2. **External tools** --- if not found in builtins, PATH is scanned for `llm-tool-*` binaries and their schema `name` fields are matched.
3. **Error** --- if the name is not found in either, llm exits with code 2: `"unknown tool: <name>"`.

This means an external tool cannot shadow a built-in tool. If you create `llm-tool-version` with `"name":"llm_version"`, the built-in `llm_version` will always be used instead.

---

## Testing your tool

### Manual testing

Test the schema:

```bash
$ llm-tool-upper --schema
{"name":"upper","description":"Convert text to uppercase",...}

# Verify it's valid JSON
$ llm-tool-upper --schema | python3 -m json.tool
```

Test execution:

```bash
$ echo '{"text":"hello"}' | llm-tool-upper
HELLO

# Test error handling
$ echo '{}' | llm-tool-upper
# (should handle missing "text" gracefully)
```

Test discovery:

```bash
$ llm tools list | grep upper
upper: Convert text to uppercase (/path/to/llm-tool-upper)
```

### Automated testing

You can test tools without a real LLM by using `--tools-debug` with a mock server:

```bash
# Point to a wiremock or similar that returns a tool call
OPENAI_BASE_URL=http://localhost:8080 \
    llm "test" -T upper --no-stream --tools-debug -m gpt-4o-mini
```

Or test the tool binary in isolation (recommended for CI):

```bash
#!/bin/sh
# test_upper.sh

# Test schema
schema=$(./llm-tool-upper --schema)
echo "$schema" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['name']=='upper'"

# Test execution
result=$(echo '{"text":"hello"}' | ./llm-tool-upper)
test "$result" = "HELLO" || { echo "FAIL: expected HELLO, got $result"; exit 1; }

# Test error exit
echo '{"bad":"input"}' | ./llm-tool-upper; code=$?
# (depends on your error handling)

echo "All tests passed"
```

---

## Reference

### Schema JSON shape

```
Tool {
    name: String,            // logical name, used with -T
    description: String,     // shown to the LLM and in tool listings
    input_schema: Value,     // JSON Schema object (must be type: "object")
}
```

### What the tool receives

- **argv**: no arguments (the binary is invoked bare, not with `--schema`).
- **stdin**: a JSON object matching `input_schema`. Always a single line. Always valid JSON (generated by the LLM provider).
- **env**: inherits the parent process environment.
- **cwd**: inherits the parent process working directory.

### What the tool returns

- **stdout**: the tool's result as plain text (not JSON, unless the tool's purpose is to return structured data). Captured fully after the process exits.
- **stderr**: error messages (only read on non-zero exit).
- **exit code**: 0 for success, non-zero for error.

### Timeouts

Default: 30 seconds. If the tool process does not exit within this window, it is killed and the LLM receives `"tool <name> timed out"`.

The timeout is currently not user-configurable from the command line (future work).

### Supported JSON Schema types in `input_schema`

The LLM providers (OpenAI, Anthropic) support standard JSON Schema types:

- `"string"` --- text values
- `"integer"` --- whole numbers
- `"number"` --- floating-point numbers
- `"boolean"` --- true/false
- `"array"` --- with `"items"` defining element type
- `"object"` --- nested objects with `"properties"`

Use `"description"` on individual properties to help the LLM generate better arguments. Use `"enum"` to constrain values. Use `"required"` to mark mandatory parameters.

The `input_schema` top-level must be `"type": "object"`. Array or scalar top-level schemas are not supported.
