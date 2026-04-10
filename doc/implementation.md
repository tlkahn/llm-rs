# Implementation Notes

Pitfall journal — gotchas and workarounds discovered during implementation. For current types, APIs, and conventions see [CLAUDE.md](../CLAUDE.md). For design rationale see [architecture.md](design/architecture.md). For status and roadmap see [roadmap.md](roadmap.md).

---

## Serde & serialization

**LineRecord tagged enum + flatten.** `LineRecord` uses `#[serde(tag = "type")]` to dispatch `"conversation"` vs `"response"`. `ResponseRecord` uses `#[serde(flatten)]` on its inner `Response` to keep fields at the top level. The `Response` variant is `Box<ResponseRecord>` to satisfy clippy's `large_enum_variant` lint.

**ContentBlock fields must all be optional.** Anthropic's `ContentBlock` serves both `text` and `tool_use` blocks. An enum would conflict with `#[serde(untagged)]` on `MessageContent`. All fields are `Option` with `skip_serializing_if` — verbose but avoids serde ambiguity.

**Prompt.tool_calls backward compat.** Adding `tool_calls: Vec<ToolCall>` with `#[serde(default)]` maintains backward-compatible deserialization for existing log files.

**ProtocolChunk parallels Chunk.** `Chunk` is an internal streaming type without `Serialize`/`Deserialize` — intentionally, to avoid a cross-cutting serde concern on all crates. The subprocess protocol needs serializable chunks for JSONL, so `ProtocolChunk` is a parallel enum with serde derives and `From`/`Into` conversions.

---

## Provider-specific pitfalls

**Anthropic model IDs: use aliases, not snapshot dates.** Model IDs come as aliases (`claude-sonnet-4-6`) and dated snapshots (`claude-sonnet-4-6-20250514`). The initial implementation used speculative snapshot dates that didn't exist, causing cryptic API rejections. Lesson: use alias-form IDs for provider model lists; let users pass specific snapshots via `-m`.

**Anthropic structured output requires streaming state tracking.** The transparent `_schema_output` tool wrapping needs an `is_schema_block` boolean to track whether the current `content_block_start` is the synthetic tool. When true, `input_json_delta` chunks emit as `Chunk::Text` instead of `Chunk::ToolCallDelta`. The `has_schema` boolean must be captured from `prompt.schema.is_some()` before the async move. Edge case: when a schema prompt also has explicit tools, both coexist — `_schema_output` is appended to the tools list.

**Anthropic tool results go in user role.** Unlike OpenAI's `"role": "tool"`, Anthropic requires tool results in a `"role": "user"` message with `tool_result` content blocks. `Message::tool_results()` uses `Role::Tool` abstractly; the Anthropic conversation builder remaps it.

**resolve_key() fails for keyless providers.** When `provider.needs_key()` returns `None`, the code mapped it to `""` via `.unwrap_or("")`, then `resolve_key("", ...)` failed. Fix: skip `resolve_key()` entirely when the provider doesn't need a key and no `--key` flag is given. Change key from `String` to `Option<String>`. This never appeared in phases 1-3 because both compiled-in providers always require keys.

---

## Chain loop

**System prompt preservation.** The chain loop builds a new `Prompt` each iteration with fresh tool_calls/tool_results. An early bug dropped the system prompt by not re-applying `with_system()`. Fixed by carrying forward `current_prompt.system`.

**Surface tool results, not just chunks.** The initial `chain()` returned `Vec<Chunk>`, so logged responses had `tool_results: []` even though tools executed successfully — results were consumed into the next prompt and discarded. Fix: `ChainResult { chunks, tool_results }` accumulates across iterations. This bug lived in the seam between `chain()` (llm-core) and `run()` (llm-cli). Unit tests missed it because they don't touch the logging layer; a live smoke test (`llm "What time is it?" -T llm_time`) caught it.

**History accumulation across iterations.** The Phase 2 chain loop rebuilt the prompt each iteration with only the latest tool_calls/tool_results, so iteration 3 had no memory of iteration 1. Fix: maintain a `Vec<Message>` that grows across iterations. Provider conversation paths dispatch on `prompt.messages.is_empty()` — empty uses the existing single-turn path, non-empty uses the multi-turn path, keeping all existing tests green.

---

## CLI workarounds

**Clap default subcommand via argv rewriting.** Clap has no native default subcommand. `main.rs::rewrite_args()` inserts `"prompt"` at position 1 when the first arg isn't a known subcommand or global flag. When no args and stdin is piped, also inserts `"prompt"`.

**Config mutation via toml::Table.** `llm models default <model>` uses `toml::Table` read-modify-write to preserve unknown fields, avoiding `Config` serde roundtrip that could drop unknown keys. Phase 3 added `Config::save()` for `logs on/off` because that modifies a typed boolean field.

**`--messages -` conflicts with stdin prompt text.** Both `--messages -` and `resolve_prompt_text()` try to read stdin. Fix: `skip_stdin` parameter prevents double-reading.

**`--json` disables streaming.** When `--json` is set, streaming is forced off and the chunk callback suppressed. The full response is buffered, then a JSON envelope emitted.

---

## wasm32 platform abstraction

Surgical cfg-gating — `llm-core` had `tokio` as a dependency but never used it in production code. The actual platform-dependent code was 3 lines in `llm-openai`.

| Location | Native | wasm32 | Why |
|----------|--------|--------|-----|
| `ResponseStream` type alias | `+ Send` | no `Send` | wasm32 is single-threaded; web-sys types aren't `Send` |
| `Provider` trait bounds | `Send + Sync`, `#[async_trait]` | no bounds, `#[async_trait(?Send)]` | Same |
| Streaming spawn | `tokio::spawn` | `wasm_bindgen_futures::spawn_local` | Different runtimes |

**`futures::channel::mpsc` everywhere, not just wasm32.** Unconditional switch from `tokio::sync::mpsc` avoids duplicating the SSE parsing loop. `futures::channel::mpsc::Receiver` implements `Stream` directly, eliminating `ReceiverStream`. Backpressure equivalent at buffer size 32.

**`cfg_attr` for impl blocks, duplication for trait.** The `Provider` trait must be duplicated across two cfg blocks because `#[async_trait]` and `#[async_trait(?Send)]` are different proc macro invocations. Impl blocks use `#[cfg_attr]` to avoid body duplication.

---

## Python / PyO3

**`Mutex` wrapper for PyO3 `Sync` requirement.** `prompt_stream()` returns a `ChunkIterator` backed by `std::sync::mpsc::Receiver`. The `Receiver` must be wrapped in `Mutex` because PyO3 requires `Sync` on `#[pyclass]` structs.

---

## Subprocess extensibility

**`providers()` had to become async.** `block_on` panics inside a tokio runtime. Subprocess discovery uses `tokio::process::Command`, so `providers()` became async and all callers now `.await` it. `models::run` also became async as a consequence.

**ExternalToolExecutor ownership across chat turns.** Moving it into `CliToolExecutor::with_external()` transfers ownership, making it unavailable for the next REPL turn. Fix: create the `CliToolExecutor` once before the loop and reuse it.

**PATH scanning deduplication.** The same binary in multiple PATH directories must return only the first occurrence (matching `which` semantics). A `HashSet<String>` tracks seen filenames.

**Arguments-only stdin for tools.** The tool protocol sends only `arguments` JSON, not the full `ToolCall` envelope. Simpler for tool implementors — they don't parse a wrapper.

---

## Testing patterns

**Wiremock mock ordering for multi-step chains.** Wiremock's priority is bottom-up: later-registered mocks have higher priority. Pattern: register the default (final text) response first, then register the tool-call response with `up_to_n_times(1)`.

**`assert_cmd` may not capture stderr from async code.** A test for `--tools-debug` stderr showed empty stderr even though the tool chain ran. Potentially related to `assert_cmd` + `tokio::main` interaction. Workaround: test chain mechanics without depending on stderr capture.

**Shell script fixtures for subprocess testing.** Scripts in `tests/fixtures/bin/` implement the full tool/provider protocol, testing the actual subprocess boundary. Scripts use `python3 -c` for JSON parsing.

**Schema ID uses SipHash not blake2.** `std::hash::DefaultHasher` formatted as 16-char hex, avoiding `blake2` + `hex` deps. IDs won't match Python `llm`'s blake2b output — cross-tool compatibility was not required.

---

## Verbose observability

**`on_event` as `Option<&mut dyn FnMut>`.** Avoids heap allocation. `None` means zero overhead. The `let mut on_event = on_event;` rebinding is needed because `Option<&mut dyn FnMut>` requires the outer binding to be mutable for `if let Some(cb) = &mut on_event`.

**Per-iteration `collect_usage()` before `all_chunks.extend()`.** Moving chunks into `all_chunks` consumes them; collect usage first. The `usage.clone()` in `IterationEnd` is cheap (`Option<Usage>`).

**`format_chain_event` shared via `pub` on prompt module.** Called from `chat` via `super::prompt::format_chain_event()`. A shared `verbose.rs` module would be premature — move it when more commands need it.
