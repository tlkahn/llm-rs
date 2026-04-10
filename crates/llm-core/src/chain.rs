use async_trait::async_trait;
use futures::StreamExt;

use crate::provider::Provider;
use crate::stream::{Chunk, collect_text, collect_tool_calls, collect_usage};
use crate::types::{Message, Prompt, ToolCall, ToolResult, Usage};

/// Trait for executing tool calls. Implement this to provide tool execution logic.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, call: &ToolCall) -> ToolResult;
}

/// Event emitted during chain loop execution for observability.
#[derive(Debug, Clone)]
pub enum ChainEvent {
    /// Emitted before the provider is called for an iteration.
    IterationStart {
        /// 1-based iteration number.
        iteration: usize,
        /// The chain limit.
        limit: usize,
        /// Current message history being sent to the provider.
        messages: Vec<Message>,
    },
    /// Emitted after an iteration completes (chunks collected, tool calls extracted).
    IterationEnd {
        /// 1-based iteration number.
        iteration: usize,
        /// Per-iteration token usage, if the provider reported it.
        usage: Option<Usage>,
        /// Cumulative usage across all iterations up to and including this one.
        cumulative_usage: Option<Usage>,
        /// Tool calls extracted from this iteration's response.
        tool_calls: Vec<ToolCall>,
    },
    /// Emitted when the budget is exhausted (after completing the current iteration).
    BudgetExhausted {
        /// Cumulative usage at the point the budget was exceeded.
        cumulative_usage: Usage,
        /// The budget limit that was exceeded.
        budget: u64,
    },
}

/// Result of a chain loop execution.
pub struct ChainResult {
    /// All chunks from all iterations.
    pub chunks: Vec<Chunk>,
    /// All tool results from all iterations (in execution order).
    pub tool_results: Vec<ToolResult>,
    /// Accumulated usage across all iterations.
    pub total_usage: Option<Usage>,
    /// Whether the chain stopped because the budget was exhausted.
    pub budget_exhausted: bool,
}

/// Run a chain loop: execute -> collect tool calls -> execute tools -> repeat.
///
/// Stops when:
/// - No tool calls are returned (normal completion)
/// - `chain_limit` iterations are reached
/// - `budget` is exceeded (graceful stop after completing current iteration)
///
/// `on_chunk` is called for every chunk from every iteration.
///
/// The chain accumulates a `Vec<Message>` across iterations so that each
/// provider call sees the full conversation history (user, assistant+tools,
/// tool results, ...).
#[allow(clippy::too_many_arguments)]
pub async fn chain(
    provider: &dyn Provider,
    model: &str,
    initial_prompt: Prompt,
    key: Option<&str>,
    stream: bool,
    executor: &dyn ToolExecutor,
    chain_limit: usize,
    on_chunk: &mut dyn FnMut(&Chunk),
    on_event: Option<&mut dyn FnMut(&ChainEvent)>,
    budget: Option<u64>,
) -> crate::Result<ChainResult> {
    let mut all_chunks = Vec::new();
    let mut all_tool_results = Vec::new();
    let mut on_event = on_event;
    let mut cumulative_usage: Option<Usage> = None;
    let mut budget_exhausted = false;

    // Seed messages from initial prompt
    let mut messages: Vec<Message> = if initial_prompt.messages.is_empty() {
        vec![Message::user(&initial_prompt.text)]
    } else {
        initial_prompt.messages.clone()
    };

    for iteration in 1..=chain_limit {
        if let Some(cb) = &mut on_event {
            cb(&ChainEvent::IterationStart {
                iteration,
                limit: chain_limit,
                messages: messages.clone(),
            });
        }

        // Build prompt with accumulated messages + preserved metadata
        let mut prompt = Prompt::new(&initial_prompt.text)
            .with_tools(initial_prompt.tools.clone())
            .with_messages(messages.clone());
        if let Some(system) = &initial_prompt.system {
            prompt = prompt.with_system(system);
        }
        if let Some(schema) = &initial_prompt.schema {
            prompt = prompt.with_schema(schema.clone());
        }

        let response_stream = provider.execute(model, &prompt, key, stream).await?;

        let mut iteration_chunks = Vec::new();
        let mut pinned = std::pin::pin!(response_stream);

        while let Some(result) = pinned.next().await {
            let chunk = result?;
            on_chunk(&chunk);
            iteration_chunks.push(chunk);
        }

        let tool_calls = collect_tool_calls(&iteration_chunks);
        let usage = collect_usage(&iteration_chunks);
        let text = collect_text(&iteration_chunks);

        // Accumulate usage
        cumulative_usage = match (&cumulative_usage, &usage) {
            (Some(cum), Some(iter_usage)) => Some(cum.add(iter_usage)),
            (None, Some(iter_usage)) => Some(iter_usage.clone()),
            (cum, None) => cum.clone(),
        };

        if let Some(cb) = &mut on_event {
            cb(&ChainEvent::IterationEnd {
                iteration,
                usage: usage.clone(),
                cumulative_usage: cumulative_usage.clone(),
                tool_calls: tool_calls.clone(),
            });
        }

        all_chunks.extend(iteration_chunks);

        // Append assistant message to history
        messages.push(Message::assistant_with_tool_calls(&text, tool_calls.clone()));

        if tool_calls.is_empty() {
            break;
        }

        // Check budget after completing the iteration
        if let (Some(b), Some(cum)) = (budget, &cumulative_usage)
            && cum.total() >= b
        {
            budget_exhausted = true;
            if let Some(cb) = &mut on_event {
                cb(&ChainEvent::BudgetExhausted {
                    cumulative_usage: cum.clone(),
                    budget: b,
                });
            }
            break;
        }

        // Execute all tool calls
        let mut tool_results = Vec::new();
        for call in &tool_calls {
            let result = executor.execute(call).await;
            tool_results.push(result);
        }

        all_tool_results.extend(tool_results.clone());

        // Append tool results to history
        messages.push(Message::tool_results(tool_results));
    }

    Ok(ChainResult {
        chunks: all_chunks,
        tool_results: all_tool_results,
        total_usage: cumulative_usage,
        budget_exhausted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;
    use crate::stream::ResponseStream;
    use crate::types::{ModelInfo, Tool};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // Mock provider that returns pre-configured responses and captures prompts
    struct MockProvider {
        responses: Vec<Vec<Chunk>>,
        call_count: AtomicUsize,
        captured_prompts: Arc<Mutex<Vec<Prompt>>>,
    }

    impl MockProvider {
        fn new(responses: Vec<Vec<Chunk>>) -> Self {
            Self {
                responses,
                call_count: AtomicUsize::new(0),
                captured_prompts: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    impl Provider for MockProvider {
        fn id(&self) -> &str {
            "mock"
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo::new("mock-model")]
        }
        async fn execute(
            &self,
            _model: &str,
            prompt: &Prompt,
            _key: Option<&str>,
            _stream: bool,
        ) -> crate::Result<ResponseStream> {
            self.captured_prompts.lock().unwrap().push(prompt.clone());
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let chunks = if idx < self.responses.len() {
                self.responses[idx].clone()
            } else {
                // Fallback: return last response
                self.responses.last().cloned().unwrap_or_default()
            };
            let items: Vec<Result<Chunk, LlmError>> = chunks.into_iter().map(Ok).collect();
            Ok(Box::pin(futures::stream::iter(items)))
        }
    }

    // Mock executor
    struct MockExecutor;

    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    impl ToolExecutor for MockExecutor {
        async fn execute(&self, call: &ToolCall) -> ToolResult {
            ToolResult {
                name: call.name.clone(),
                output: format!("result for {}", call.name),
                tool_call_id: call.tool_call_id.clone(),
                error: None,
            }
        }
    }

    struct ErrorExecutor;

    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    impl ToolExecutor for ErrorExecutor {
        async fn execute(&self, call: &ToolCall) -> ToolResult {
            ToolResult {
                name: call.name.clone(),
                output: String::new(),
                tool_call_id: call.tool_call_id.clone(),
                error: Some("tool failed".into()),
            }
        }
    }

    fn text_response(text: &str) -> Vec<Chunk> {
        vec![Chunk::Text(text.into()), Chunk::Done]
    }

    fn tool_call_response(name: &str, id: &str, args: &str) -> Vec<Chunk> {
        vec![
            Chunk::ToolCallStart {
                name: name.into(),
                id: Some(id.into()),
            },
            Chunk::ToolCallDelta {
                content: args.into(),
            },
            Chunk::Done,
        ]
    }

    fn make_tool() -> Tool {
        Tool {
            name: "test_tool".into(),
            description: "A test".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    #[tokio::test]
    async fn chain_no_tool_calls_single_iteration() {
        let provider = MockProvider::new(vec![text_response("Hello!")]);
        let prompt = Prompt::new("Hi").with_tools(vec![make_tool()]);
        let mut callback_count = 0;

        let result = chain(
            &provider,
            "mock-model",
            prompt,
            None,
            false,
            &MockExecutor,
            5,
            &mut |_| callback_count += 1,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(crate::collect_text(&result.chunks), "Hello!");
        assert!(result.tool_results.is_empty());
        assert_eq!(callback_count, 2); // Text + Done
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn chain_single_tool_call_two_iterations() {
        let provider = MockProvider::new(vec![
            tool_call_response("test_tool", "tc_1", "{}"),
            text_response("Done!"),
        ]);
        let prompt = Prompt::new("Do something").with_tools(vec![make_tool()]);

        let result = chain(
            &provider,
            "mock-model",
            prompt,
            None,
            false,
            &MockExecutor,
            5,
            &mut |_| {},
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(crate::collect_text(&result.chunks), "Done!");
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].name, "test_tool");
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chain_limit_stops_loop() {
        // Always returns tool calls - should stop at limit
        let provider = MockProvider::new(vec![
            tool_call_response("test_tool", "tc_1", "{}"),
        ]);
        let prompt = Prompt::new("Loop").with_tools(vec![make_tool()]);

        let result = chain(
            &provider,
            "mock-model",
            prompt,
            None,
            false,
            &MockExecutor,
            3,
            &mut |_| {},
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(provider.call_count.load(Ordering::SeqCst), 3);
        assert_eq!(result.tool_results.len(), 3);
    }

    #[tokio::test]
    async fn chain_multiple_tool_calls() {
        let response = vec![
            Chunk::ToolCallStart {
                name: "tool_a".into(),
                id: Some("tc_1".into()),
            },
            Chunk::ToolCallDelta {
                content: "{}".into(),
            },
            Chunk::ToolCallStart {
                name: "tool_b".into(),
                id: Some("tc_2".into()),
            },
            Chunk::ToolCallDelta {
                content: "{}".into(),
            },
            Chunk::Done,
        ];

        let provider = MockProvider::new(vec![response, text_response("All done")]);
        let prompt = Prompt::new("Do both").with_tools(vec![make_tool()]);

        let result = chain(
            &provider,
            "mock-model",
            prompt,
            None,
            false,
            &MockExecutor,
            5,
            &mut |_| {},
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(crate::collect_text(&result.chunks), "All done");
        assert_eq!(result.tool_results.len(), 2);
        assert_eq!(result.tool_results[0].name, "tool_a");
        assert_eq!(result.tool_results[1].name, "tool_b");
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chain_tool_error_continues() {
        let provider = MockProvider::new(vec![
            tool_call_response("test_tool", "tc_1", "{}"),
            text_response("Handled error"),
        ]);
        let prompt = Prompt::new("Try").with_tools(vec![make_tool()]);

        let result = chain(
            &provider,
            "mock-model",
            prompt,
            None,
            false,
            &ErrorExecutor,
            5,
            &mut |_| {},
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(crate::collect_text(&result.chunks), "Handled error");
        assert_eq!(result.tool_results.len(), 1);
        assert!(result.tool_results[0].error.is_some());
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chain_callback_receives_chunks() {
        let provider = MockProvider::new(vec![text_response("Hi")]);
        let prompt = Prompt::new("Hello").with_tools(vec![make_tool()]);
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_clone = received.clone();

        let _ = chain(
            &provider,
            "mock-model",
            prompt,
            None,
            false,
            &MockExecutor,
            5,
            &mut |chunk| received_clone.lock().unwrap().push(chunk.clone()),
            None,
            None,
        )
        .await
        .unwrap();

        let chunks = received.lock().unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(matches!(&chunks[0], Chunk::Text(t) if t == "Hi"));
        assert!(matches!(&chunks[1], Chunk::Done));
    }

    #[tokio::test]
    async fn chain_accumulates_messages_across_turns() {
        // 3-iteration test: tool call → tool call → text
        let provider = MockProvider::new(vec![
            tool_call_response("test_tool", "tc_1", "{}"),
            tool_call_response("test_tool", "tc_2", "{}"),
            text_response("Done!"),
        ]);
        let prompt = Prompt::new("Do it").with_tools(vec![make_tool()]);

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        let prompts = provider.captured_prompts.lock().unwrap();
        assert_eq!(prompts.len(), 3);

        // Iteration 1: [user]
        assert_eq!(prompts[0].messages.len(), 1);
        assert_eq!(prompts[0].messages[0].role, crate::Role::User);

        // Iteration 2: [user, assistant+tools, tool_results]
        assert_eq!(prompts[1].messages.len(), 3);
        assert_eq!(prompts[1].messages[0].role, crate::Role::User);
        assert_eq!(prompts[1].messages[1].role, crate::Role::Assistant);
        assert!(!prompts[1].messages[1].tool_calls.is_empty());
        assert_eq!(prompts[1].messages[2].role, crate::Role::Tool);

        // Iteration 3: [user, assistant+tools, tool_results, assistant+tools, tool_results]
        assert_eq!(prompts[2].messages.len(), 5);
    }

    #[tokio::test]
    async fn chain_preserves_initial_messages() {
        let initial = vec![
            Message::user("Earlier question"),
            Message::assistant("Earlier answer"),
        ];
        let provider = MockProvider::new(vec![text_response("Follow up done")]);
        let prompt = Prompt::new("Follow up")
            .with_tools(vec![make_tool()])
            .with_messages(initial);

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        let prompts = provider.captured_prompts.lock().unwrap();
        // Should see initial 2 messages preserved
        assert_eq!(prompts[0].messages.len(), 2);
        assert_eq!(prompts[0].messages[0].content, "Earlier question");
        assert_eq!(prompts[0].messages[1].content, "Earlier answer");
    }

    #[tokio::test]
    async fn chain_captures_assistant_text_in_history() {
        // Provider returns text + tool call in first response
        let response1 = vec![
            Chunk::Text("Let me check. ".into()),
            Chunk::ToolCallStart { name: "test_tool".into(), id: Some("tc_1".into()) },
            Chunk::ToolCallDelta { content: "{}".into() },
            Chunk::Done,
        ];
        let provider = MockProvider::new(vec![response1, text_response("All done")]);
        let prompt = Prompt::new("Do it").with_tools(vec![make_tool()]);

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        let prompts = provider.captured_prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2);
        // Second prompt should have assistant message with both text and tool_calls
        let assistant = &prompts[1].messages[1];
        assert_eq!(assistant.role, crate::Role::Assistant);
        assert_eq!(assistant.content, "Let me check. ");
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].name, "test_tool");
    }

    #[tokio::test]
    async fn chain_emits_iteration_start_event() {
        let provider = MockProvider::new(vec![text_response("Hello!")]);
        let prompt = Prompt::new("Hi").with_tools(vec![make_tool()]);
        let mut events = Vec::new();

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {},
            Some(&mut |e: &ChainEvent| events.push(e.clone())),
            None,
        ).await.unwrap();

        assert_eq!(events.len(), 2); // IterationStart + IterationEnd
        match &events[0] {
            ChainEvent::IterationStart { iteration, limit, messages } => {
                assert_eq!(*iteration, 1);
                assert_eq!(*limit, 5);
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, crate::Role::User);
            }
            _ => panic!("expected IterationStart"),
        }
        match &events[1] {
            ChainEvent::IterationEnd { iteration, usage, cumulative_usage, tool_calls } => {
                assert_eq!(*iteration, 1);
                assert!(usage.is_none());
                assert!(cumulative_usage.is_none());
                assert!(tool_calls.is_empty());
            }
            _ => panic!("expected IterationEnd"),
        }
    }

    #[tokio::test]
    async fn chain_emits_per_iteration_usage() {
        let response1 = vec![
            Chunk::ToolCallStart { name: "test_tool".into(), id: Some("tc_1".into()) },
            Chunk::ToolCallDelta { content: "{}".into() },
            Chunk::Usage(Usage { input: Some(10), output: Some(5), details: None }),
            Chunk::Done,
        ];
        let response2 = vec![
            Chunk::Text("Done".into()),
            Chunk::Usage(Usage { input: Some(20), output: Some(10), details: None }),
            Chunk::Done,
        ];
        let provider = MockProvider::new(vec![response1, response2]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);
        let mut events = Vec::new();

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {},
            Some(&mut |e: &ChainEvent| events.push(e.clone())),
            None,
        ).await.unwrap();

        // 2 iterations -> 4 events (start, end, start, end)
        assert_eq!(events.len(), 4);
        match &events[1] {
            ChainEvent::IterationEnd { iteration, usage, cumulative_usage, tool_calls } => {
                assert_eq!(*iteration, 1);
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input, Some(10));
                assert_eq!(u.output, Some(5));
                let cum = cumulative_usage.as_ref().unwrap();
                assert_eq!(cum.input, Some(10));
                assert_eq!(cum.output, Some(5));
                assert_eq!(tool_calls.len(), 1);
            }
            _ => panic!("expected IterationEnd"),
        }
        match &events[3] {
            ChainEvent::IterationEnd { iteration, usage, cumulative_usage, tool_calls } => {
                assert_eq!(*iteration, 2);
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input, Some(20));
                assert_eq!(u.output, Some(10));
                let cum = cumulative_usage.as_ref().unwrap();
                assert_eq!(cum.input, Some(30));
                assert_eq!(cum.output, Some(15));
                assert!(tool_calls.is_empty());
            }
            _ => panic!("expected IterationEnd"),
        }
    }

    #[tokio::test]
    async fn chain_events_correct_sequence() {
        // 3-iteration chain: tool -> tool -> text
        let provider = MockProvider::new(vec![
            tool_call_response("test_tool", "tc_1", "{}"),
            tool_call_response("test_tool", "tc_2", "{}"),
            text_response("Done!"),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);
        let mut events = Vec::new();

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {},
            Some(&mut |e: &ChainEvent| events.push(e.clone())),
            None,
        ).await.unwrap();

        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], ChainEvent::IterationStart { iteration: 1, .. }));
        assert!(matches!(&events[1], ChainEvent::IterationEnd { iteration: 1, .. }));
        assert!(matches!(&events[2], ChainEvent::IterationStart { iteration: 2, .. }));
        assert!(matches!(&events[3], ChainEvent::IterationEnd { iteration: 2, .. }));
        assert!(matches!(&events[4], ChainEvent::IterationStart { iteration: 3, .. }));
        assert!(matches!(&events[5], ChainEvent::IterationEnd { iteration: 3, .. }));

        // Verify tool calls in end events
        if let ChainEvent::IterationEnd { tool_calls, cumulative_usage, .. } = &events[1] {
            assert_eq!(tool_calls.len(), 1);
            assert!(cumulative_usage.is_none()); // no usage in mock tool_call_response
        }
        if let ChainEvent::IterationEnd { tool_calls, .. } = &events[5] {
            assert!(tool_calls.is_empty());
        }

        // Verify message growth in start events
        if let ChainEvent::IterationStart { messages, .. } = &events[0] {
            assert_eq!(messages.len(), 1); // [user]
        }
        if let ChainEvent::IterationStart { messages, .. } = &events[2] {
            assert_eq!(messages.len(), 3); // [user, assistant+tools, tool]
        }
        if let ChainEvent::IterationStart { messages, .. } = &events[4] {
            assert_eq!(messages.len(), 5); // [user, a+t, tool, a+t, tool]
        }
    }

    #[tokio::test]
    async fn chain_none_on_event_works() {
        let provider = MockProvider::new(vec![
            tool_call_response("test_tool", "tc_1", "{}"),
            text_response("Done!"),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        assert_eq!(crate::collect_text(&result.chunks), "Done!");
        assert_eq!(result.tool_results.len(), 1);
    }

    // --- ChainResult.total_usage tests ---

    fn text_response_with_usage(text: &str, input: u64, output: u64) -> Vec<Chunk> {
        vec![
            Chunk::Text(text.into()),
            Chunk::Usage(Usage { input: Some(input), output: Some(output), details: None }),
            Chunk::Done,
        ]
    }

    fn tool_call_response_with_usage(name: &str, id: &str, args: &str, input: u64, output: u64) -> Vec<Chunk> {
        vec![
            Chunk::ToolCallStart { name: name.into(), id: Some(id.into()) },
            Chunk::ToolCallDelta { content: args.into() },
            Chunk::Usage(Usage { input: Some(input), output: Some(output), details: None }),
            Chunk::Done,
        ]
    }

    #[tokio::test]
    async fn chain_result_total_usage_single_iteration() {
        let provider = MockProvider::new(vec![
            text_response_with_usage("Hello!", 10, 5),
        ]);
        let prompt = Prompt::new("Hi").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        let usage = result.total_usage.unwrap();
        assert_eq!(usage.input, Some(10));
        assert_eq!(usage.output, Some(5));
        assert!(!result.budget_exhausted);
    }

    #[tokio::test]
    async fn chain_result_total_usage_multi_iteration() {
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 10, 5),
            text_response_with_usage("Done!", 20, 10),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        let usage = result.total_usage.unwrap();
        assert_eq!(usage.input, Some(30));
        assert_eq!(usage.output, Some(15));
    }

    #[tokio::test]
    async fn chain_result_total_usage_none() {
        let provider = MockProvider::new(vec![text_response("Hello!")]);
        let prompt = Prompt::new("Hi").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        assert!(result.total_usage.is_none());
    }

    // --- cumulative_usage in ChainEvent::IterationEnd ---

    #[tokio::test]
    async fn chain_event_cumulative_usage() {
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 10, 5),
            text_response_with_usage("Done!", 20, 10),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);
        let mut events = Vec::new();

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {},
            Some(&mut |e: &ChainEvent| events.push(e.clone())),
            None,
        ).await.unwrap();

        // 2 iterations -> 4 events
        assert_eq!(events.len(), 4);

        // Iter 1 end: cumulative = (10, 5)
        if let ChainEvent::IterationEnd { cumulative_usage, .. } = &events[1] {
            let cum = cumulative_usage.as_ref().unwrap();
            assert_eq!(cum.input, Some(10));
            assert_eq!(cum.output, Some(5));
        } else {
            panic!("expected IterationEnd");
        }

        // Iter 2 end: cumulative = (30, 15)
        if let ChainEvent::IterationEnd { cumulative_usage, .. } = &events[3] {
            let cum = cumulative_usage.as_ref().unwrap();
            assert_eq!(cum.input, Some(30));
            assert_eq!(cum.output, Some(15));
        } else {
            panic!("expected IterationEnd");
        }
    }

    // --- Budget enforcement tests ---

    #[tokio::test]
    async fn chain_budget_stops_when_exceeded() {
        // budget=25, iter1 usage=30 (10+20) → stops after 1 iteration
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 10, 20),
            text_response_with_usage("Should not reach", 10, 10),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, Some(25),
        ).await.unwrap();

        assert!(result.budget_exhausted);
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 1);
        let usage = result.total_usage.unwrap();
        assert_eq!(usage.total(), 30);
    }

    #[tokio::test]
    async fn chain_budget_allows_under() {
        // budget=100, iter1 usage=15 → text response, continues normally
        let provider = MockProvider::new(vec![
            text_response_with_usage("Hello!", 10, 5),
        ]);
        let prompt = Prompt::new("Hi").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, Some(100),
        ).await.unwrap();

        assert!(!result.budget_exhausted);
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn chain_budget_multi_iteration_accumulates() {
        // budget=40, iter1=15, iter2=15, iter3 would exceed → stops after 2
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 10, 5),
            tool_call_response_with_usage("test_tool", "tc_2", "{}", 10, 5),
            text_response_with_usage("Should not reach", 10, 5),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, Some(40),
        ).await.unwrap();

        // iter1: 15 total (under 40), iter2: 30 total (under 40) → both allowed
        // Actually 30 < 40, so it should continue. Let me set budget to 25 instead.
        assert!(!result.budget_exhausted);
        // With budget=40 and 15 per iter, it will do 2 tool iterations (30 total < 40)
        // then the 3rd would run (15+15+15=45 > 40 would trigger IF there were tool calls)
        // But actually iter 3 is text, so it stops naturally
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn chain_budget_multi_iteration_stops() {
        // budget=25, iter1=15 (ok), iter2=15 (cumulative=30 > 25) → stops
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 10, 5),
            tool_call_response_with_usage("test_tool", "tc_2", "{}", 10, 5),
            text_response_with_usage("Should not reach", 10, 5),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, Some(25),
        ).await.unwrap();

        assert!(result.budget_exhausted);
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 2);
        let usage = result.total_usage.unwrap();
        assert_eq!(usage.total(), 30);
    }

    #[tokio::test]
    async fn chain_budget_none_no_enforcement() {
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 100, 100),
            text_response_with_usage("Done!", 100, 100),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);

        let result = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {}, None, None,
        ).await.unwrap();

        assert!(!result.budget_exhausted);
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chain_budget_emits_event() {
        let provider = MockProvider::new(vec![
            tool_call_response_with_usage("test_tool", "tc_1", "{}", 20, 15),
            text_response_with_usage("Should not reach", 10, 10),
        ]);
        let prompt = Prompt::new("Go").with_tools(vec![make_tool()]);
        let mut events = Vec::new();

        let _ = chain(
            &provider, "mock-model", prompt, None, false,
            &MockExecutor, 5, &mut |_| {},
            Some(&mut |e: &ChainEvent| events.push(e.clone())),
            Some(30),
        ).await.unwrap();

        // Should have: IterationStart, IterationEnd, BudgetExhausted
        assert_eq!(events.len(), 3);
        match &events[2] {
            ChainEvent::BudgetExhausted { cumulative_usage, budget } => {
                assert_eq!(*budget, 30);
                assert_eq!(cumulative_usage.total(), 35);
            }
            _ => panic!("expected BudgetExhausted, got {:?}", events[2]),
        }
    }
}
