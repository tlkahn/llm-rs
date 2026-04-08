use async_trait::async_trait;
use futures::StreamExt;

use crate::provider::Provider;
use crate::stream::{Chunk, collect_tool_calls};
use crate::types::{Prompt, ToolCall, ToolResult};

/// Trait for executing tool calls. Implement this to provide tool execution logic.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, call: &ToolCall) -> ToolResult;
}

/// Result of a chain loop execution.
pub struct ChainResult {
    /// All chunks from all iterations.
    pub chunks: Vec<Chunk>,
    /// All tool results from all iterations (in execution order).
    pub tool_results: Vec<ToolResult>,
}

/// Run a chain loop: execute -> collect tool calls -> execute tools -> repeat.
///
/// Stops when:
/// - No tool calls are returned (normal completion)
/// - `chain_limit` iterations are reached
///
/// `on_chunk` is called for every chunk from every iteration.
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
) -> crate::Result<ChainResult> {
    let mut all_chunks = Vec::new();
    let mut all_tool_results = Vec::new();
    let mut current_prompt = initial_prompt;

    for _ in 0..chain_limit {
        let response_stream = provider
            .execute(model, &current_prompt, key, stream)
            .await?;

        let mut iteration_chunks = Vec::new();
        let mut pinned = std::pin::pin!(response_stream);

        while let Some(result) = pinned.next().await {
            let chunk = result?;
            on_chunk(&chunk);
            iteration_chunks.push(chunk);
        }

        let tool_calls = collect_tool_calls(&iteration_chunks);
        all_chunks.extend(iteration_chunks);

        if tool_calls.is_empty() {
            break;
        }

        // Execute all tool calls
        let mut tool_results = Vec::new();
        for call in &tool_calls {
            let result = executor.execute(call).await;
            tool_results.push(result);
        }

        all_tool_results.extend(tool_results.clone());

        // Build next prompt with tool context, preserving system and tools
        let mut next_prompt = Prompt::new(&current_prompt.text)
            .with_tools(current_prompt.tools.clone())
            .with_tool_calls(tool_calls)
            .with_tool_results(tool_results);
        if let Some(system) = &current_prompt.system {
            next_prompt = next_prompt.with_system(system);
        }
        current_prompt = next_prompt;
    }

    Ok(ChainResult {
        chunks: all_chunks,
        tool_results: all_tool_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;
    use crate::stream::ResponseStream;
    use crate::types::{ModelInfo, Tool};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Mock provider that returns pre-configured responses
    struct MockProvider {
        responses: Vec<Vec<Chunk>>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<Vec<Chunk>>) -> Self {
            Self {
                responses,
                call_count: AtomicUsize::new(0),
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
            _prompt: &Prompt,
            _key: Option<&str>,
            _stream: bool,
        ) -> crate::Result<ResponseStream> {
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
        )
        .await
        .unwrap();

        let chunks = received.lock().unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(matches!(&chunks[0], Chunk::Text(t) if t == "Hi"));
        assert!(matches!(&chunks[1], Chunk::Done));
    }
}
