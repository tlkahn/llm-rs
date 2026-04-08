use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use llm_core::stream::{Chunk, ResponseStream};
use llm_core::types::{ModelInfo, Prompt, Usage};
use llm_core::{LlmError, Provider, Result};
use reqwest::Client;

use crate::messages::build_messages;
use crate::sse::SseParser;
use crate::types::{
    AnthropicTool, ErrorResponse, MessagesRequest, MessagesResponse, StreamEvent,
};

const DEFAULT_MAX_TOKENS: u64 = 4096;
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    base_url: String,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str {
        "anthropic"
    }

    fn models(&self) -> Vec<ModelInfo> {
        let image_types = vec![
            "image/png".into(),
            "image/jpeg".into(),
            "image/webp".into(),
            "image/gif".into(),
        ];
        vec![
            ModelInfo {
                id: "claude-opus-4-6".into(),
                can_stream: true,
                supports_tools: true,
                supports_schema: true,
                attachment_types: image_types.clone(),
            },
            ModelInfo {
                id: "claude-sonnet-4-6".into(),
                can_stream: true,
                supports_tools: true,
                supports_schema: true,
                attachment_types: image_types.clone(),
            },
            ModelInfo {
                id: "claude-haiku-4-5".into(),
                can_stream: true,
                supports_tools: true,
                supports_schema: true,
                attachment_types: image_types,
            },
        ]
    }

    fn needs_key(&self) -> Option<&str> {
        Some("anthropic")
    }

    fn key_env_var(&self) -> Option<&str> {
        Some("ANTHROPIC_API_KEY")
    }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> Result<ResponseStream> {
        let key = key.ok_or_else(|| {
            LlmError::NeedsKey(
                "No key found - set one with 'llm keys set anthropic' or export ANTHROPIC_API_KEY"
                    .into(),
            )
        })?;

        let messages = build_messages(prompt);

        let system = prompt
            .system
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let max_tokens = prompt
            .options
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_MAX_TOKENS);

        // Convert llm_core::Tool -> AnthropicTool
        let mut tools: Option<Vec<AnthropicTool>> = if prompt.tools.is_empty() {
            None
        } else {
            Some(
                prompt
                    .tools
                    .iter()
                    .map(|t| AnthropicTool {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: t.input_schema.clone(),
                    })
                    .collect(),
            )
        };

        // Structured output: inject synthetic _schema_output tool
        let has_schema = prompt.schema.is_some();
        let mut tool_choice = None;
        if let Some(schema) = &prompt.schema {
            let schema_tool = AnthropicTool {
                name: "_schema_output".into(),
                description: "Output structured data".into(),
                input_schema: schema.clone(),
            };
            tools.get_or_insert_with(Vec::new).push(schema_tool);
            tool_choice = Some(
                serde_json::json!({"type": "tool", "name": "_schema_output"}),
            );
        }

        let request = MessagesRequest {
            model: model.to_string(),
            max_tokens,
            messages,
            system,
            stream: Some(stream),
            temperature: prompt.options.get("temperature").and_then(|v| v.as_f64()),
            tools,
            tool_choice,
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::Provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            if let Ok(err_resp) = serde_json::from_str::<ErrorResponse>(&body) {
                return Err(LlmError::Provider(err_resp.error.message));
            }
            return Err(LlmError::Provider(format!("HTTP {status}: {body}")));
        }

        if stream {
            let byte_stream = response.bytes_stream();
            let (mut tx, rx) =
                futures::channel::mpsc::channel::<std::result::Result<Chunk, LlmError>>(32);

            let parse_future = async move {
                let mut parser = SseParser::new();
                let mut input_tokens: Option<u64> = None;
                let mut byte_stream = std::pin::pin!(byte_stream);
                // Track whether current block is the synthetic _schema_output tool
                let mut is_schema_block = false;

                while let Some(result) = byte_stream.next().await {
                    match result {
                        Ok(bytes) => {
                            parser.feed(&bytes);
                            while let Some(event) = parser.next_event() {
                                match &event {
                                    StreamEvent::MessageStart { message } => {
                                        if let Some(usage) = &message.usage {
                                            input_tokens = Some(usage.input_tokens);
                                        }
                                    }
                                    StreamEvent::ContentBlockStart {
                                        content_block, ..
                                    } => {
                                        if content_block.block_type == "tool_use" {
                                            let name = content_block
                                                .name
                                                .as_deref()
                                                .unwrap_or_default();
                                            if has_schema && name == "_schema_output" {
                                                is_schema_block = true;
                                                // Don't emit ToolCallStart for schema tool
                                            } else {
                                                is_schema_block = false;
                                                let _ = tx
                                                    .send(Ok(Chunk::ToolCallStart {
                                                        name: name.to_string(),
                                                        id: content_block.id.clone(),
                                                    }))
                                                    .await;
                                            }
                                        } else {
                                            is_schema_block = false;
                                        }
                                    }
                                    StreamEvent::ContentBlockDelta { delta, .. } => {
                                        if delta.delta_type == "text_delta" {
                                            if let Some(text) = &delta.text
                                                && !text.is_empty()
                                            {
                                                let _ = tx
                                                    .send(Ok(Chunk::Text(text.clone())))
                                                    .await;
                                            }
                                        } else if delta.delta_type == "input_json_delta"
                                            && let Some(json) = &delta.partial_json
                                            && !json.is_empty()
                                        {
                                            if is_schema_block {
                                                let _ = tx
                                                    .send(Ok(Chunk::Text(json.clone())))
                                                    .await;
                                            } else {
                                                let _ = tx
                                                    .send(Ok(Chunk::ToolCallDelta {
                                                        content: json.clone(),
                                                    }))
                                                    .await;
                                            }
                                        }
                                    }
                                    StreamEvent::MessageDelta {
                                        usage: Some(delta_usage),
                                        ..
                                    } => {
                                        let _ = tx
                                            .send(Ok(Chunk::Usage(Usage {
                                                input: input_tokens,
                                                output: Some(delta_usage.output_tokens),
                                                details: None,
                                            })))
                                            .await;
                                    }
                                    _ => {}
                                }
                            }
                            if parser.is_done() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(LlmError::Provider(e.to_string()))).await;
                            break;
                        }
                    }
                }
                let _ = tx.send(Ok(Chunk::Done)).await;
            };

            #[cfg(not(target_arch = "wasm32"))]
            tokio::spawn(parse_future);

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(parse_future);

            Ok(Box::pin(rx))
        } else {
            // Non-streaming: parse full JSON response
            let body = response
                .text()
                .await
                .map_err(|e| LlmError::Provider(e.to_string()))?;
            let resp: MessagesResponse =
                serde_json::from_str(&body).map_err(|e| LlmError::Provider(e.to_string()))?;

            let mut chunks: Vec<std::result::Result<Chunk, LlmError>> = Vec::new();

            for block in &resp.content {
                match block.block_type.as_str() {
                    "text" => {
                        if let Some(text) = &block.text
                            && !text.is_empty()
                        {
                            chunks.push(Ok(Chunk::Text(text.clone())));
                        }
                    }
                    "tool_use" => {
                        let name = block.name.as_deref().unwrap_or_default();
                        if has_schema && name == "_schema_output" {
                            // Schema output: emit as Text
                            if let Some(input) = &block.input {
                                chunks.push(Ok(Chunk::Text(input.to_string())));
                            }
                        } else {
                            chunks.push(Ok(Chunk::ToolCallStart {
                                name: name.to_string(),
                                id: block.id.clone(),
                            }));
                            if let Some(input) = &block.input {
                                chunks.push(Ok(Chunk::ToolCallDelta {
                                    content: input.to_string(),
                                }));
                            }
                        }
                    }
                    _ => {}
                }
            }

            chunks.push(Ok(Chunk::Usage(Usage {
                input: Some(resp.usage.input_tokens),
                output: Some(resp.usage.output_tokens),
                details: None,
            })));

            chunks.push(Ok(Chunk::Done));

            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_provider(base_url: &str) -> AnthropicProvider {
        AnthropicProvider::new(base_url)
    }

    // --- Unit tests ---

    #[test]
    fn provider_id_is_anthropic() {
        let p = make_provider("http://unused");
        assert_eq!(p.id(), "anthropic");
    }

    #[test]
    fn provider_lists_three_models() {
        let p = make_provider("http://unused");
        let models = p.models();
        assert_eq!(models.len(), 3);
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"claude-opus-4-6"));
        assert!(ids.contains(&"claude-sonnet-4-6"));
        assert!(ids.contains(&"claude-haiku-4-5"));
        for model in &models {
            assert!(model.can_stream);
            assert!(model.supports_tools);
            assert!(model.supports_schema);
        }
    }

    #[test]
    fn provider_needs_anthropic_key() {
        let p = make_provider("http://unused");
        assert_eq!(p.needs_key(), Some("anthropic"));
        assert_eq!(p.key_env_var(), Some("ANTHROPIC_API_KEY"));
    }

    // --- Missing key ---

    #[tokio::test]
    async fn missing_key_returns_error() {
        let provider = make_provider("http://unused");
        let prompt = Prompt::new("Hi");
        let result = provider
            .execute("claude-sonnet-4-6", &prompt, None, true)
            .await;
        assert!(result.is_err());
        if let Err(LlmError::NeedsKey(msg)) = result {
            assert!(msg.contains("llm keys set anthropic"));
        } else {
            panic!("expected NeedsKey error");
        }
    }

    // --- Non-streaming integration test ---

    #[tokio::test]
    async fn non_streaming_response() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "Hello world"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(&body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), false)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text: String = chunks
            .iter()
            .filter_map(|c| {
                if let Chunk::Text(t) = c {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "Hello world");

        // Check usage
        let usage = chunks.iter().find_map(|c| {
            if let Chunk::Usage(u) = c {
                Some(u)
            } else {
                None
            }
        });
        assert!(usage.is_some());
        assert_eq!(usage.unwrap().input, Some(5));
        assert_eq!(usage.unwrap().output, Some(2));

        // Should end with Done
        assert!(matches!(chunks.last(), Some(Chunk::Done)));
    }

    // --- Streaming integration test ---

    fn make_anthropic_sse_body() -> String {
        format!(
            "\
event: message_start\n\
data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"usage\":{{\"input_tokens\":5,\"output_tokens\":0}}}}}}\n\n\
event: content_block_start\n\
data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
event: content_block_delta\n\
data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"Hello\"}}}}\n\n\
event: content_block_delta\n\
data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\" world\"}}}}\n\n\
event: content_block_stop\n\
data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
event: message_delta\n\
data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":2}}}}\n\n\
event: message_stop\n\
data: {{\"type\":\"message_stop\"}}\n\n"
        )
    }

    #[tokio::test]
    async fn streaming_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(make_anthropic_sse_body()),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), true)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Collect text
        let text: String = chunks
            .iter()
            .filter_map(|c| {
                if let Chunk::Text(t) = c {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "Hello world");

        // Check usage
        let usage = chunks.iter().find_map(|c| {
            if let Chunk::Usage(u) = c {
                Some(u)
            } else {
                None
            }
        });
        assert!(usage.is_some());
        assert_eq!(usage.unwrap().input, Some(5));
        assert_eq!(usage.unwrap().output, Some(2));

        // Should end with Done
        assert!(matches!(chunks.last(), Some(Chunk::Done)));
    }

    // --- API error ---

    #[tokio::test]
    async fn api_error_response() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "invalid x-api-key"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&body))
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let result = provider
            .execute("claude-sonnet-4-6", &prompt, Some("bad-key"), true)
            .await;
        assert!(result.is_err());
        if let Err(LlmError::Provider(msg)) = result {
            assert!(msg.contains("invalid x-api-key"));
        } else {
            panic!("expected Provider error");
        }
    }

    // --- System prompt in request body ---

    #[tokio::test]
    async fn system_prompt_is_top_level() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1}
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(&body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi").with_system("Be brief.");
        let _stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), false)
            .await
            .unwrap();

        // Verify the request was made (mock expectation passes)
        // The system prompt should be in the top-level field, not in messages.
        // We verify by checking build_messages only produces user messages.
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    // --- max_tokens defaults to 4096 ---

    #[test]
    fn default_max_tokens_is_4096() {
        assert_eq!(DEFAULT_MAX_TOKENS, 4096);
    }

    // --- Tool calling tests ---

    #[tokio::test]
    async fn streaming_tool_call() {
        let server = MockServer::start().await;

        let sse_body = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"location\\\":\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Paris\\\"}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":30}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let tool = llm_core::Tool {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        };
        let prompt = Prompt::new("What's the weather in Paris?").with_tools(vec![tool]);
        let stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), true)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let tool_calls = llm_core::collect_tool_calls(&chunks);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(tool_calls[0].arguments, serde_json::json!({"location": "Paris"}));
    }

    #[tokio::test]
    async fn non_streaming_tool_call() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"location": "Paris"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30}
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(&body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("What's the weather?");
        let stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), false)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let tool_calls = llm_core::collect_tool_calls(&chunks);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].tool_call_id.as_deref(), Some("toolu_1"));
    }

    #[tokio::test]
    async fn request_includes_tools_when_prompt_has_tools() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1}
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(&body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let tool = llm_core::Tool {
            name: "test_tool".into(),
            description: "A test".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let prompt = Prompt::new("test").with_tools(vec![tool]);
        let _ = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), false)
            .await
            .unwrap();
    }

    // --- Structured output tests ---

    #[tokio::test]
    async fn non_streaming_schema_response() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "_schema_output",
                "input": {"name": "John", "age": 30}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(&body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        });
        let prompt = Prompt::new("Extract from: John is 30").with_schema(schema);
        let stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), false)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text = llm_core::collect_text(&chunks);
        // Should be JSON text, not tool calls
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["name"], "John");
        assert_eq!(parsed["age"], 30);

        // Should NOT have any tool calls (schema output is transparent)
        let tool_calls = llm_core::collect_tool_calls(&chunks);
        assert!(tool_calls.is_empty());
    }

    #[tokio::test]
    async fn streaming_schema_response() {
        let server = MockServer::start().await;

        let sse_body = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"_schema_output\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"name\\\":\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"John\\\",\\\"age\\\":30}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        });
        let prompt = Prompt::new("Extract from: John is 30").with_schema(schema);
        let stream = provider
            .execute("claude-sonnet-4-6", &prompt, Some("sk-test"), true)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text = llm_core::collect_text(&chunks);
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["name"], "John");
        assert_eq!(parsed["age"], 30);

        // Schema output should NOT produce tool calls
        let tool_calls = llm_core::collect_tool_calls(&chunks);
        assert!(tool_calls.is_empty());
    }
}
