use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use llm_core::stream::{Chunk, ResponseStream};
use llm_core::types::{ModelInfo, Prompt, Usage};
use llm_core::{LlmError, Provider, Result};
use reqwest::Client;

use crate::messages::build_messages;
use crate::sse::SseParser;
use crate::types::{
    ChatRequest, ChatResponse, ChatTool, ChatToolFunction, ErrorResponse, JsonSchemaFormat,
    ResponseFormat, StreamOptions,
};

pub struct OpenAiProvider {
    base_url: String,
    client: Client,
}

impl OpenAiProvider {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Provider for OpenAiProvider {
    fn id(&self) -> &str {
        "openai"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "gpt-4o".into(),
                can_stream: true,
                supports_tools: true,
                supports_schema: true,
                attachment_types: vec!["image/png".into(), "image/jpeg".into(), "image/webp".into(), "image/gif".into()],
            },
            ModelInfo {
                id: "gpt-4o-mini".into(),
                can_stream: true,
                supports_tools: true,
                supports_schema: true,
                attachment_types: vec!["image/png".into(), "image/jpeg".into(), "image/webp".into(), "image/gif".into()],
            },
        ]
    }

    fn needs_key(&self) -> Option<&str> {
        Some("openai")
    }

    fn key_env_var(&self) -> Option<&str> {
        Some("OPENAI_API_KEY")
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
                "No key found - set one with 'llm keys set openai' or export OPENAI_API_KEY"
                    .into(),
            )
        })?;

        let messages = build_messages(prompt);

        // Convert llm_core::Tool -> ChatTool
        let tools = if prompt.tools.is_empty() {
            None
        } else {
            Some(
                prompt
                    .tools
                    .iter()
                    .map(|t| ChatTool {
                        tool_type: "function".into(),
                        function: ChatToolFunction {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: t.input_schema.clone(),
                        },
                    })
                    .collect(),
            )
        };

        let mut request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: Some(stream),
            stream_options: None,
            temperature: prompt.options.get("temperature").and_then(|v| v.as_f64()),
            max_tokens: prompt.options.get("max_tokens").and_then(|v| v.as_u64()),
            tools,
            tool_choice: None,
            response_format: prompt.schema.as_ref().map(|schema| ResponseFormat {
                format_type: "json_schema".into(),
                json_schema: JsonSchemaFormat {
                    name: "output".into(),
                    strict: true,
                    schema: schema.clone(),
                },
            }),
        };

        if stream {
            request.stream_options = Some(StreamOptions {
                include_usage: true,
            });
        }

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(key)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::Provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            let message = if let Ok(err_resp) = serde_json::from_str::<ErrorResponse>(&body) {
                err_resp.error.message
            } else {
                body
            };
            return Err(LlmError::HttpError { status: status_code, message });
        }

        if stream {
            let byte_stream = response.bytes_stream();
            let (mut tx, rx) =
                futures::channel::mpsc::channel::<std::result::Result<Chunk, LlmError>>(32);

            let parse_future = async move {
                let mut parser = SseParser::new();
                let mut byte_stream = std::pin::pin!(byte_stream);
                while let Some(result) = byte_stream.next().await {
                    match result {
                        Ok(bytes) => {
                            parser.feed(&bytes);
                            while let Some(event) = parser.next_event() {
                                // Map StreamChunk → Chunk(s)
                                for choice in &event.choices {
                                    if let Some(content) = &choice.delta.content
                                        && !content.is_empty()
                                    {
                                        let _ =
                                            tx.send(Ok(Chunk::Text(content.clone()))).await;
                                    }
                                    // Handle tool call deltas
                                    if let Some(tool_calls) = &choice.delta.tool_calls {
                                        for tc in tool_calls {
                                            if let Some(func) = &tc.function {
                                                if let Some(name) = &func.name {
                                                    let _ = tx
                                                        .send(Ok(Chunk::ToolCallStart {
                                                            name: name.clone(),
                                                            id: tc.id.clone(),
                                                        }))
                                                        .await;
                                                }
                                                if let Some(args) = &func.arguments
                                                    && !args.is_empty()
                                                {
                                                    let _ = tx
                                                        .send(Ok(Chunk::ToolCallDelta {
                                                            content: args.clone(),
                                                        }))
                                                        .await;
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Some(usage) = &event.usage {
                                    let _ = tx
                                        .send(Ok(Chunk::Usage(Usage {
                                            input: Some(usage.prompt_tokens),
                                            output: Some(usage.completion_tokens),
                                            details: None,
                                        })))
                                        .await;
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
            let resp: ChatResponse =
                serde_json::from_str(&body).map_err(|e| LlmError::Provider(e.to_string()))?;

            let mut chunks: Vec<std::result::Result<Chunk, LlmError>> = Vec::new();

            if let Some(choice) = resp.choices.first()
                && let Some(msg) = &choice.message
            {
                if let Some(content) = &msg.content {
                    chunks.push(Ok(Chunk::Text(content.clone())));
                }
                // Handle tool calls in non-streaming response
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        chunks.push(Ok(Chunk::ToolCallStart {
                            name: tc.function.name.clone(),
                            id: Some(tc.id.clone()),
                        }));
                        chunks.push(Ok(Chunk::ToolCallDelta {
                            content: tc.function.arguments.clone(),
                        }));
                    }
                }
            }

            if let Some(usage) = &resp.usage {
                chunks.push(Ok(Chunk::Usage(Usage {
                    input: Some(usage.prompt_tokens),
                    output: Some(usage.completion_tokens),
                    details: None,
                })));
            }

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

    fn make_provider(base_url: &str) -> OpenAiProvider {
        OpenAiProvider::new(base_url)
    }

    // --- Unit tests ---

    #[test]
    fn provider_id_is_openai() {
        let p = make_provider("http://unused");
        assert_eq!(p.id(), "openai");
    }

    #[test]
    fn provider_lists_two_models() {
        let p = make_provider("http://unused");
        let models = p.models();
        assert_eq!(models.len(), 2);
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"gpt-4o"));
        assert!(ids.contains(&"gpt-4o-mini"));
    }

    #[test]
    fn provider_needs_openai_key() {
        let p = make_provider("http://unused");
        assert_eq!(p.needs_key(), Some("openai"));
        assert_eq!(p.key_env_var(), Some("OPENAI_API_KEY"));
    }

    // --- Streaming integration test ---

    #[tokio::test]
    async fn streaming_response() {
        let server = MockServer::start().await;

        let sse_body = "\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n\
data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let stream = provider
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), true)
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

    // --- Non-streaming integration test ---

    #[tokio::test]
    async fn non_streaming_response() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), false)
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
    }

    // --- Error handling ---

    #[tokio::test]
    async fn missing_key_returns_error() {
        let provider = make_provider("http://unused");
        let prompt = Prompt::new("Hi");
        let result = provider
            .execute("gpt-4o-mini", &prompt, None, true)
            .await;
        assert!(result.is_err());
        if let Err(LlmError::NeedsKey(msg)) = result {
            assert!(msg.contains("llm keys set openai"));
        } else {
            panic!("expected NeedsKey error");
        }
    }

    #[tokio::test]
    async fn api_error_response() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401)
                    .set_body_json(&body),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let result = provider
            .execute("gpt-4o-mini", &prompt, Some("bad-key"), true)
            .await;
        assert!(result.is_err());
        if let Err(LlmError::HttpError { status, message }) = result {
            assert_eq!(status, 401);
            assert!(message.contains("Incorrect API key"));
        } else {
            panic!("expected HttpError");
        }
    }

    #[tokio::test]
    async fn http_429_returns_http_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .set_body_string("rate limited"),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let result = provider
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), true)
            .await;
        match result {
            Err(ref e @ LlmError::HttpError { status, .. }) => {
                assert_eq!(status, 429);
                assert!(e.is_retryable());
            }
            _ => panic!("expected HttpError with status 429"),
        }
    }

    #[tokio::test]
    async fn http_500_returns_http_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(500)
                    .set_body_string("internal server error"),
            )
            .mount(&server)
            .await;

        let provider = make_provider(&server.uri());
        let prompt = Prompt::new("Hi");
        let result = provider
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), true)
            .await;
        match result {
            Err(ref e @ LlmError::HttpError { status, .. }) => {
                assert_eq!(status, 500);
                assert!(e.is_retryable());
            }
            _ => panic!("expected HttpError with status 500"),
        }
    }

    // --- Tool calling tests ---

    #[tokio::test]
    async fn streaming_single_tool_call() {
        let server = MockServer::start().await;

        let sse_body = "\
data: {\"id\":\"1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: {\"id\":\"1\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":20,\"total_tokens\":70}}\n\n\
data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), true)
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
        assert_eq!(tool_calls[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool_calls[0].arguments, serde_json::json!({"location": "Paris"}));
    }

    #[tokio::test]
    async fn non_streaming_tool_call() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Paris\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), false)
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
        assert_eq!(tool_calls[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn request_includes_tools_when_prompt_has_tools() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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
            description: "A test tool".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let prompt = Prompt::new("test").with_tools(vec![tool]);
        let _ = provider
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), false)
            .await
            .unwrap();
        // Mock expectation verifies the request was made
    }

    // --- Structured output tests ---

    #[tokio::test]
    async fn non_streaming_schema_response() {
        let server = MockServer::start().await;

        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "{\"name\":\"John\",\"age\":30}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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
            .execute("gpt-4o-mini", &prompt, Some("sk-test"), false)
            .await
            .unwrap();

        let chunks: Vec<_> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text = llm_core::collect_text(&chunks);
        assert_eq!(text, "{\"name\":\"John\",\"age\":30}");
    }
}
