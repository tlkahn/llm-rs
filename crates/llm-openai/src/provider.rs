use async_trait::async_trait;
use futures::StreamExt;
use llm_core::stream::{Chunk, ResponseStream};
use llm_core::types::{ModelInfo, Prompt, Usage};
use llm_core::{LlmError, Provider, Result};
use reqwest::Client;

use crate::messages::build_messages;
use crate::sse::SseParser;
use crate::types::{
    ChatRequest, ChatResponse, ErrorResponse, StreamOptions,
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

#[async_trait]
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

        let mut request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: Some(stream),
            stream_options: None,
            temperature: prompt.options.get("temperature").and_then(|v| v.as_f64()),
            max_tokens: prompt.options.get("max_tokens").and_then(|v| v.as_u64()),
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
            let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Chunk, LlmError>>(32);

            tokio::spawn(async move {
                let mut parser = SseParser::new();
                let mut byte_stream = std::pin::pin!(byte_stream);
                while let Some(result) = byte_stream.next().await {
                    match result {
                        Ok(bytes) => {
                            parser.feed(&bytes);
                            while let Some(event) = parser.next_event() {
                                // Map StreamChunk → Chunk(s)
                                for choice in &event.choices {
                                    if let Some(content) = &choice.delta.content {
                                        if !content.is_empty() {
                                            let _ = tx.send(Ok(Chunk::Text(content.clone()))).await;
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
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            Ok(Box::pin(stream))
        } else {
            // Non-streaming: parse full JSON response
            let body = response
                .text()
                .await
                .map_err(|e| LlmError::Provider(e.to_string()))?;
            let resp: ChatResponse =
                serde_json::from_str(&body).map_err(|e| LlmError::Provider(e.to_string()))?;

            let mut chunks: Vec<std::result::Result<Chunk, LlmError>> = Vec::new();

            if let Some(choice) = resp.choices.first() {
                if let Some(msg) = &choice.message {
                    if let Some(content) = &msg.content {
                        chunks.push(Ok(Chunk::Text(content.clone())));
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
        if let Err(LlmError::Provider(msg)) = result {
            assert!(msg.contains("Incorrect API key"));
        } else {
            panic!("expected Provider error");
        }
    }
}
