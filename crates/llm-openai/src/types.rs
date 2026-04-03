use serde::{Deserialize, Serialize};

// --- Request types ---

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

// --- Non-streaming response ---

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<UsageResponse>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Option<Message>,
    pub finish_reason: Option<String>,
}

// --- Streaming response ---

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<UsageResponse>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}

// --- Shared ---

#[derive(Debug, Deserialize)]
pub struct UsageResponse {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// --- Error ---

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ChatRequest tests ---

    #[test]
    fn chat_request_minimal() {
        let req = ChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![Message {
                role: "user".into(),
                content: Some("Hello".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: Some(true),
            stream_options: None,
            temperature: None,
            max_tokens: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "Hello");
        assert_eq!(json["stream"], true);
        // None fields should be absent (skip_serializing_if)
        assert!(json.get("temperature").is_none());
    }

    #[test]
    fn chat_request_with_options() {
        let req = ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            stream: Some(false),
            stream_options: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["temperature"], 0.7);
        assert_eq!(json["max_tokens"], 100);
    }

    #[test]
    fn chat_request_with_stream_options() {
        let req = ChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![],
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: None,
            max_tokens: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    // --- Message tests ---

    #[test]
    fn message_system() {
        let msg = Message {
            role: "system".into(),
            content: Some("You are helpful.".into()),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "You are helpful.");
    }

    // --- ChatResponse (non-streaming) ---

    #[test]
    fn chat_response_deserialize() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        });
        let resp: ChatResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.model, "gpt-4o-mini");
        assert_eq!(resp.choices[0].message.as_ref().unwrap().content.as_deref(), Some("Hello!"));
        assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 5);
    }

    // --- StreamChunk (streaming) ---

    #[test]
    fn stream_chunk_deserialize_text_delta() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hi"
                },
                "finish_reason": null
            }]
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
        assert_eq!(chunk.choices[0].finish_reason, None);
    }

    #[test]
    fn stream_chunk_deserialize_role_only() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": null
            }]
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn stream_chunk_deserialize_finish() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn stream_chunk_with_usage() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert!(chunk.choices.is_empty());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    // --- UsageResponse ---

    #[test]
    fn usage_response_deserialize() {
        let json = serde_json::json!({
            "prompt_tokens": 42,
            "completion_tokens": 13,
            "total_tokens": 55
        });
        let usage: UsageResponse = serde_json::from_value(json).unwrap();
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 13);
        assert_eq!(usage.total_tokens, 55);
    }

    // --- ErrorResponse ---

    #[test]
    fn error_response_deserialize() {
        let json = serde_json::json!({
            "error": {
                "message": "Incorrect API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        let err: ErrorResponse = serde_json::from_value(json).unwrap();
        assert_eq!(err.error.message, "Incorrect API key");
        assert_eq!(err.error.r#type, "invalid_request_error");
        assert_eq!(err.error.code.as_deref(), Some("invalid_api_key"));
    }
}
