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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
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
    pub tool_calls: Option<Vec<MessageToolCall>>,
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
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

// --- Tool calling types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: MessageToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeltaToolCall {
    pub index: u32,
    pub id: Option<String>,
    pub function: Option<DeltaFunction>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeltaFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// --- Structured output types ---

#[derive(Debug, Clone, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: JsonSchemaFormat,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub strict: bool,
    pub schema: serde_json::Value,
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
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "Hello");
        assert_eq!(json["stream"], true);
        // None fields should be absent (skip_serializing_if)
        assert!(json.get("temperature").is_none());
        assert!(json.get("tools").is_none());
        assert!(json.get("response_format").is_none());
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
            tools: None,
            tool_choice: None,
            response_format: None,
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
            tools: None,
            tool_choice: None,
            response_format: None,
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

    // --- Tool calling types ---

    #[test]
    fn chat_request_with_tools_serializes() {
        let req = ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            stream: None,
            stream_options: None,
            temperature: None,
            max_tokens: None,
            tools: Some(vec![ChatTool {
                tool_type: "function".into(),
                function: ChatToolFunction {
                    name: "get_weather".into(),
                    description: "Get weather".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }),
                },
            }]),
            tool_choice: None,
            response_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn delta_with_tool_calls_deserializes() {
        let json = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": ""}
            }]
        });
        let delta: Delta = serde_json::from_value(json).unwrap();
        let tc = delta.tool_calls.unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].index, 0);
        assert_eq!(tc[0].id.as_deref(), Some("call_1"));
        assert_eq!(tc[0].function.as_ref().unwrap().name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn delta_tool_call_first_chunk_has_name_and_id() {
        let json = serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": ""}
            }]
        });
        let delta: Delta = serde_json::from_value(json).unwrap();
        let tc = &delta.tool_calls.unwrap()[0];
        assert_eq!(tc.id.as_deref(), Some("call_1"));
        assert_eq!(tc.function.as_ref().unwrap().name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn delta_tool_call_subsequent_has_only_arguments() {
        let json = serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": "{\"location\":"}
            }]
        });
        let delta: Delta = serde_json::from_value(json).unwrap();
        let tc = &delta.tool_calls.unwrap()[0];
        assert_eq!(tc.id, None);
        assert_eq!(
            tc.function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"location\":")
        );
    }

    #[test]
    fn message_tool_call_deserializes() {
        let json = serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris\"}"
            }
        });
        let tc: MessageToolCall = serde_json::from_value(json).unwrap();
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.call_type, "function");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, "{\"location\":\"Paris\"}");
    }
}
