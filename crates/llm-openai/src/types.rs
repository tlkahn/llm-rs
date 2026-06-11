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
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MessageToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content can be a plain string or an array of content parts.
/// Text variant listed first for backward-compatible deserialization:
/// `"content": "Hello"` deserializes as `Text(String)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract text content, returning None for multi-part content.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            MessageContent::Parts(_) => None,
        }
    }

    /// Extract all text content from any variant.
    ///
    /// For `Text(s)`, returns the string directly.
    /// For `Parts(parts)`, concatenates all `ContentPart::Text` parts.
    /// Returns an empty string if there is no text.
    pub fn text_content(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => {
                let mut buf = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        buf.push_str(text);
                    }
                }
                buf
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
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
                content: Some(MessageContent::Text("Hello".into())),
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
            content: Some(MessageContent::Text("You are helpful.".into())),
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
        assert_eq!(resp.choices[0].message.as_ref().unwrap().content.as_ref().and_then(|c| c.as_text()), Some("Hello!"));
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

    // --- MessageContent tests ---

    #[test]
    fn message_content_text_serializes_as_string() {
        let content = MessageContent::Text("Hello".into());
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json, serde_json::json!("Hello"));
    }

    #[test]
    fn message_content_parts_serializes_as_array() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Look at this:".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/cat.jpg".into(),
                    detail: Some("high".into()),
                },
            },
        ]);
        let json = serde_json::to_value(&content).unwrap();
        assert!(json.is_array());
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "Look at this:");
        assert_eq!(json[1]["type"], "image_url");
        assert_eq!(json[1]["image_url"]["url"], "https://example.com/cat.jpg");
        assert_eq!(json[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn message_content_text_deserializes_from_string() {
        let json = serde_json::json!("Hello");
        let content: MessageContent = serde_json::from_value(json).unwrap();
        assert_eq!(content.as_text(), Some("Hello"));
    }

    #[test]
    fn message_content_parts_deserializes_from_array() {
        let json = serde_json::json!([
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]);
        let content: MessageContent = serde_json::from_value(json).unwrap();
        match content {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    ContentPart::Text { text } => assert_eq!(text, "Hello"),
                    _ => panic!("expected Text part"),
                }
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "https://example.com/img.png");
                        assert_eq!(image_url.detail, None);
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("expected Parts variant"),
        }
    }

    // --- text_content() tests ---

    #[test]
    fn text_content_from_text_variant() {
        let content = MessageContent::Text("Hello".into());
        assert_eq!(content.text_content(), "Hello");
    }

    #[test]
    fn text_content_from_parts_single_text() {
        let content = MessageContent::Parts(vec![ContentPart::Text {
            text: "Hello".into(),
        }]);
        assert_eq!(content.text_content(), "Hello");
    }

    #[test]
    fn text_content_from_parts_multiple_text() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Hello".into(),
            },
            ContentPart::Text {
                text: " world".into(),
            },
        ]);
        assert_eq!(content.text_content(), "Hello world");
    }

    #[test]
    fn text_content_from_parts_mixed_with_images() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Caption".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".into(),
                    detail: None,
                },
            },
        ]);
        assert_eq!(content.text_content(), "Caption");
    }

    #[test]
    fn text_content_from_parts_empty() {
        let content = MessageContent::Parts(vec![]);
        assert_eq!(content.text_content(), "");
    }

    #[test]
    fn text_content_from_parts_image_only() {
        let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/img.png".into(),
                detail: Some("high".into()),
            },
        }]);
        assert_eq!(content.text_content(), "");
    }

    #[test]
    fn image_url_detail_none_omitted() {
        let part = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/img.png".into(),
                detail: None,
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert!(json["image_url"].get("detail").is_none());
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
