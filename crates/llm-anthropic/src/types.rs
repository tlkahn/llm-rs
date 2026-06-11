use serde::{Deserialize, Serialize};

// --- Request types ---

#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u64,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// Content can be a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// --- Non-streaming response ---

#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: UsageResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    // tool_use fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    // tool_result fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    // image fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ImageSource>,
}

impl ContentBlock {
    /// Create an image content block with base64-encoded data.
    ///
    /// Produces the Anthropic wire format:
    /// ```json
    /// {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    /// ```
    #[must_use]
    pub fn image_base64(media_type: &str, data: &str) -> Self {
        Self {
            block_type: "image".into(),
            text: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            source: Some(ImageSource {
                source_type: "base64".into(),
                media_type: Some(media_type.to_string()),
                data: Some(data.to_string()),
                url: None,
            }),
        }
    }

    /// Create an image content block with a URL reference.
    ///
    /// Produces the Anthropic wire format:
    /// ```json
    /// {"type": "image", "source": {"type": "url", "url": "https://..."}}
    /// ```
    #[must_use]
    pub fn image_url(url: &str) -> Self {
        Self {
            block_type: "image".into(),
            text: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            source: Some(ImageSource {
                source_type: "url".into(),
                media_type: None,
                data: None,
                url: Some(url.to_string()),
            }),
        }
    }
}

/// Image source for Anthropic's image content blocks.
///
/// Represents either a base64-encoded image or a URL reference.
/// See: <https://docs.anthropic.com/en/docs/build-with-claude/vision>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageSource {
    /// "base64" or "url"
    #[serde(rename = "type")]
    pub source_type: String,
    /// MIME type, e.g. "image/png". Required for base64, absent for URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Base64-encoded image data. Present when source_type is "base64".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Image URL. Present when source_type is "url".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageResponse {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// --- Streaming events ---

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessageStartBody,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: Option<DeltaUsage>,
    },
    MessageStop,
    Ping,
}

#[derive(Debug, Deserialize)]
pub struct MessageStartBody {
    pub id: String,
    pub model: String,
    pub role: String,
    #[serde(default)]
    pub usage: Option<UsageResponse>,
}

#[derive(Debug, Deserialize)]
pub struct ContentDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaUsage {
    pub output_tokens: u64,
}

// --- Error ---

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MessagesRequest tests ---

    #[test]
    fn messages_request_minimal() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".into(),
            max_tokens: 4096,
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Text("Hello".into()),
            }],
            system: None,
            stream: None,
            temperature: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-sonnet-4-6");
        assert_eq!(json["max_tokens"], 4096);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "Hello");
        // None fields should be absent
        assert!(json.get("system").is_none());
        assert!(json.get("stream").is_none());
        assert!(json.get("temperature").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn messages_request_with_system_and_stream() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".into(),
            max_tokens: 8192,
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: Some("Be brief.".into()),
            stream: Some(true),
            temperature: Some(0.7),
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"], "Be brief.");
        assert_eq!(json["stream"], true);
        assert_eq!(json["temperature"], 0.7);
        assert_eq!(json["max_tokens"], 8192);
    }

    #[test]
    fn max_tokens_always_serialized() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".into(),
            max_tokens: 1024,
            messages: vec![],
            system: None,
            stream: None,
            temperature: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("max_tokens").is_some());
        assert_eq!(json["max_tokens"], 1024);
    }

    // --- MessagesResponse (non-streaming) ---

    #[test]
    fn messages_response_deserialize() {
        let json = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "Hello!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });
        let resp: MessagesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msg_123");
        assert_eq!(resp.model, "claude-sonnet-4-6");
        assert_eq!(resp.content[0].text.as_deref(), Some("Hello!"));
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
    }

    // --- StreamEvent variants ---

    #[test]
    fn stream_event_message_start() {
        let json = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 12, "output_tokens": 0}
            }
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        if let StreamEvent::MessageStart { message } = event {
            assert_eq!(message.id, "msg_1");
            assert_eq!(message.usage.unwrap().input_tokens, 12);
        } else {
            panic!("expected MessageStart");
        }
    }

    #[test]
    fn stream_event_content_block_start() {
        let json = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, StreamEvent::ContentBlockStart { index: 0, .. }));
    }

    #[test]
    fn stream_event_content_block_delta() {
        let json = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        if let StreamEvent::ContentBlockDelta { delta, .. } = event {
            assert_eq!(delta.text.as_deref(), Some("Hello"));
        } else {
            panic!("expected ContentBlockDelta");
        }
    }

    #[test]
    fn stream_event_content_block_stop() {
        let json = serde_json::json!({
            "type": "content_block_stop",
            "index": 0
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, StreamEvent::ContentBlockStop { index: 0 }));
    }

    #[test]
    fn stream_event_message_delta() {
        let json = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        if let StreamEvent::MessageDelta { delta, usage } = event {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(usage.unwrap().output_tokens, 15);
        } else {
            panic!("expected MessageDelta");
        }
    }

    #[test]
    fn stream_event_message_stop() {
        let json = serde_json::json!({"type": "message_stop"});
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStop));
    }

    #[test]
    fn stream_event_ping() {
        let json = serde_json::json!({"type": "ping"});
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, StreamEvent::Ping));
    }

    // --- ErrorResponse ---

    #[test]
    fn error_response_deserialize() {
        let json = serde_json::json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "invalid x-api-key"
            }
        });
        let err: ErrorResponse = serde_json::from_value(json).unwrap();
        assert_eq!(err.error.message, "invalid x-api-key");
        assert_eq!(err.error.error_type, "authentication_error");
    }

    // --- Tool calling types ---

    #[test]
    fn messages_request_with_tools_serializes() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6".into(),
            max_tokens: 4096,
            messages: vec![],
            system: None,
            stream: None,
            temperature: None,
            tools: Some(vec![AnthropicTool {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }),
            }]),
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["name"], "get_weather");
        assert_eq!(json["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn content_block_tool_use_deserializes() {
        let json = serde_json::json!({
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"location": "Paris"}
        });
        let block: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block.block_type, "tool_use");
        assert_eq!(block.id.as_deref(), Some("toolu_1"));
        assert_eq!(block.name.as_deref(), Some("get_weather"));
        assert_eq!(block.input.as_ref().unwrap()["location"], "Paris");
    }

    #[test]
    fn content_delta_input_json_delta_deserializes() {
        let json = serde_json::json!({
            "type": "input_json_delta",
            "partial_json": "{\"location\":"
        });
        let delta: ContentDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta.delta_type, "input_json_delta");
        assert_eq!(delta.partial_json.as_deref(), Some("{\"location\":"));
    }

    #[test]
    fn content_block_tool_result_serializes() {
        let block = ContentBlock {
            block_type: "tool_result".into(),
            text: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: Some("toolu_1".into()),
            content: Some("Sunny, 22C".into()),
            is_error: None,
            source: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "toolu_1");
        assert_eq!(json["content"], "Sunny, 22C");
        assert!(json.get("text").is_none());
    }

    #[test]
    fn stream_event_content_block_start_tool_use() {
        let json = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        if let StreamEvent::ContentBlockStart { content_block, .. } = event {
            assert_eq!(content_block.block_type, "tool_use");
            assert_eq!(content_block.name.as_deref(), Some("get_weather"));
            assert_eq!(content_block.id.as_deref(), Some("toolu_1"));
        } else {
            panic!("expected ContentBlockStart");
        }
    }

    // --- Image content block tests ---

    #[test]
    fn image_block_base64_serializes_correctly() {
        let block = ContentBlock::image_base64("image/png", "iVBORw0KGgo=");
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "base64");
        assert_eq!(json["source"]["media_type"], "image/png");
        assert_eq!(json["source"]["data"], "iVBORw0KGgo=");
        // Optional fields should be absent
        assert!(json["source"].get("url").is_none());
        assert!(json.get("text").is_none());
        assert!(json.get("id").is_none());
        assert!(json.get("name").is_none());
        assert!(json.get("input").is_none());
        assert!(json.get("tool_use_id").is_none());
        assert!(json.get("content").is_none());
        assert!(json.get("is_error").is_none());
    }

    #[test]
    fn image_block_url_serializes_correctly() {
        let block = ContentBlock::image_url("https://example.com/cat.jpg");
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "url");
        assert_eq!(json["source"]["url"], "https://example.com/cat.jpg");
        // base64-specific fields should be absent
        assert!(json["source"].get("media_type").is_none());
        assert!(json["source"].get("data").is_none());
    }

    #[test]
    fn image_block_base64_deserializes() {
        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "abc123=="
            }
        });
        let block: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block.block_type, "image");
        let source = block.source.unwrap();
        assert_eq!(source.source_type, "base64");
        assert_eq!(source.media_type.as_deref(), Some("image/jpeg"));
        assert_eq!(source.data.as_deref(), Some("abc123=="));
        assert_eq!(source.url, None);
    }

    #[test]
    fn image_block_url_deserializes() {
        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/img.png"
            }
        });
        let block: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block.block_type, "image");
        let source = block.source.unwrap();
        assert_eq!(source.source_type, "url");
        assert_eq!(source.url.as_deref(), Some("https://example.com/img.png"));
        assert_eq!(source.media_type, None);
        assert_eq!(source.data, None);
    }

    #[test]
    fn image_source_absent_when_not_image_block() {
        let json = serde_json::json!({
            "type": "text",
            "text": "Hello"
        });
        let block: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block.block_type, "text");
        assert!(block.source.is_none());
    }
}
