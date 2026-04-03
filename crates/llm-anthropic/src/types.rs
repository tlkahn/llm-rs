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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
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
}
