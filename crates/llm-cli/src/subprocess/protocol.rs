use llm_core::{Chunk, Prompt, ToolCall, Usage};
use serde::{Deserialize, Serialize};

/// Wire protocol chunk for subprocess provider communication (JSONL).
/// Maps to/from llm_core::Chunk but has its own serde implementation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProtocolChunk {
    Text { content: String },
    ToolCallStart { name: String, id: Option<String> },
    ToolCallDelta { content: String },
    Usage { input: Option<u64>, output: Option<u64> },
    Done {},
}

/// Request sent to a subprocess provider on stdin.
#[derive(Serialize, Deserialize, Debug)]
pub struct ProviderRequest {
    pub model: String,
    pub prompt: Prompt,
    pub key: Option<String>,
    pub stream: bool,
}

/// Non-streaming response from a subprocess provider.
#[derive(Serialize, Deserialize, Debug)]
pub struct ProviderResponse {
    pub text: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<ResponseUsage>,
}

/// Usage in a provider response (always concrete, not Option fields).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResponseUsage {
    pub input: u64,
    pub output: u64,
}

impl From<ProtocolChunk> for Chunk {
    fn from(pc: ProtocolChunk) -> Self {
        match pc {
            ProtocolChunk::Text { content } => Chunk::Text(content),
            ProtocolChunk::ToolCallStart { name, id } => Chunk::ToolCallStart { name, id },
            ProtocolChunk::ToolCallDelta { content } => Chunk::ToolCallDelta { content },
            ProtocolChunk::Usage { input, output } => Chunk::Usage(Usage {
                input,
                output,
                details: None,
            }),
            ProtocolChunk::Done {} => Chunk::Done,
        }
    }
}

impl From<&Chunk> for ProtocolChunk {
    fn from(chunk: &Chunk) -> Self {
        match chunk {
            Chunk::Text(content) => ProtocolChunk::Text {
                content: content.clone(),
            },
            Chunk::ToolCallStart { name, id } => ProtocolChunk::ToolCallStart {
                name: name.clone(),
                id: id.clone(),
            },
            Chunk::ToolCallDelta { content } => ProtocolChunk::ToolCallDelta {
                content: content.clone(),
            },
            Chunk::Usage(usage) => ProtocolChunk::Usage {
                input: usage.input,
                output: usage.output,
            },
            Chunk::Done => ProtocolChunk::Done {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_serializes_correctly() {
        let chunk = ProtocolChunk::Text {
            content: "hello".into(),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert_eq!(json, r#"{"type":"text","content":"hello"}"#);
    }

    #[test]
    fn tool_call_start_serializes_with_name_and_id() {
        let chunk = ProtocolChunk::ToolCallStart {
            name: "search".into(),
            id: Some("tc_1".into()),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "tool_call_start");
        assert_eq!(parsed["name"], "search");
        assert_eq!(parsed["id"], "tc_1");
    }

    #[test]
    fn tool_call_delta_serializes() {
        let chunk = ProtocolChunk::ToolCallDelta {
            content: r#"{"query":"rust"}"#.into(),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "tool_call_delta");
    }

    #[test]
    fn usage_serializes_with_input_output() {
        let chunk = ProtocolChunk::Usage {
            input: Some(10),
            output: Some(20),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "usage");
        assert_eq!(parsed["input"], 10);
        assert_eq!(parsed["output"], 20);
    }

    #[test]
    fn done_serializes_correctly() {
        let chunk = ProtocolChunk::Done {};
        let json = serde_json::to_string(&chunk).unwrap();
        assert_eq!(json, r#"{"type":"done"}"#);
    }

    #[test]
    fn roundtrip_all_variants() {
        let variants = vec![
            ProtocolChunk::Text {
                content: "hi".into(),
            },
            ProtocolChunk::ToolCallStart {
                name: "s".into(),
                id: None,
            },
            ProtocolChunk::ToolCallDelta {
                content: "{}".into(),
            },
            ProtocolChunk::Usage {
                input: Some(1),
                output: None,
            },
            ProtocolChunk::Done {},
        ];
        for chunk in variants {
            let json = serde_json::to_string(&chunk).unwrap();
            let restored: ProtocolChunk = serde_json::from_str(&json).unwrap();
            assert_eq!(chunk, restored);
        }
    }

    #[test]
    fn protocol_chunk_to_chunk_conversion() {
        assert!(matches!(
            Chunk::from(ProtocolChunk::Text { content: "hi".into() }),
            Chunk::Text(t) if t == "hi"
        ));
        assert!(matches!(
            Chunk::from(ProtocolChunk::ToolCallStart { name: "s".into(), id: Some("1".into()) }),
            Chunk::ToolCallStart { name, id } if name == "s" && id == Some("1".into())
        ));
        assert!(matches!(
            Chunk::from(ProtocolChunk::ToolCallDelta { content: "x".into() }),
            Chunk::ToolCallDelta { content } if content == "x"
        ));
        assert!(matches!(
            Chunk::from(ProtocolChunk::Usage { input: Some(5), output: Some(10) }),
            Chunk::Usage(u) if u.input == Some(5) && u.output == Some(10)
        ));
        assert!(matches!(
            Chunk::from(ProtocolChunk::Done {}),
            Chunk::Done
        ));
    }

    #[test]
    fn chunk_to_protocol_chunk_conversion() {
        assert_eq!(
            ProtocolChunk::from(&Chunk::Text("hi".into())),
            ProtocolChunk::Text { content: "hi".into() }
        );
        assert_eq!(
            ProtocolChunk::from(&Chunk::Done),
            ProtocolChunk::Done {}
        );
        assert_eq!(
            ProtocolChunk::from(&Chunk::Usage(Usage {
                input: Some(3),
                output: Some(7),
                details: None,
            })),
            ProtocolChunk::Usage { input: Some(3), output: Some(7) }
        );
    }

    #[test]
    fn provider_request_serializes() {
        let req = ProviderRequest {
            model: "llama3".into(),
            prompt: Prompt::new("Hello"),
            key: Some("sk-test".into()),
            stream: true,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "llama3");
        assert_eq!(json["stream"], true);
        assert_eq!(json["key"], "sk-test");
    }

    #[test]
    fn provider_response_deserializes() {
        let json = r#"{"text":"Hello there","tool_calls":[],"usage":{"input":10,"output":20}}"#;
        let resp: ProviderResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "Hello there");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.usage.as_ref().unwrap().input, 10);
        assert_eq!(resp.usage.as_ref().unwrap().output, 20);
    }
}
