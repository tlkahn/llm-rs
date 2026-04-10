use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub input: Option<u64>,
    pub output: Option<u64>,
    pub details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Usage {
    /// Combine two `Usage` values by summing their fields.
    #[must_use]
    pub fn add(&self, other: &Usage) -> Usage {
        Usage {
            input: match (self.input, other.input) {
                (Some(a), Some(b)) => Some(a + b),
                (a, b) => a.or(b),
            },
            output: match (self.output, other.output) {
                (Some(a), Some(b)) => Some(a + b),
                (a, b) => a.or(b),
            },
            details: None,
        }
    }

    /// Total tokens (input + output), treating None as 0.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.input.unwrap_or(0) + self.output.unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub can_stream: bool,
    pub supports_tools: bool,
    pub supports_schema: bool,
    pub attachment_types: Vec<String>,
}

impl ModelInfo {
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            can_stream: true,
            supports_tools: false,
            supports_schema: false,
            attachment_types: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttachmentSource {
    Path(PathBuf),
    Url(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Attachment {
    pub mime_type: Option<String>,
    pub source: AttachmentSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub tool_call_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<ToolResult>,
}

impl Message {
    #[must_use]
    pub fn user(text: &str) -> Self {
        Self {
            role: Role::User,
            content: text.to_string(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        }
    }

    #[must_use]
    pub fn assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: text.to_string(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        }
    }

    #[must_use]
    pub fn assistant_with_tool_calls(text: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: text.to_string(),
            tool_calls,
            tool_results: Vec::new(),
        }
    }

    #[must_use]
    pub fn tool_results(results: Vec<ToolResult>) -> Self {
        Self {
            role: Role::Tool,
            content: String::new(),
            tool_calls: Vec::new(),
            tool_results: results,
        }
    }
}

pub type Options = std::collections::HashMap<String, serde_json::Value>;

/// A materialized response after stream collection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub response: String,
    pub options: Options,
    pub usage: Option<Usage>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub attachments: Vec<Attachment>,
    pub schema: Option<serde_json::Value>,
    pub schema_id: Option<String>,
    pub duration_ms: u64,
    pub datetime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prompt {
    pub text: String,
    pub system: Option<String>,
    pub attachments: Vec<Attachment>,
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub tool_results: Vec<ToolResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    pub schema: Option<serde_json::Value>,
    pub options: Options,
}

impl Prompt {
    #[must_use]
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            system: None,
            attachments: Vec::new(),
            tools: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            messages: Vec::new(),
            schema: None,
            options: Options::new(),
        }
    }

    #[must_use]
    pub fn with_system(mut self, system: &str) -> Self {
        self.system = Some(system.to_string());
        self
    }

    #[must_use]
    pub fn with_option(mut self, key: &str, value: serde_json::Value) -> Self {
        self.options.insert(key.to_string(), value);
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    #[must_use]
    pub fn with_tool_results(mut self, results: Vec<ToolResult>) -> Self {
        self.tool_results = results;
        self
    }

    #[must_use]
    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.schema = Some(schema);
        self
    }

    #[must_use]
    pub fn with_attachments(mut self, attachments: Vec<Attachment>) -> Self {
        self.attachments = attachments;
        self
    }

    #[must_use]
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_default_is_empty() {
        let usage = Usage::default();
        assert_eq!(usage.input, None);
        assert_eq!(usage.output, None);
        assert_eq!(usage.details, None);
    }

    #[test]
    fn usage_with_tokens() {
        let usage = Usage {
            input: Some(10),
            output: Some(20),
            details: None,
        };
        assert_eq!(usage.input, Some(10));
        assert_eq!(usage.output, Some(20));
    }

    #[test]
    fn usage_with_details() {
        let mut details = serde_json::Map::new();
        details.insert("cached".into(), serde_json::Value::Number(5.into()));
        let usage = Usage {
            input: Some(10),
            output: Some(20),
            details: Some(details),
        };
        assert_eq!(usage.details.as_ref().unwrap()["cached"], 5);
    }

    #[test]
    fn model_info_defaults() {
        let info = ModelInfo::new("gpt-4o");
        assert_eq!(info.id, "gpt-4o");
        assert!(info.can_stream);
        assert!(!info.supports_tools);
        assert!(!info.supports_schema);
        assert!(info.attachment_types.is_empty());
    }

    #[test]
    fn model_info_with_capabilities() {
        let info = ModelInfo {
            id: "gpt-4o".into(),
            can_stream: true,
            supports_tools: true,
            supports_schema: true,
            attachment_types: vec!["image/png".into(), "audio/wav".into()],
        };
        assert!(info.supports_tools);
        assert_eq!(info.attachment_types.len(), 2);
    }

    #[test]
    fn model_info_serializes_roundtrip() {
        let info = ModelInfo::new("gpt-4o");
        let json = serde_json::to_string(&info).unwrap();
        let restored: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, restored);
    }

    // --- Attachment tests ---

    #[test]
    fn attachment_from_path() {
        let att = Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Path("/tmp/test.png".into()),
        };
        assert_eq!(att.mime_type.as_deref(), Some("image/png"));
        assert!(matches!(att.source, AttachmentSource::Path(_)));
    }

    #[test]
    fn attachment_from_url() {
        let att = Attachment {
            mime_type: None,
            source: AttachmentSource::Url("https://example.com/img.png".into()),
        };
        assert_eq!(att.mime_type, None);
        if let AttachmentSource::Url(url) = &att.source {
            assert_eq!(url, "https://example.com/img.png");
        } else {
            panic!("expected Url source");
        }
    }

    #[test]
    fn attachment_from_bytes() {
        let data = vec![0x89, 0x50, 0x4e, 0x47]; // PNG magic bytes
        let att = Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Bytes(data.clone()),
        };
        if let AttachmentSource::Bytes(bytes) = &att.source {
            assert_eq!(bytes, &data);
        } else {
            panic!("expected Bytes source");
        }
    }

    // --- Tool tests ---

    #[test]
    fn tool_construction() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let tool = Tool {
            name: "search".into(),
            description: "Web search".into(),
            input_schema: schema.clone(),
        };
        assert_eq!(tool.name, "search");
        assert_eq!(tool.input_schema["type"], "object");
    }

    #[test]
    fn tool_serializes_roundtrip() {
        let tool = Tool {
            name: "calc".into(),
            description: "Calculator".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let restored: Tool = serde_json::from_str(&json).unwrap();
        assert_eq!(tool, restored);
    }

    // --- ToolCall tests ---

    #[test]
    fn tool_call_construction() {
        let call = ToolCall {
            name: "search".into(),
            arguments: serde_json::json!({"query": "rust async"}),
            tool_call_id: Some("tc_1".into()),
        };
        assert_eq!(call.name, "search");
        assert_eq!(call.arguments["query"], "rust async");
        assert_eq!(call.tool_call_id.as_deref(), Some("tc_1"));
    }

    #[test]
    fn tool_call_without_id() {
        let call = ToolCall {
            name: "time".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };
        assert_eq!(call.tool_call_id, None);
    }

    // --- ToolResult tests ---

    #[test]
    fn tool_result_success() {
        let result = ToolResult {
            name: "search".into(),
            output: "Found 3 results".into(),
            tool_call_id: Some("tc_1".into()),
            error: None,
        };
        assert_eq!(result.output, "Found 3 results");
        assert!(result.error.is_none());
    }

    #[test]
    fn tool_result_with_error() {
        let result = ToolResult {
            name: "search".into(),
            output: String::new(),
            tool_call_id: Some("tc_1".into()),
            error: Some("timeout".into()),
        };
        assert!(result.error.is_some());
    }

    // --- Prompt tests ---

    #[test]
    fn prompt_minimal() {
        let prompt = Prompt::new("Hello");
        assert_eq!(prompt.text, "Hello");
        assert_eq!(prompt.system, None);
        assert!(prompt.attachments.is_empty());
        assert!(prompt.tools.is_empty());
        assert!(prompt.tool_calls.is_empty());
        assert!(prompt.tool_results.is_empty());
        assert_eq!(prompt.schema, None);
        assert!(prompt.options.is_empty());
    }

    #[test]
    fn prompt_with_system() {
        let prompt = Prompt::new("Hello").with_system("You are helpful.");
        assert_eq!(prompt.system.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn prompt_with_options() {
        let prompt = Prompt::new("Hello")
            .with_option("temperature", serde_json::json!(0.7));
        assert_eq!(prompt.options["temperature"], 0.7);
    }

    #[test]
    fn prompt_with_tools() {
        let tool = Tool {
            name: "search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let prompt = Prompt::new("Find info").with_tools(vec![tool]);
        assert_eq!(prompt.tools.len(), 1);
        assert_eq!(prompt.tools[0].name, "search");
    }

    #[test]
    fn prompt_with_tool_results() {
        let result = ToolResult {
            name: "search".into(),
            output: "found it".into(),
            tool_call_id: Some("tc_1".into()),
            error: None,
        };
        let prompt = Prompt::new("Continue").with_tool_results(vec![result]);
        assert_eq!(prompt.tool_results.len(), 1);
    }

    #[test]
    fn prompt_with_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}}
        });
        let prompt = Prompt::new("Extract name").with_schema(schema.clone());
        assert_eq!(prompt.schema, Some(schema));
    }

    #[test]
    fn prompt_with_attachments() {
        let att = Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Path("/tmp/test.png".into()),
        };
        let prompt = Prompt::new("Describe this").with_attachments(vec![att]);
        assert_eq!(prompt.attachments.len(), 1);
    }

    #[test]
    fn prompt_builder_chains() {
        let prompt = Prompt::new("Hello")
            .with_system("Be brief")
            .with_option("temperature", serde_json::json!(0.5))
            .with_option("max_tokens", serde_json::json!(100));
        assert_eq!(prompt.system.as_deref(), Some("Be brief"));
        assert_eq!(prompt.options.len(), 2);
    }

    // --- Response tests ---

    #[test]
    fn response_minimal() {
        let resp = Response {
            id: "01J5B".into(),
            model: "gpt-4o".into(),
            prompt: "Hello".into(),
            system: None,
            response: "Hi there!".into(),
            options: Options::new(),
            usage: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            attachments: Vec::new(),
            schema: None,
            schema_id: None,
            duration_ms: 230,
            datetime: "2026-04-03T12:00:01Z".into(),
        };
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.prompt, "Hello");
        assert_eq!(resp.response, "Hi there!");
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn response_with_all_fields() {
        let resp = Response {
            id: "01J5C".into(),
            model: "gpt-4o".into(),
            prompt: "Search".into(),
            system: Some("Be helpful".into()),
            response: "Found it".into(),
            options: {
                let mut opts = Options::new();
                opts.insert("temperature".into(), serde_json::json!(0.7));
                opts
            },
            usage: Some(Usage {
                input: Some(50),
                output: Some(30),
                details: None,
            }),
            tool_calls: vec![ToolCall {
                name: "search".into(),
                arguments: serde_json::json!({"q": "X"}),
                tool_call_id: Some("tc_1".into()),
            }],
            tool_results: vec![ToolResult {
                name: "search".into(),
                output: "result".into(),
                tool_call_id: Some("tc_1".into()),
                error: None,
            }],
            attachments: vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Path("/tmp/img.png".into()),
            }],
            schema: Some(serde_json::json!({"type": "object"})),
            schema_id: Some("b3a8".into()),
            duration_ms: 1200,
            datetime: "2026-04-03T12:01:00Z".into(),
        };
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_results.len(), 1);
        assert_eq!(resp.attachments.len(), 1);
        assert_eq!(resp.usage.as_ref().unwrap().input, Some(50));
    }

    #[test]
    fn response_serializes_roundtrip() {
        let resp = Response {
            id: "01J5B".into(),
            model: "gpt-4o".into(),
            prompt: "Hello".into(),
            system: None,
            response: "Hi!".into(),
            options: Options::new(),
            usage: Some(Usage {
                input: Some(5),
                output: Some(8),
                details: None,
            }),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            attachments: Vec::new(),
            schema: None,
            schema_id: None,
            duration_ms: 230,
            datetime: "2026-04-03T12:00:01Z".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let restored: Response = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, restored);
    }

    #[test]
    fn usage_serializes_to_json() {
        let usage = Usage {
            input: Some(10),
            output: Some(20),
            details: None,
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["input"], 10);
        assert_eq!(json["output"], 20);
        assert_eq!(json["details"], serde_json::Value::Null);
    }

    // --- Role tests ---

    #[test]
    fn role_serde_roundtrip() {
        for (role, expected) in [
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
            (Role::Tool, "\"tool\""),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, expected);
            let restored: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, role);
        }
    }

    // --- Message tests ---

    #[test]
    fn message_user_construction() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
        assert!(msg.tool_calls.is_empty());
        assert!(msg.tool_results.is_empty());
    }

    #[test]
    fn message_assistant_construction() {
        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi there");
        assert!(msg.tool_calls.is_empty());
    }

    #[test]
    fn message_assistant_with_tool_calls_construction() {
        let calls = vec![ToolCall {
            name: "search".into(),
            arguments: serde_json::json!({"q": "rust"}),
            tool_call_id: Some("tc_1".into()),
        }];
        let msg = Message::assistant_with_tool_calls("Let me search", calls.clone());
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Let me search");
        assert_eq!(msg.tool_calls, calls);
    }

    #[test]
    fn message_tool_results_construction() {
        let results = vec![ToolResult {
            name: "search".into(),
            output: "found it".into(),
            tool_call_id: Some("tc_1".into()),
            error: None,
        }];
        let msg = Message::tool_results(results.clone());
        assert_eq!(msg.role, Role::Tool);
        assert!(msg.content.is_empty());
        assert_eq!(msg.tool_results, results);
    }

    #[test]
    fn message_serde_roundtrip() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, restored);
    }

    #[test]
    fn message_serde_skips_empty_vecs() {
        let msg = Message::user("Hello");
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("tool_results").is_none());
    }

    #[test]
    fn message_serde_includes_nonempty_tool_calls() {
        let msg = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                name: "t".into(),
                arguments: serde_json::json!({}),
                tool_call_id: None,
            }],
        );
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("tool_calls").is_some());
    }

    // --- Prompt with messages ---

    #[test]
    fn prompt_with_messages() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi!"),
            Message::user("Follow up"),
        ];
        let prompt = Prompt::new("Follow up").with_messages(messages.clone());
        assert_eq!(prompt.messages.len(), 3);
        assert_eq!(prompt.messages, messages);
    }

    #[test]
    fn prompt_default_has_empty_messages() {
        let prompt = Prompt::new("Hello");
        assert!(prompt.messages.is_empty());
    }

    #[test]
    fn prompt_messages_serde_skips_when_empty() {
        let prompt = Prompt::new("Hello");
        let json = serde_json::to_value(&prompt).unwrap();
        assert!(json.get("messages").is_none());
    }

    // --- Usage::add() / Usage::total() tests ---

    #[test]
    fn usage_add_both_some() {
        let a = Usage { input: Some(10), output: Some(5), details: None };
        let b = Usage { input: Some(20), output: Some(10), details: None };
        let result = a.add(&b);
        assert_eq!(result.input, Some(30));
        assert_eq!(result.output, Some(15));
        assert!(result.details.is_none());
    }

    #[test]
    fn usage_add_one_none() {
        let a = Usage { input: Some(10), output: None, details: None };
        let b = Usage { input: None, output: Some(5), details: None };
        let result = a.add(&b);
        assert_eq!(result.input, Some(10));
        assert_eq!(result.output, Some(5));
    }

    #[test]
    fn usage_add_both_none() {
        let a = Usage::default();
        let b = Usage::default();
        let result = a.add(&b);
        assert_eq!(result.input, None);
        assert_eq!(result.output, None);
    }

    #[test]
    fn usage_total() {
        let u = Usage { input: Some(10), output: Some(5), details: None };
        assert_eq!(u.total(), 15);

        let empty = Usage::default();
        assert_eq!(empty.total(), 0);
    }
}
