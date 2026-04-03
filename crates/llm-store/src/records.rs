use serde::{Deserialize, Serialize};

use llm_core::Response;

/// First line of each conversation JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationRecord {
    pub v: u32,
    pub id: String,
    pub model: String,
    pub name: Option<String>,
    pub created: String,
}

/// A response line in a conversation JSONL file.
/// Wraps llm_core::Response with serde(flatten) so all Response fields
/// appear at the top level of the JSON object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseRecord {
    #[serde(flatten)]
    pub response: Response,
}

/// Tagged union for deserializing any line from a conversation JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LineRecord {
    #[serde(rename = "conversation")]
    Conversation(ConversationRecord),
    #[serde(rename = "response")]
    Response(Box<ResponseRecord>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::{Options, Usage};

    fn sample_response() -> Response {
        Response {
            id: "01J5B".into(),
            model: "gpt-4o".into(),
            prompt: "Hello".into(),
            system: None,
            response: "Hi there!".into(),
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
        }
    }

    #[test]
    fn conversation_record_serializes_with_correct_fields() {
        let rec = ConversationRecord {
            v: 1,
            id: "01J5A".into(),
            model: "gpt-4o".into(),
            name: Some("Hello".into()),
            created: "2026-04-03T12:00:00Z".into(),
        };
        let json: serde_json::Value = serde_json::to_value(&rec).unwrap();
        assert_eq!(json["v"], 1);
        assert_eq!(json["id"], "01J5A");
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["name"], "Hello");
        assert_eq!(json["created"], "2026-04-03T12:00:00Z");
    }

    #[test]
    fn conversation_record_roundtrip() {
        let rec = ConversationRecord {
            v: 1,
            id: "01J5A".into(),
            model: "gpt-4o".into(),
            name: None,
            created: "2026-04-03T12:00:00Z".into(),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let restored: ConversationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(rec, restored);
    }

    #[test]
    fn response_record_serializes_with_flattened_fields() {
        let rec = ResponseRecord {
            response: sample_response(),
        };
        let json: serde_json::Value = serde_json::to_value(&rec).unwrap();
        // Response fields should be at the top level, not nested under "response"
        assert_eq!(json["id"], "01J5B");
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["prompt"], "Hello");
        assert_eq!(json["response"], "Hi there!");
        assert_eq!(json["duration_ms"], 230);
        assert!(json.get("response_record").is_none()); // no wrapper key
    }

    #[test]
    fn response_record_roundtrip() {
        let rec = ResponseRecord {
            response: sample_response(),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let restored: ResponseRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(rec, restored);
    }

    #[test]
    fn line_record_deserializes_conversation() {
        let json = r#"{"type":"conversation","v":1,"id":"01J5A","model":"gpt-4o","name":"Hello","created":"2026-04-03T12:00:00Z"}"#;
        let record: LineRecord = serde_json::from_str(json).unwrap();
        match record {
            LineRecord::Conversation(c) => {
                assert_eq!(c.id, "01J5A");
                assert_eq!(c.v, 1);
            }
            _ => panic!("expected Conversation variant"),
        }
    }

    #[test]
    fn line_record_deserializes_response() {
        let rec = ResponseRecord {
            response: sample_response(),
        };
        // Build the JSON with a "type":"response" tag
        let mut json_val = serde_json::to_value(&rec).unwrap();
        json_val.as_object_mut().unwrap().insert("type".into(), "response".into());
        let json = serde_json::to_string(&json_val).unwrap();

        let record: LineRecord = serde_json::from_str(&json).unwrap();
        match record {
            LineRecord::Response(r) => {
                assert_eq!(r.response.id, "01J5B");
                assert_eq!(r.response.prompt, "Hello");
            }
            _ => panic!("expected Response variant"),
        }
    }

    #[test]
    fn line_record_serializes_conversation_with_type_tag() {
        let conv = ConversationRecord {
            v: 1,
            id: "01J5A".into(),
            model: "gpt-4o".into(),
            name: None,
            created: "2026-04-03T12:00:00Z".into(),
        };
        let line = LineRecord::Conversation(conv);
        let json: serde_json::Value = serde_json::to_value(&line).unwrap();
        assert_eq!(json["type"], "conversation");
        assert_eq!(json["v"], 1);
    }

    #[test]
    fn line_record_serializes_response_with_type_tag() {
        let rec = ResponseRecord {
            response: sample_response(),
        };
        let line = LineRecord::Response(Box::new(rec));
        let json: serde_json::Value = serde_json::to_value(&line).unwrap();
        assert_eq!(json["type"], "response");
        assert_eq!(json["id"], "01J5B");
        assert_eq!(json["prompt"], "Hello");
    }
}
