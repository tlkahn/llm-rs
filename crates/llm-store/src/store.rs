use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use llm_core::Result;
use llm_core::types::Response;

use crate::records::ConversationRecord;

/// Summary of a conversation for listing.
///
/// Lives in `store.rs` rather than `query.rs` so it is visible on wasm32 —
/// the `ConversationStore` trait signature references it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: String,
    pub model: String,
    pub name: Option<String>,
    pub created: String,
}

impl From<ConversationRecord> for ConversationSummary {
    fn from(rec: ConversationRecord) -> Self {
        Self {
            id: rec.id,
            model: rec.model,
            name: rec.name,
            created: rec.created,
        }
    }
}

/// Abstract conversation store interface.
///
/// `LogStore` implements this on native targets; `llm-wasm` provides a JS-callback
/// implementation. The trait is wasm-clean — no `std::fs`, no `tokio`.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait ConversationStore: Send + Sync {
    async fn log_response(
        &self,
        conversation_id: Option<&str>,
        model: &str,
        response: &Response,
    ) -> Result<String>;

    async fn read_conversation(
        &self,
        id: &str,
    ) -> Result<(ConversationRecord, Vec<Response>)>;

    async fn list_conversations(&self, limit: usize) -> Result<Vec<ConversationSummary>>;

    async fn latest_conversation_id(&self) -> Result<Option<String>>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait ConversationStore {
    async fn log_response(
        &self,
        conversation_id: Option<&str>,
        model: &str,
        response: &Response,
    ) -> Result<String>;

    async fn read_conversation(
        &self,
        id: &str,
    ) -> Result<(ConversationRecord, Vec<Response>)>;

    async fn list_conversations(&self, limit: usize) -> Result<Vec<ConversationSummary>>;

    async fn latest_conversation_id(&self) -> Result<Option<String>>;
}

/// Construct a `Response` from collected completion data, populating `id`
/// (fresh ULID) and `datetime` (now, RFC 3339). Native-only because it uses
/// `ulid` + `chrono`; the Python binding calls this from `LlmClient` after a
/// completion to synthesize the record for logging.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn build_response(
    model: &str,
    prompt: &str,
    system: Option<&str>,
    options: llm_core::types::Options,
    response_text: String,
    usage: Option<llm_core::types::Usage>,
    tool_calls: Vec<llm_core::types::ToolCall>,
    tool_results: Vec<llm_core::types::ToolResult>,
    schema: Option<serde_json::Value>,
    schema_id: Option<String>,
    duration_ms: u64,
) -> Response {
    Response {
        id: ulid::Ulid::new().to_string().to_lowercase(),
        model: model.to_string(),
        prompt: prompt.to_string(),
        system: system.map(|s| s.to_string()),
        response: response_text,
        options,
        usage,
        tool_calls,
        tool_results,
        attachments: Vec::new(),
        schema,
        schema_id,
        duration_ms,
        datetime: chrono::Utc::now().to_rfc3339(),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use llm_core::types::{Options, Usage};

    #[test]
    fn build_response_populates_id_and_datetime() {
        let r = build_response(
            "gpt-4o",
            "hello",
            None,
            Options::new(),
            "hi".into(),
            None,
            Vec::new(),
            Vec::new(),
            None,
            None,
            100,
        );
        assert_eq!(r.id.len(), 26);
        assert_eq!(r.id, r.id.to_lowercase());
        // RFC 3339 datetime parses.
        assert!(chrono::DateTime::parse_from_rfc3339(&r.datetime).is_ok());
        assert_eq!(r.model, "gpt-4o");
        assert_eq!(r.prompt, "hello");
        assert_eq!(r.response, "hi");
        assert!(r.attachments.is_empty());
    }

    #[test]
    fn build_response_preserves_all_fields() {
        let mut opts = Options::new();
        opts.insert("temperature".into(), serde_json::json!(0.7));
        let usage = Usage {
            input: Some(10),
            output: Some(20),
            details: None,
        };
        let r = build_response(
            "claude-sonnet-4-6",
            "prompt",
            Some("system"),
            opts.clone(),
            "text".into(),
            Some(usage.clone()),
            Vec::new(),
            Vec::new(),
            Some(serde_json::json!({"type": "object"})),
            Some("sch1".into()),
            450,
        );
        assert_eq!(r.system.as_deref(), Some("system"));
        assert_eq!(r.options, opts);
        assert_eq!(r.usage, Some(usage));
        assert_eq!(r.schema_id.as_deref(), Some("sch1"));
        assert_eq!(r.duration_ms, 450);
    }

    #[test]
    fn conversation_summary_from_record() {
        let rec = ConversationRecord {
            v: 1,
            id: "abc".into(),
            model: "gpt-4o".into(),
            name: Some("hi".into()),
            created: "2026-04-11T00:00:00Z".into(),
        };
        let sum: ConversationSummary = rec.into();
        assert_eq!(sum.id, "abc");
        assert_eq!(sum.model, "gpt-4o");
        assert_eq!(sum.name.as_deref(), Some("hi"));
    }
}
