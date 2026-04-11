//! Pure helper that converts collected stream output + chain state into a
//! [`llm_core::types::Response`]. Split out so it can be unit-tested without
//! touching pyo3 / tokio.

use llm_core::stream::Chunk;
use llm_core::types::{Options, Response, ToolCall, ToolResult, Usage};
use llm_store::build_response;

/// Inputs to `synthesize_response`. All fields are owned / borrowed by the
/// caller; the function just glues them together and delegates to
/// `llm_store::build_response`.
pub struct ResponseInputs<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub system: Option<&'a str>,
    pub options: Options,
    pub chunks: &'a [Chunk],
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub total_usage: Option<Usage>,
    pub schema: Option<serde_json::Value>,
    pub schema_id: Option<String>,
    pub duration_ms: u64,
}

/// Build a `Response` from collected chunks + chain result data.
///
/// `tool_calls` and `tool_results` should come from the `ChainResult`
/// directly (or from `collect_tool_calls(&chunks)` for the no-chain path),
/// not re-derived from the final message history — that's lossy across
/// multi-iteration chains.
pub fn synthesize_response(inputs: ResponseInputs<'_>) -> Response {
    let text = llm_core::collect_text(inputs.chunks);
    build_response(
        inputs.model,
        inputs.prompt,
        inputs.system,
        inputs.options,
        text,
        inputs.total_usage,
        inputs.tool_calls,
        inputs.tool_results,
        inputs.schema,
        inputs.schema_id,
        inputs.duration_ms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks_with_text(parts: &[&str]) -> Vec<Chunk> {
        parts.iter().map(|p| Chunk::Text((*p).into())).collect()
    }

    #[test]
    fn concatenates_text_chunks() {
        let chunks = chunks_with_text(&["Hello", ", ", "world"]);
        let r = synthesize_response(ResponseInputs {
            model: "gpt-4o",
            prompt: "hi",
            system: None,
            options: Options::new(),
            chunks: &chunks,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            total_usage: None,
            schema: None,
            schema_id: None,
            duration_ms: 100,
        });
        assert_eq!(r.response, "Hello, world");
    }

    #[test]
    fn preserves_tool_calls_from_chain_result() {
        let tc = ToolCall {
            name: "t".into(),
            arguments: serde_json::json!({}),
            tool_call_id: Some("id1".into()),
        };
        let tr = ToolResult {
            name: "t".into(),
            output: "42".into(),
            tool_call_id: Some("id1".into()),
            error: None,
        };
        let r = synthesize_response(ResponseInputs {
            model: "gpt-4o",
            prompt: "hi",
            system: None,
            options: Options::new(),
            chunks: &[],
            tool_calls: vec![tc.clone()],
            tool_results: vec![tr.clone()],
            total_usage: None,
            schema: None,
            schema_id: None,
            duration_ms: 50,
        });
        assert_eq!(r.tool_calls, vec![tc]);
        assert_eq!(r.tool_results, vec![tr]);
    }

    #[test]
    fn system_prompt_passthrough() {
        let r = synthesize_response(ResponseInputs {
            model: "gpt-4o",
            prompt: "hi",
            system: Some("be concise"),
            options: Options::new(),
            chunks: &[],
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            total_usage: None,
            schema: None,
            schema_id: None,
            duration_ms: 0,
        });
        assert_eq!(r.system.as_deref(), Some("be concise"));
    }

    #[test]
    fn usage_passthrough() {
        let usage = Usage {
            input: Some(10),
            output: Some(5),
            details: None,
        };
        let r = synthesize_response(ResponseInputs {
            model: "gpt-4o",
            prompt: "hi",
            system: None,
            options: Options::new(),
            chunks: &[],
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            total_usage: Some(usage.clone()),
            schema: None,
            schema_id: None,
            duration_ms: 0,
        });
        assert_eq!(r.usage, Some(usage));
    }
}
