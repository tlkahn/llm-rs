use std::pin::Pin;

use futures::Stream;

use crate::error::LlmError;
use crate::types::{ToolCall, Usage};

#[derive(Debug, Clone, PartialEq)]
pub enum Chunk {
    Text(String),
    ToolCallStart { name: String, id: Option<String> },
    ToolCallDelta { content: String },
    Usage(Usage),
    Done,
}

pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<Chunk, LlmError>> + Send>>;

/// Extract concatenated text from a slice of chunks.
pub fn collect_text(chunks: &[Chunk]) -> String {
    let mut text = String::new();
    for chunk in chunks {
        if let Chunk::Text(t) = chunk {
            text.push_str(t);
        }
    }
    text
}

/// Assemble tool calls from a sequence of ToolCallStart/ToolCallDelta chunks.
pub fn collect_tool_calls(chunks: &[Chunk]) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_id: Option<String> = None;
    let mut current_args = String::new();

    for chunk in chunks {
        match chunk {
            Chunk::ToolCallStart { name, id } => {
                // Flush previous tool call if any
                if let Some(prev_name) = current_name.take() {
                    let arguments = serde_json::from_str(&current_args).unwrap_or_default();
                    calls.push(ToolCall {
                        name: prev_name,
                        arguments,
                        tool_call_id: current_id.take(),
                    });
                    current_args.clear();
                }
                current_name = Some(name.clone());
                current_id = id.clone();
            }
            Chunk::ToolCallDelta { content } => {
                current_args.push_str(content);
            }
            _ => {}
        }
    }
    // Flush last tool call
    if let Some(name) = current_name {
        let arguments = serde_json::from_str(&current_args).unwrap_or_default();
        calls.push(ToolCall {
            name,
            arguments,
            tool_call_id: current_id,
        });
    }
    calls
}

/// Extract the last Usage chunk, if any.
pub fn collect_usage(chunks: &[Chunk]) -> Option<Usage> {
    chunks.iter().rev().find_map(|c| {
        if let Chunk::Usage(u) = c {
            Some(u.clone())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    fn chunk_text_carries_content() {
        let chunk = Chunk::Text("hello".into());
        if let Chunk::Text(t) = &chunk {
            assert_eq!(t, "hello");
        } else {
            panic!("expected Text chunk");
        }
    }

    #[test]
    fn chunk_tool_call_start() {
        let chunk = Chunk::ToolCallStart {
            name: "search".into(),
            id: Some("tc_1".into()),
        };
        if let Chunk::ToolCallStart { name, id } = &chunk {
            assert_eq!(name, "search");
            assert_eq!(id.as_deref(), Some("tc_1"));
        } else {
            panic!("expected ToolCallStart");
        }
    }

    #[test]
    fn chunk_tool_call_delta() {
        let chunk = Chunk::ToolCallDelta {
            content: r#"{"query":"#.into(),
        };
        if let Chunk::ToolCallDelta { content } = &chunk {
            assert_eq!(content, r#"{"query":"#);
        } else {
            panic!("expected ToolCallDelta");
        }
    }

    #[test]
    fn chunk_usage() {
        let usage = Usage {
            input: Some(10),
            output: Some(5),
            details: None,
        };
        let chunk = Chunk::Usage(usage.clone());
        if let Chunk::Usage(u) = &chunk {
            assert_eq!(u, &usage);
        } else {
            panic!("expected Usage chunk");
        }
    }

    #[test]
    fn chunk_done() {
        let chunk = Chunk::Done;
        assert!(matches!(chunk, Chunk::Done));
    }

    #[tokio::test]
    async fn response_stream_collects_text() {
        let chunks = vec![
            Ok(Chunk::Text("Hello".into())),
            Ok(Chunk::Text(" world".into())),
            Ok(Chunk::Done),
        ];
        let stream: ResponseStream = Box::pin(futures::stream::iter(chunks));
        let collected: Vec<_> = stream.collect().await;
        assert_eq!(collected.len(), 3);

        let mut text = String::new();
        for item in &collected {
            if let Ok(Chunk::Text(t)) = item {
                text.push_str(t);
            }
        }
        assert_eq!(text, "Hello world");
    }

    #[tokio::test]
    async fn response_stream_propagates_error() {
        let chunks: Vec<Result<Chunk, LlmError>> = vec![
            Ok(Chunk::Text("Hi".into())),
            Err(LlmError::Provider("connection reset".into())),
        ];
        let stream: ResponseStream = Box::pin(futures::stream::iter(chunks));
        let collected: Vec<_> = stream.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(collected[0].is_ok());
        assert!(collected[1].is_err());
    }

    #[test]
    fn collect_text_from_chunks() {
        let chunks = vec![
            Chunk::Text("Hello".into()),
            Chunk::Text(" ".into()),
            Chunk::ToolCallStart {
                name: "x".into(),
                id: None,
            },
            Chunk::Text("world".into()),
            Chunk::Done,
        ];
        let text = collect_text(&chunks);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn collect_tool_calls_from_chunks() {
        let chunks = vec![
            Chunk::Text("Let me search.".into()),
            Chunk::ToolCallStart {
                name: "search".into(),
                id: Some("tc_1".into()),
            },
            Chunk::ToolCallDelta {
                content: r#"{"query":"#.into(),
            },
            Chunk::ToolCallDelta {
                content: r#""rust"}"#.into(),
            },
            Chunk::Done,
        ];
        let calls = collect_tool_calls(&chunks);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(calls[0].arguments, serde_json::json!({"query": "rust"}));
    }

    #[test]
    fn collect_tool_calls_multiple() {
        let chunks = vec![
            Chunk::ToolCallStart {
                name: "a".into(),
                id: Some("1".into()),
            },
            Chunk::ToolCallDelta {
                content: r#"{}"#.into(),
            },
            Chunk::ToolCallStart {
                name: "b".into(),
                id: Some("2".into()),
            },
            Chunk::ToolCallDelta {
                content: r#"{}"#.into(),
            },
            Chunk::Done,
        ];
        let calls = collect_tool_calls(&chunks);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn collect_usage_from_chunks() {
        let chunks = vec![
            Chunk::Text("Hi".into()),
            Chunk::Usage(Usage {
                input: Some(5),
                output: Some(1),
                details: None,
            }),
            Chunk::Done,
        ];
        let usage = collect_usage(&chunks);
        assert!(usage.is_some());
        assert_eq!(usage.unwrap().input, Some(5));
    }

    #[test]
    fn collect_usage_returns_none_when_absent() {
        let chunks = vec![Chunk::Text("Hi".into()), Chunk::Done];
        assert!(collect_usage(&chunks).is_none());
    }
}
