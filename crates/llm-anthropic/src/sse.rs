use std::collections::VecDeque;

use crate::types::StreamEvent;

const SSE_DATA_PREFIX: &str = "data: ";

/// Parse a complete SSE text into a list of StreamEvents.
/// Handles Anthropic's `event: <type>\ndata: {json}` format by ignoring `event:` lines
/// and relying on the `type` field in the JSON payload.
pub fn parse_sse_events(input: &str) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("event:") {
            continue;
        }
        if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX)
            && let Ok(event) = serde_json::from_str::<StreamEvent>(data)
        {
            events.push(event);
        }
    }
    events
}

/// Incremental SSE parser for streaming byte buffers.
/// Feed bytes as they arrive; pull parsed events out.
pub struct SseParser {
    buffer: String,
    events: VecDeque<StreamEvent>,
    done: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            events: VecDeque::new(),
            done: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn feed(&mut self, data: &[u8]) {
        let text = String::from_utf8_lossy(data);
        self.buffer.push_str(&text);
        self.drain_buffer();
    }

    pub fn next_event(&mut self) -> Option<StreamEvent> {
        self.events.pop_front()
    }

    fn drain_buffer(&mut self) {
        // SSE events are separated by "\n\n"
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            // Extract the data: line from the event block (may also contain event: line)
            for line in event_text.lines() {
                let line = line.trim();
                if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX)
                    && let Ok(event) = serde_json::from_str::<StreamEvent>(data)
                {
                    if matches!(event, StreamEvent::MessageStop) {
                        self.done = true;
                    }
                    self.events.push_back(event);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sse_event(event_type: &str, data: &str) -> String {
        format!("event: {event_type}\ndata: {data}\n\n")
    }

    #[test]
    fn parse_single_content_block_delta() {
        let input = make_sse_event(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        );
        let events = parse_sse_events(&input);
        assert_eq!(events.len(), 1);
        if let StreamEvent::ContentBlockDelta { delta, .. } = &events[0] {
            assert_eq!(delta.text.as_deref(), Some("Hello"));
        } else {
            panic!("expected ContentBlockDelta");
        }
    }

    #[test]
    fn parse_full_session() {
        let input = format!(
            "{}{}{}{}{}{}{}",
            make_sse_event("message_start", r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":10,"output_tokens":0}}}"#),
            make_sse_event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
            make_sse_event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#),
            make_sse_event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#),
            make_sse_event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            make_sse_event("message_delta", r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#),
            make_sse_event("message_stop", r#"{"type":"message_stop"}"#),
        );
        let events = parse_sse_events(&input);
        assert_eq!(events.len(), 7);
        assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
        assert!(matches!(events[1], StreamEvent::ContentBlockStart { .. }));
        assert!(matches!(events[2], StreamEvent::ContentBlockDelta { .. }));
        assert!(matches!(events[3], StreamEvent::ContentBlockDelta { .. }));
        assert!(matches!(events[4], StreamEvent::ContentBlockStop { .. }));
        assert!(matches!(events[5], StreamEvent::MessageDelta { .. }));
        assert!(matches!(events[6], StreamEvent::MessageStop));
    }

    #[test]
    fn message_stop_is_detected() {
        let input = make_sse_event("message_stop", r#"{"type":"message_stop"}"#);
        let events = parse_sse_events(&input);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::MessageStop));
    }

    #[test]
    fn ping_events_handled() {
        let input = make_sse_event("ping", r#"{"type":"ping"}"#);
        let events = parse_sse_events(&input);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Ping));
    }

    #[test]
    fn event_lines_ignored_in_favor_of_json_type() {
        // Even without event: line, the JSON type field drives dispatch
        let input = "data: {\"type\":\"ping\"}\n\n";
        let events = parse_sse_events(input);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Ping));
    }

    #[test]
    fn message_start_contains_usage() {
        let input = make_sse_event(
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","usage":{"input_tokens":42,"output_tokens":0}}}"#,
        );
        let events = parse_sse_events(&input);
        if let StreamEvent::MessageStart { message } = &events[0] {
            let usage = message.usage.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 42);
        } else {
            panic!("expected MessageStart");
        }
    }

    #[test]
    fn message_delta_contains_output_usage() {
        let input = make_sse_event(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#,
        );
        let events = parse_sse_events(&input);
        if let StreamEvent::MessageDelta { usage, .. } = &events[0] {
            assert_eq!(usage.as_ref().unwrap().output_tokens, 15);
        } else {
            panic!("expected MessageDelta");
        }
    }

    // --- Incremental parser tests ---

    #[test]
    fn incremental_parser_handles_partial_data() {
        let mut parser = SseParser::new();

        // Feed partial data
        parser.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"");
        assert!(parser.next_event().is_none());

        // Feed the rest
        parser.feed(b",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n");
        let event = parser.next_event();
        assert!(event.is_some());
        if let Some(StreamEvent::ContentBlockDelta { delta, .. }) = event {
            assert_eq!(delta.text.as_deref(), Some("Hi"));
        } else {
            panic!("expected ContentBlockDelta");
        }
    }

    #[test]
    fn incremental_parser_multiple_events_in_one_feed() {
        let mut parser = SseParser::new();
        let input = format!(
            "{}{}",
            make_sse_event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"A"}}"#),
            make_sse_event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"B"}}"#),
        );
        parser.feed(input.as_bytes());

        let e1 = parser.next_event().unwrap();
        if let StreamEvent::ContentBlockDelta { delta, .. } = e1 {
            assert_eq!(delta.text.as_deref(), Some("A"));
        } else {
            panic!("expected delta A");
        }

        let e2 = parser.next_event().unwrap();
        if let StreamEvent::ContentBlockDelta { delta, .. } = e2 {
            assert_eq!(delta.text.as_deref(), Some("B"));
        } else {
            panic!("expected delta B");
        }

        assert!(parser.next_event().is_none());
    }

    #[test]
    fn incremental_parser_done_on_message_stop() {
        let mut parser = SseParser::new();
        let input = make_sse_event("message_stop", r#"{"type":"message_stop"}"#);
        parser.feed(input.as_bytes());
        assert!(parser.is_done());
        let event = parser.next_event();
        assert!(matches!(event, Some(StreamEvent::MessageStop)));
    }
}
