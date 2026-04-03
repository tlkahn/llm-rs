use std::collections::VecDeque;

use crate::types::StreamChunk;

const SSE_DATA_PREFIX: &str = "data: ";
const SSE_DONE: &str = "data: [DONE]";

/// Parse a complete SSE text into a list of StreamChunks.
/// Skips the `[DONE]` terminator.
pub fn parse_sse_events(input: &str) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == SSE_DONE {
            break;
        }
        if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                chunks.push(chunk);
            }
        }
    }
    chunks
}

/// Incremental SSE parser for streaming byte buffers.
/// Feed bytes as they arrive; pull parsed events out.
pub struct SseParser {
    buffer: String,
    events: VecDeque<StreamChunk>,
    done: bool,
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

    pub fn next_event(&mut self) -> Option<StreamChunk> {
        self.events.pop_front()
    }

    fn drain_buffer(&mut self) {
        // SSE events are separated by "\n\n"
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            let line = event_text.trim();
            if line == SSE_DONE {
                self.done = true;
                return;
            }
            if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    self.events.push_back(chunk);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_sse_event() {
        let input = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n";
        let events = parse_sse_events(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn parse_multiple_sse_events() {
        let input = "\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\".\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        let events = parse_sse_events(input);
        assert_eq!(events.len(), 4); // [DONE] is not a StreamChunk
    }

    #[test]
    fn parse_done_signal() {
        let input = "data: [DONE]\n\n";
        let events = parse_sse_events(input);
        assert!(events.is_empty());
    }

    #[test]
    fn parse_ignores_empty_lines() {
        let input = "\n\ndata: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\n";
        let events = parse_sse_events(input);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn parse_with_usage_event() {
        let input = "\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n\
data: [DONE]\n\n";
        let events = parse_sse_events(input);
        assert_eq!(events.len(), 1);
        let usage = events[0].usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 10);
    }

    // Test the incremental parser that works with partial byte buffers
    #[test]
    fn incremental_parser_handles_partial_data() {
        let mut parser = SseParser::new();

        // Feed partial data
        parser.feed(b"data: {\"id\":\"1\",\"object\":\"x\",\"model\":");
        assert!(parser.next_event().is_none());

        // Feed the rest
        parser.feed(b"\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n");
        let event = parser.next_event();
        assert!(event.is_some());
        let chunk = event.unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn incremental_parser_multiple_events_in_one_feed() {
        let mut parser = SseParser::new();
        let input = "\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"1\",\"object\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"B\"},\"finish_reason\":null}]}\n\n";
        parser.feed(input.as_bytes());

        let e1 = parser.next_event().unwrap();
        assert_eq!(e1.choices[0].delta.content.as_deref(), Some("A"));
        let e2 = parser.next_event().unwrap();
        assert_eq!(e2.choices[0].delta.content.as_deref(), Some("B"));
        assert!(parser.next_event().is_none());
    }

    #[test]
    fn incremental_parser_done_returns_none() {
        let mut parser = SseParser::new();
        parser.feed(b"data: [DONE]\n\n");
        assert!(parser.next_event().is_none());
        assert!(parser.is_done());
    }
}
