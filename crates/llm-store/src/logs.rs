use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use ulid::Ulid;

use llm_core::{LlmError, Response, Result};

use crate::records::{ConversationRecord, LineRecord, ResponseRecord};

/// Handle to the logs directory for reading/writing conversation JSONL files.
pub struct LogStore {
    logs_dir: PathBuf,
}

impl LogStore {
    /// Open (or create) the logs directory.
    pub fn open(logs_dir: &Path) -> Result<Self> {
        fs::create_dir_all(logs_dir)?;
        Ok(Self {
            logs_dir: logs_dir.to_path_buf(),
        })
    }

    /// Log a response to a conversation file.
    ///
    /// If `conversation_id` is `None`, creates a new conversation and returns its ID.
    /// If `conversation_id` is `Some(id)`, appends to the existing conversation file.
    pub fn log_response(
        &self,
        conversation_id: Option<&str>,
        model: &str,
        response: &Response,
    ) -> Result<String> {
        match conversation_id {
            None => self.create_conversation(model, response),
            Some(id) => {
                self.append_response(id, response)?;
                Ok(id.to_string())
            }
        }
    }

    fn conversation_path(&self, id: &str) -> PathBuf {
        self.logs_dir.join(format!("{id}.jsonl"))
    }

    fn create_conversation(&self, model: &str, response: &Response) -> Result<String> {
        let conv_id = Ulid::new().to_string().to_lowercase();
        let path = self.conversation_path(&conv_id);

        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write conversation header
        let header = LineRecord::Conversation(ConversationRecord {
            v: 1,
            id: conv_id.clone(),
            model: model.to_string(),
            name: conversation_name(&response.prompt),
            created: Utc::now().to_rfc3339(),
        });
        serde_json::to_writer(&mut writer, &header)
            .map_err(|e| LlmError::Store(e.to_string()))?;
        writer.write_all(b"\n")?;

        // Write response record
        let record = LineRecord::Response(Box::new(ResponseRecord {
            response: response.clone(),
        }));
        serde_json::to_writer(&mut writer, &record)
            .map_err(|e| LlmError::Store(e.to_string()))?;
        writer.write_all(b"\n")?;

        writer.flush()?;
        Ok(conv_id)
    }

    fn append_response(&self, id: &str, response: &Response) -> Result<()> {
        let path = self.conversation_path(id);
        if !path.exists() {
            return Err(LlmError::Store(format!(
                "conversation not found: {id}"
            )));
        }

        let file = fs::OpenOptions::new().append(true).open(&path)?;
        let mut writer = BufWriter::new(file);

        let record = LineRecord::Response(Box::new(ResponseRecord {
            response: response.clone(),
        }));
        serde_json::to_writer(&mut writer, &record)
            .map_err(|e| LlmError::Store(e.to_string()))?;
        writer.write_all(b"\n")?;
        writer.flush()?;

        Ok(())
    }

    /// Read a full conversation by ID.
    pub fn read_conversation(
        &self,
        id: &str,
    ) -> Result<(ConversationRecord, Vec<Response>)> {
        let path = self.conversation_path(id);
        if !path.exists() {
            return Err(LlmError::Store(format!(
                "conversation not found: {id}"
            )));
        }

        let content = fs::read_to_string(&path)?;
        let mut meta: Option<ConversationRecord> = None;
        let mut responses = Vec::new();

        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<LineRecord>(line) {
                Ok(LineRecord::Conversation(c)) => {
                    if meta.is_none() {
                        meta = Some(c);
                    }
                }
                Ok(LineRecord::Response(r)) => {
                    responses.push(r.response);
                }
                Err(e) => {
                    eprintln!(
                        "warning: skipping malformed line {} in {}: {e}",
                        i + 1,
                        path.display()
                    );
                }
            }
        }

        let meta = meta.ok_or_else(|| {
            LlmError::Store(format!(
                "no conversation header found in {}",
                path.display()
            ))
        })?;

        Ok((meta, responses))
    }
}

/// Generate a human-readable conversation name from prompt text.
pub fn conversation_name(prompt: &str) -> Option<String> {
    let collapsed: String = prompt
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.is_empty() {
        return None;
    }
    if collapsed.len() <= 100 {
        Some(collapsed)
    } else {
        // Truncate at a char boundary near 100
        let truncated = &collapsed[..collapsed.floor_char_boundary(100)];
        Some(format!("{truncated}..."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::{Options, Usage};
    use tempfile::TempDir;

    fn sample_response(prompt: &str, text: &str) -> Response {
        Response {
            id: Ulid::new().to_string().to_lowercase(),
            model: "gpt-4o".into(),
            prompt: prompt.into(),
            system: None,
            response: text.into(),
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

    // --- Cycle 2: Write new conversation ---

    #[test]
    fn log_response_creates_new_conversation_file() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi there!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        let path = dir.path().join(format!("{conv_id}.jsonl"));
        assert!(path.exists(), "conversation file should exist");
    }

    #[test]
    fn log_response_new_has_two_lines() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        let content = fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "file should have exactly 2 lines");
    }

    #[test]
    fn log_response_new_first_line_is_conversation_header() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        let content = fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let first_line = content.lines().next().unwrap();
        let json: serde_json::Value = serde_json::from_str(first_line).unwrap();
        assert_eq!(json["type"], "conversation");
        assert_eq!(json["v"], 1);
        assert_eq!(json["id"], conv_id);
        assert_eq!(json["model"], "gpt-4o");
    }

    #[test]
    fn log_response_new_second_line_is_response() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        let content = fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let second_line = content.lines().nth(1).unwrap();
        let json: serde_json::Value = serde_json::from_str(second_line).unwrap();
        assert_eq!(json["type"], "response");
        assert_eq!(json["prompt"], "Hello");
        assert_eq!(json["response"], "Hi!");
    }

    #[test]
    fn log_response_returns_valid_ulid() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        // ULID is 26 chars, lowercase
        assert_eq!(conv_id.len(), 26);
        assert_eq!(conv_id, conv_id.to_lowercase());
    }

    // --- Cycle 3: Append to existing conversation ---

    #[test]
    fn log_response_appends_to_existing() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp1 = sample_response("Hello", "Hi!");
        let resp2 = sample_response("Follow up", "Sure!");

        let conv_id = store.log_response(None, "gpt-4o", &resp1).unwrap();
        store
            .log_response(Some(&conv_id), "gpt-4o", &resp2)
            .unwrap();

        let content =
            fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3, "file should have 3 lines after append");
    }

    #[test]
    fn log_response_append_preserves_header() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp1 = sample_response("Hello", "Hi!");
        let resp2 = sample_response("Follow up", "Sure!");

        let conv_id = store.log_response(None, "gpt-4o", &resp1).unwrap();

        // Capture header before append
        let content_before =
            fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let header_before = content_before.lines().next().unwrap().to_string();

        store
            .log_response(Some(&conv_id), "gpt-4o", &resp2)
            .unwrap();

        let content_after =
            fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let header_after = content_after.lines().next().unwrap();

        assert_eq!(header_before, header_after, "header should be unchanged");
    }

    #[test]
    fn log_response_append_third_line_is_second_response() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp1 = sample_response("Hello", "Hi!");
        let resp2 = sample_response("Follow up", "Sure!");

        let conv_id = store.log_response(None, "gpt-4o", &resp1).unwrap();
        store
            .log_response(Some(&conv_id), "gpt-4o", &resp2)
            .unwrap();

        let content =
            fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        let third_line = content.lines().nth(2).unwrap();
        let json: serde_json::Value = serde_json::from_str(third_line).unwrap();
        assert_eq!(json["type"], "response");
        assert_eq!(json["prompt"], "Follow up");
        assert_eq!(json["response"], "Sure!");
    }

    #[test]
    fn log_response_append_nonexistent_errors() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let result = store.log_response(Some("nonexistent"), "gpt-4o", &resp);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("conversation not found"),
            "error should mention conversation not found, got: {err}"
        );
    }

    #[test]
    fn log_store_creates_directory_if_missing() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("deep").join("logs");
        assert!(!nested.exists());

        let store = LogStore::open(&nested).unwrap();
        assert!(nested.exists());

        let resp = sample_response("Hello", "Hi!");
        store.log_response(None, "gpt-4o", &resp).unwrap();
    }

    // --- Cycle 4: Read conversation ---

    #[test]
    fn read_conversation_roundtrip_single_response() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();
        let (meta, responses) = store.read_conversation(&conv_id).unwrap();

        assert_eq!(meta.id, conv_id);
        assert_eq!(meta.model, "gpt-4o");
        assert_eq!(meta.v, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].prompt, "Hello");
        assert_eq!(responses[0].response, "Hi!");
    }

    #[test]
    fn read_conversation_roundtrip_multiple_responses() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp1 = sample_response("Hello", "Hi!");
        let resp2 = sample_response("Follow up", "Sure!");

        let conv_id = store.log_response(None, "gpt-4o", &resp1).unwrap();
        store
            .log_response(Some(&conv_id), "gpt-4o", &resp2)
            .unwrap();

        let (_, responses) = store.read_conversation(&conv_id).unwrap();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].prompt, "Hello");
        assert_eq!(responses[1].prompt, "Follow up");
    }

    #[test]
    fn read_conversation_preserves_all_fields() {
        use llm_core::{Attachment, AttachmentSource, ToolCall, ToolResult};

        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let resp = Response {
            id: Ulid::new().to_string().to_lowercase(),
            model: "gpt-4o".into(),
            prompt: "Search for X".into(),
            system: Some("Be helpful".into()),
            response: "Found it.".into(),
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
                output: "result...".into(),
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

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();
        let (_, responses) = store.read_conversation(&conv_id).unwrap();

        assert_eq!(responses.len(), 1);
        let r = &responses[0];
        assert_eq!(r.system.as_deref(), Some("Be helpful"));
        assert_eq!(r.options["temperature"], 0.7);
        assert_eq!(r.usage.as_ref().unwrap().input, Some(50));
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "search");
        assert_eq!(r.tool_results.len(), 1);
        assert_eq!(r.attachments.len(), 1);
        assert_eq!(r.schema, Some(serde_json::json!({"type": "object"})));
        assert_eq!(r.schema_id.as_deref(), Some("b3a8"));
        assert_eq!(r.duration_ms, 1200);
    }

    #[test]
    fn read_conversation_nonexistent_errors() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let result = store.read_conversation("nonexistent");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("conversation not found"));
    }

    // --- Cycle 7: Conversation name helper ---

    #[test]
    fn conversation_name_short_text() {
        let name = conversation_name("Hello world");
        assert_eq!(name, Some("Hello world".into()));
    }

    #[test]
    fn conversation_name_truncates_long_text() {
        let long = "a ".repeat(80); // 160 chars
        let name = conversation_name(&long).unwrap();
        assert!(name.ends_with("..."));
        assert!(name.len() <= 104); // 100 + "..."
    }

    #[test]
    fn conversation_name_collapses_newlines() {
        let name = conversation_name("Hello\n\nworld\n");
        assert_eq!(name, Some("Hello world".into()));
    }

    #[test]
    fn conversation_name_collapses_extra_whitespace() {
        let name = conversation_name("  Hello   world  ");
        assert_eq!(name, Some("Hello world".into()));
    }

    #[test]
    fn conversation_name_empty_returns_none() {
        assert_eq!(conversation_name(""), None);
        assert_eq!(conversation_name("   "), None);
    }

    #[test]
    fn read_conversation_skips_malformed_lines() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "Hi!");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        // Append a malformed line
        let path = dir.path().join(format!("{conv_id}.jsonl"));
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        use std::io::Write;
        writeln!(file, "{{not valid json!!!").unwrap();

        // Should still read successfully, skipping the bad line
        let (meta, responses) = store.read_conversation(&conv_id).unwrap();
        assert_eq!(meta.id, conv_id);
        assert_eq!(responses.len(), 1);
    }

    // --- Cycle 8: Edge cases ---

    #[test]
    fn unicode_in_prompt_and_response() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("你好世界 🌍", "こんにちは！ 🎉");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();
        let (_, responses) = store.read_conversation(&conv_id).unwrap();

        assert_eq!(responses[0].prompt, "你好世界 🌍");
        assert_eq!(responses[0].response, "こんにちは！ 🎉");
    }

    #[test]
    fn newlines_in_response_text_dont_break_jsonl() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response(
            "Write code",
            "Here's some code:\n\nfn main() {\n    println!(\"hello\");\n}\n",
        );

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();

        // File should still have exactly 2 lines (newlines in content are escaped)
        let content =
            fs::read_to_string(dir.path().join(format!("{conv_id}.jsonl"))).unwrap();
        assert_eq!(content.lines().count(), 2);

        // Round-trip preserves the newlines
        let (_, responses) = store.read_conversation(&conv_id).unwrap();
        assert!(responses[0].response.contains('\n'));
        assert!(responses[0].response.contains("fn main()"));
    }

    #[test]
    fn empty_response_text() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let resp = sample_response("Hello", "");

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();
        let (_, responses) = store.read_conversation(&conv_id).unwrap();

        assert_eq!(responses[0].response, "");
    }

    #[test]
    fn response_with_all_optionals_none() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let resp = Response {
            id: Ulid::new().to_string().to_lowercase(),
            model: "gpt-4o".into(),
            prompt: "Hello".into(),
            system: None,
            response: "Hi".into(),
            options: Options::new(),
            usage: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            attachments: Vec::new(),
            schema: None,
            schema_id: None,
            duration_ms: 0,
            datetime: "2026-04-03T12:00:00Z".into(),
        };

        let conv_id = store.log_response(None, "gpt-4o", &resp).unwrap();
        let (_, responses) = store.read_conversation(&conv_id).unwrap();

        assert_eq!(responses[0].usage, None);
        assert_eq!(responses[0].schema, None);
        assert_eq!(responses[0].schema_id, None);
        assert_eq!(responses[0].system, None);
        assert!(responses[0].tool_calls.is_empty());
    }

    #[test]
    fn conversation_name_unicode_truncation() {
        // Ensure truncation doesn't split a multi-byte char
        let name = conversation_name(&"日本語テスト ".repeat(30)).unwrap();
        assert!(name.ends_with("..."));
        // Verify it's valid UTF-8 (would panic if char boundary is wrong)
        assert!(name.is_char_boundary(name.len()));
    }
}
