use std::fs;
use std::io::BufRead;
use std::path::Path;

use llm_core::Result;

use crate::records::{ConversationRecord, LineRecord};

/// Summary of a conversation for listing.
#[derive(Debug, Clone, PartialEq)]
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

/// List recent conversations sorted by file modification time (newest first).
///
/// Reads only the first line of each `.jsonl` file to extract metadata.
/// Files with malformed headers or non-`.jsonl` extensions are silently skipped.
pub fn list_conversations(logs_dir: &Path, limit: usize) -> Result<Vec<ConversationSummary>> {
    if !logs_dir.exists() {
        return Ok(Vec::new());
    }

    // Collect .jsonl files with their modification times
    let mut entries: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    for entry in fs::read_dir(logs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if let Ok(meta) = entry.metadata()
            && let Ok(mtime) = meta.modified()
        {
            entries.push((mtime, path));
        }
    }

    // Sort by mtime descending (newest first)
    entries.sort_by(|a, b| b.0.cmp(&a.0));

    let mut summaries = Vec::new();
    for (_, path) in entries {
        if summaries.len() >= limit {
            break;
        }
        if let Some(summary) = read_first_line_summary(&path) {
            summaries.push(summary);
        }
    }

    Ok(summaries)
}

/// Get the most recently modified conversation ID, if any.
pub fn latest_conversation_id(logs_dir: &Path) -> Result<Option<String>> {
    let summaries = list_conversations(logs_dir, 1)?;
    Ok(summaries.into_iter().next().map(|s| s.id))
}

fn read_first_line_summary(path: &Path) -> Option<ConversationSummary> {
    let file = fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line).ok()?;
    let first_line = first_line.trim();
    if first_line.is_empty() {
        return None;
    }
    match serde_json::from_str::<LineRecord>(first_line) {
        Ok(LineRecord::Conversation(c)) => Some(c.into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logs::LogStore;
    use llm_core::{Options, Response, Usage};
    use tempfile::TempDir;
    use ulid::Ulid;

    fn sample_response(prompt: &str) -> Response {
        Response {
            id: Ulid::new().to_string().to_lowercase(),
            model: "gpt-4o".into(),
            prompt: prompt.into(),
            system: None,
            response: "Reply".into(),
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
            duration_ms: 100,
            datetime: "2026-04-03T12:00:00Z".into(),
        }
    }

    #[test]
    fn list_conversations_empty_dir() {
        let dir = TempDir::new().unwrap();
        let summaries = list_conversations(dir.path(), 10).unwrap();
        assert!(summaries.is_empty());
    }

    #[test]
    fn list_conversations_nonexistent_dir() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("nope");
        let summaries = list_conversations(&missing, 10).unwrap();
        assert!(summaries.is_empty());
    }

    #[test]
    fn list_conversations_returns_summaries() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let id1 = store
            .log_response(None, "gpt-4o", &sample_response("First"))
            .unwrap();
        // Small sleep to ensure different mtime
        std::thread::sleep(std::time::Duration::from_millis(50));
        let id2 = store
            .log_response(None, "gpt-4o-mini", &sample_response("Second"))
            .unwrap();

        let summaries = list_conversations(dir.path(), 10).unwrap();
        assert_eq!(summaries.len(), 2);
        // Newest first
        assert_eq!(summaries[0].id, id2);
        assert_eq!(summaries[0].model, "gpt-4o-mini");
        assert_eq!(summaries[1].id, id1);
        assert_eq!(summaries[1].model, "gpt-4o");
    }

    #[test]
    fn list_conversations_respects_limit() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        for i in 0..5 {
            store
                .log_response(None, "gpt-4o", &sample_response(&format!("Prompt {i}")))
                .unwrap();
        }

        let summaries = list_conversations(dir.path(), 3).unwrap();
        assert_eq!(summaries.len(), 3);
    }

    #[test]
    fn list_conversations_skips_non_jsonl_files() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        store
            .log_response(None, "gpt-4o", &sample_response("Hello"))
            .unwrap();

        // Create a non-jsonl file
        fs::write(dir.path().join("notes.txt"), "not a conversation").unwrap();

        let summaries = list_conversations(dir.path(), 10).unwrap();
        assert_eq!(summaries.len(), 1);
    }

    #[test]
    fn list_conversations_skips_malformed_first_line() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        store
            .log_response(None, "gpt-4o", &sample_response("Hello"))
            .unwrap();

        // Create a .jsonl file with garbage content
        fs::write(dir.path().join("bad.jsonl"), "not valid json\n").unwrap();

        let summaries = list_conversations(dir.path(), 10).unwrap();
        assert_eq!(summaries.len(), 1);
    }

    #[test]
    fn list_conversations_includes_name() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        store
            .log_response(None, "gpt-4o", &sample_response("What is Rust?"))
            .unwrap();

        let summaries = list_conversations(dir.path(), 10).unwrap();
        assert_eq!(summaries[0].name.as_deref(), Some("What is Rust?"));
    }

    // --- Cycle 6: Latest conversation ---

    #[test]
    fn latest_conversation_empty_dir() {
        let dir = TempDir::new().unwrap();
        let result = latest_conversation_id(dir.path()).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn latest_conversation_returns_most_recent() {
        let dir = TempDir::new().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        store
            .log_response(None, "gpt-4o", &sample_response("First"))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let id2 = store
            .log_response(None, "gpt-4o", &sample_response("Second"))
            .unwrap();

        let latest = latest_conversation_id(dir.path()).unwrap();
        assert_eq!(latest, Some(id2));
    }
}
