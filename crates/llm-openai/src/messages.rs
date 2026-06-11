use crate::types::{ContentPart, ImageUrl, Message, MessageContent, MessageToolCall, MessageToolCallFunction};
use llm_core::attachment::{ResolvedImageBlock, resolve_attachments};
use llm_core::types::Attachment;
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> llm_core::Result<Vec<Message>> {
    if prompt.messages.is_empty() {
        build_single_turn(prompt)
    } else {
        build_from_conversation(prompt)
    }
}

/// Map resolved image blocks to OpenAI `ContentPart`s.
fn resolved_to_parts(resolved: Vec<ResolvedImageBlock>) -> Vec<ContentPart> {
    resolved
        .into_iter()
        .map(|r| match r {
            ResolvedImageBlock::Base64 { media_type, base64_data } => {
                let data_uri = format!("data:{media_type};base64,{base64_data}");
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: data_uri,
                        detail: None,
                    },
                }
            }
            ResolvedImageBlock::Url { url } => ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url,
                    detail: None,
                },
            },
        })
        .collect()
}

/// Build a `MessageContent` for a user message, incorporating image attachments.
///
/// When attachments are present, produces `Parts` with the text part FIRST, then
/// image parts (OpenAI convention: text first, images after).
/// When attachments are empty, produces `Text(text)`.
fn build_user_content(text: &str, attachments: &[Attachment]) -> llm_core::Result<MessageContent> {
    if attachments.is_empty() {
        return Ok(MessageContent::Text(text.to_string()));
    }

    let resolved = resolve_attachments(attachments)?;
    let mut parts: Vec<ContentPart> = Vec::new();

    // Text part first (OpenAI convention)
    if !text.is_empty() {
        parts.push(ContentPart::Text { text: text.to_string() });
    }

    // Image parts after text
    parts.extend(resolved_to_parts(resolved));

    // Guard: OpenAI rejects an empty content array with 400.
    if parts.is_empty() {
        return Err(llm_core::LlmError::Provider(
            "empty content: no text and no image parts were produced from attachments".into(),
        ));
    }

    Ok(MessageContent::Parts(parts))
}

fn build_single_turn(prompt: &Prompt) -> llm_core::Result<Vec<Message>> {
    let mut messages = Vec::new();

    if let Some(system) = &prompt.system
        && !system.is_empty()
    {
        messages.push(Message {
            role: "system".into(),
            content: Some(MessageContent::Text(system.clone())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    let user_content = build_user_content(&prompt.text, &prompt.attachments)?;
    messages.push(Message {
        role: "user".into(),
        content: Some(user_content),
        tool_calls: None,
        tool_call_id: None,
    });

    // If there are tool calls and tool results, add assistant + tool messages
    if !prompt.tool_calls.is_empty() && !prompt.tool_results.is_empty() {
        append_tool_exchange(&mut messages, &prompt.tool_calls, &prompt.tool_results);
    }

    Ok(messages)
}

fn build_from_conversation(prompt: &Prompt) -> llm_core::Result<Vec<Message>> {
    let mut messages = Vec::new();

    // System prompt as first message
    if let Some(system) = &prompt.system
        && !system.is_empty()
    {
        messages.push(Message {
            role: "system".into(),
            content: Some(MessageContent::Text(system.clone())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for msg in &prompt.messages {
        match msg.role {
            llm_core::Role::User => {
                messages.push(Message {
                    role: "user".into(),
                    content: Some(MessageContent::Text(msg.content.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            llm_core::Role::Assistant => {
                if msg.tool_calls.is_empty() {
                    messages.push(Message {
                        role: "assistant".into(),
                        content: Some(MessageContent::Text(msg.content.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                } else {
                    let tool_calls = map_tool_calls(&msg.tool_calls);
                    messages.push(Message {
                        role: "assistant".into(),
                        content: if msg.content.is_empty() {
                            None
                        } else {
                            Some(MessageContent::Text(msg.content.clone()))
                        },
                        tool_calls: Some(tool_calls),
                        tool_call_id: None,
                    });
                }
            }
            llm_core::Role::Tool => {
                for result in &msg.tool_results {
                    messages.push(Message {
                        role: "tool".into(),
                        content: Some(MessageContent::Text(result.output.clone())),
                        tool_calls: None,
                        tool_call_id: result.tool_call_id.clone(),
                    });
                }
            }
        }
    }

    // Inject attachments into the last user message
    if !prompt.attachments.is_empty() {
        inject_attachments_into_last_user_message(&mut messages, &prompt.attachments)?;
    }

    Ok(messages)
}

/// Inject image attachments into the last user-role message in the conversation.
///
/// If the last user message was `Text(s)`, it becomes `Parts([text, images...])`.
/// If it was already `Parts(ps)`, image parts are appended.
fn inject_attachments_into_last_user_message(
    messages: &mut [Message],
    attachments: &[Attachment],
) -> llm_core::Result<()> {
    let last_user = messages
        .iter_mut()
        .rev()
        .find(|m| m.role == "user");

    let Some(msg) = last_user else { return Ok(()) };

    let resolved = resolve_attachments(attachments)?;
    let image_parts = resolved_to_parts(resolved);

    if image_parts.is_empty() {
        return Ok(());
    }

    // Move existing content out to avoid cloning large payloads.
    match msg.content.take() {
        Some(MessageContent::Text(text)) => {
            let mut parts = Vec::new();
            // Text first (OpenAI convention)
            if !text.is_empty() {
                parts.push(ContentPart::Text { text });
            }
            parts.extend(image_parts);
            // Guard: OpenAI rejects an empty content array with 400.
            if parts.is_empty() {
                return Err(llm_core::LlmError::Provider(
                    "empty content: no text and no image parts were produced from attachments"
                        .into(),
                ));
            }
            msg.content = Some(MessageContent::Parts(parts));
        }
        Some(MessageContent::Parts(mut existing)) => {
            existing.extend(image_parts);
            msg.content = Some(MessageContent::Parts(existing));
        }
        None => {
            msg.content = Some(MessageContent::Parts(image_parts));
        }
    }

    Ok(())
}

fn map_tool_calls(calls: &[llm_core::ToolCall]) -> Vec<MessageToolCall> {
    calls
        .iter()
        .map(|tc| MessageToolCall {
            id: tc.tool_call_id.clone().unwrap_or_default(),
            call_type: "function".into(),
            function: MessageToolCallFunction {
                name: tc.name.clone(),
                arguments: tc.arguments.to_string(),
            },
        })
        .collect()
}

fn append_tool_exchange(
    messages: &mut Vec<Message>,
    tool_calls: &[llm_core::ToolCall],
    tool_results: &[llm_core::ToolResult],
) {
    messages.push(Message {
        role: "assistant".into(),
        content: None,
        tool_calls: Some(map_tool_calls(tool_calls)),
        tool_call_id: None,
    });

    for result in tool_results {
        messages.push(Message {
            role: "tool".into(),
            content: Some(MessageContent::Text(result.output.clone())),
            tool_calls: None,
            tool_call_id: result.tool_call_id.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_user_only() {
        let prompt = Prompt::new("Hello");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_ref().and_then(|c| c.as_text()), Some("Hello"));
    }

    #[test]
    fn build_messages_with_system() {
        let prompt = Prompt::new("Hello").with_system("Be brief.");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content.as_ref().and_then(|c| c.as_text()), Some("Be brief."));
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_ref().and_then(|c| c.as_text()), Some("Hello"));
    }

    #[test]
    fn build_messages_empty_system_is_skipped() {
        let prompt = Prompt::new("Hello").with_system("");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn build_messages_with_tool_results() {
        use llm_core::{ToolCall, ToolResult};

        let prompt = Prompt::new("What's the weather?")
            .with_tool_calls(vec![ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"location": "Paris"}),
                tool_call_id: Some("call_1".into()),
            }])
            .with_tool_results(vec![ToolResult {
                name: "get_weather".into(),
                output: "Sunny, 22C".into(),
                tool_call_id: Some("call_1".into()),
                error: None,
            }]);

        let messages = build_messages(&prompt).unwrap();
        // system(0) + user(1) + assistant(2) + tool(3) = 3 messages (no system)
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].tool_calls.is_some());
        let tcs = messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].function.name, "get_weather");
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content.as_ref().and_then(|c| c.as_text()), Some("Sunny, 22C"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn build_messages_without_tool_results_unchanged() {
        let prompt = Prompt::new("Hello");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn build_messages_multi_turn_conversation() {
        use llm_core::Message as CoreMessage;

        let prompt = Prompt::new("")
            .with_system("Be helpful")
            .with_messages(vec![
                CoreMessage::user("Hello"),
                CoreMessage::assistant("Hi!"),
                CoreMessage::user("How are you?"),
            ]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 4); // system + 3 conversation
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_ref().and_then(|c| c.as_text()), Some("Hello"));
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].content.as_ref().and_then(|c| c.as_text()), Some("Hi!"));
        assert_eq!(messages[3].role, "user");
        assert_eq!(messages[3].content.as_ref().and_then(|c| c.as_text()), Some("How are you?"));
    }

    #[test]
    fn build_messages_multi_turn_with_tool_calls() {
        use llm_core::{Message as CoreMessage, ToolCall, ToolResult};

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("What time is it?"),
                CoreMessage::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        name: "get_time".into(),
                        arguments: serde_json::json!({}),
                        tool_call_id: Some("call_1".into()),
                    }],
                ),
                CoreMessage::tool_results(vec![ToolResult {
                    name: "get_time".into(),
                    output: "12:00 PM".into(),
                    tool_call_id: Some("call_1".into()),
                    error: None,
                }]),
                CoreMessage::assistant("It's 12:00 PM."),
                CoreMessage::user("Thanks!"),
            ]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content.as_ref().and_then(|c| c.as_text()), Some("12:00 PM"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[3].role, "assistant");
        assert_eq!(messages[3].content.as_ref().and_then(|c| c.as_text()), Some("It's 12:00 PM."));
        assert_eq!(messages[4].role, "user");
    }

    #[test]
    fn build_messages_with_attachments_single_turn() {
        use llm_core::types::{Attachment, AttachmentSource};

        let prompt = Prompt::new("Describe this image")
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Bytes(vec![0x89, 0x50, 0x4e, 0x47]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // Should be Parts, not Text
        match &messages[0].content {
            Some(MessageContent::Parts(parts)) => {
                // Text part first (OpenAI convention)
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    ContentPart::Text { text } => assert_eq!(text, "Describe this image"),
                    _ => panic!("expected Text part first"),
                }
                // Image part second
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                        assert!(image_url.url.len() > "data:image/png;base64,".len());
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("expected Parts content when attachments present"),
        }
    }

    #[test]
    fn build_messages_with_attachments_multi_turn() {
        use llm_core::types::{Attachment, AttachmentSource};
        use llm_core::Message as CoreMessage;

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("Hello"),
                CoreMessage::assistant("Hi! How can I help?"),
                CoreMessage::user("What is in this image?"),
            ])
            .with_attachments(vec![Attachment {
                mime_type: Some("image/jpeg".into()),
                source: AttachmentSource::Bytes(vec![0xFF, 0xD8, 0xFF]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 3);

        // First two messages unchanged
        assert_eq!(messages[0].content.as_ref().and_then(|c| c.as_text()), Some("Hello"));
        assert_eq!(messages[1].content.as_ref().and_then(|c| c.as_text()), Some("Hi! How can I help?"));

        // Last user message should have attachments injected
        assert_eq!(messages[2].role, "user");
        match &messages[2].content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                // Text first
                match &parts[0] {
                    ContentPart::Text { text } => assert_eq!(text, "What is in this image?"),
                    _ => panic!("expected Text part first"),
                }
                // Image second
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/jpeg;base64,"));
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("last user message should be Parts with attachments"),
        }
    }

    #[test]
    fn without_attachments_unchanged() {
        let prompt = Prompt::new("Hello");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_ref().and_then(|c| c.as_text()), Some("Hello"));
    }

    #[test]
    fn build_messages_path_attachment_not_found_returns_error() {
        use llm_core::types::{Attachment, AttachmentSource};

        // Single-turn: nonexistent path should error
        let prompt = Prompt::new("Describe this image")
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Path("/nonexistent/image.png".into()),
            }]);

        let result = build_messages(&prompt);
        assert!(result.is_err(), "expected Err for nonexistent path attachment (single-turn)");
        assert!(matches!(result.unwrap_err(), llm_core::LlmError::Io(_)));
    }

    #[test]
    fn build_messages_multi_turn_bad_path_attachment_returns_error() {
        use llm_core::types::{Attachment, AttachmentSource};
        use llm_core::Message as CoreMessage;

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("Hello"),
                CoreMessage::assistant("Hi!"),
                CoreMessage::user("What is this?"),
            ])
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Path("/nonexistent/image.png".into()),
            }]);

        let result = build_messages(&prompt);
        assert!(result.is_err(), "expected Err for nonexistent path attachment (multi-turn)");
        assert!(matches!(result.unwrap_err(), llm_core::LlmError::Io(_)));
    }

    #[test]
    fn build_messages_mixed_url_and_bad_path_returns_error() {
        use llm_core::types::{Attachment, AttachmentSource};

        let prompt = Prompt::new("Describe")
            .with_attachments(vec![
                Attachment {
                    mime_type: Some("image/png".into()),
                    source: AttachmentSource::Url("https://example.com/cat.jpg".into()),
                },
                Attachment {
                    mime_type: Some("image/png".into()),
                    source: AttachmentSource::Path("/nonexistent/bad.png".into()),
                },
            ]);

        let result = build_messages(&prompt);
        assert!(result.is_err(), "one bad path should fail the whole request");
    }

    #[test]
    fn empty_text_with_valid_attachment_produces_image_only_parts() {
        use llm_core::types::{Attachment, AttachmentSource};

        let prompt = Prompt::new("")
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Bytes(vec![0x89, 0x50, 0x4e, 0x47]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // Should be Parts with only an ImageUrl part (no Text part since text is empty)
        match &messages[0].content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 1, "expected exactly 1 image part, no text part");
                match &parts[0] {
                    ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("expected Parts content when attachments present"),
        }
    }

    #[test]
    fn empty_text_no_attachments_produces_text_not_parts() {
        let prompt = Prompt::new("");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // Should be Text(""), NOT Parts([])
        match &messages[0].content {
            Some(MessageContent::Text(t)) => {
                assert_eq!(t, "");
            }
            _ => panic!("expected Text content for empty text with no attachments, not Parts"),
        }
    }

    #[test]
    fn inject_empty_text_with_attachments_multi_turn() {
        use llm_core::types::{Attachment, AttachmentSource};
        use llm_core::Message as CoreMessage;

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("Hello"),
                CoreMessage::assistant("Hi!"),
                CoreMessage::user(""),
            ])
            .with_attachments(vec![Attachment {
                mime_type: Some("image/jpeg".into()),
                source: AttachmentSource::Bytes(vec![0xFF, 0xD8, 0xFF]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 3);

        // Last user message should have image part only, no text part
        assert_eq!(messages[2].role, "user");
        match &messages[2].content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 1, "expected 1 image part, no text part for empty text");
                match &parts[0] {
                    ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/jpeg;base64,"));
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("expected Parts content on last user message after injection"),
        }
    }

    #[test]
    fn build_messages_with_url_attachment() {
        use llm_core::types::{Attachment, AttachmentSource};

        let prompt = Prompt::new("What is this?")
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Url("https://example.com/cat.jpg".into()),
            }]);

        let messages = build_messages(&prompt).unwrap();
        match &messages[0].content {
            Some(MessageContent::Parts(parts)) => {
                // Text first
                match &parts[0] {
                    ContentPart::Text { text } => assert_eq!(text, "What is this?"),
                    _ => panic!("expected Text first"),
                }
                // URL passthrough (no data URI encoding)
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "https://example.com/cat.jpg");
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("expected Parts"),
        }
    }
}
