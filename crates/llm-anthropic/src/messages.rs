use crate::types::{ContentBlock, Message, MessageContent};
use llm_core::resolve_to_base64;
use llm_core::types::{Attachment, AttachmentSource};
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> Vec<Message> {
    if prompt.messages.is_empty() {
        build_single_turn(prompt)
    } else {
        build_from_conversation(prompt)
    }
}

/// Build a `MessageContent` for a user message, incorporating image attachments.
///
/// When attachments are present, produces `Blocks` with image blocks BEFORE the
/// text block (Anthropic convention: images first, text after).
/// When attachments are empty, produces `Text(text)`.
fn build_user_content(text: &str, attachments: &[Attachment]) -> MessageContent {
    if attachments.is_empty() {
        return MessageContent::Text(text.to_string());
    }

    let mut blocks: Vec<ContentBlock> = Vec::new();

    // Image blocks first (Anthropic convention)
    for att in attachments {
        match &att.source {
            AttachmentSource::Url(url) => {
                blocks.push(ContentBlock::image_url(url));
            }
            _ => {
                // Path or Bytes: resolve to base64
                if let Ok(resolved) = resolve_to_base64(att) {
                    blocks.push(ContentBlock::image_base64(
                        &resolved.media_type,
                        &resolved.base64_data,
                    ));
                }
                // If resolution fails (e.g. missing file), silently skip.
                // The error is non-fatal: the model still gets the text.
            }
        }
    }

    // Text block last
    if !text.is_empty() {
        blocks.push(ContentBlock {
            block_type: "text".into(),
            text: Some(text.to_string()),
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            source: None,
        });
    }

    MessageContent::Blocks(blocks)
}

fn build_single_turn(prompt: &Prompt) -> Vec<Message> {
    let content = build_user_content(&prompt.text, &prompt.attachments);

    let mut messages = vec![Message {
        role: "user".into(),
        content,
    }];

    // If there are tool calls and tool results, add assistant + user tool_result messages
    if !prompt.tool_calls.is_empty() && !prompt.tool_results.is_empty() {
        append_tool_exchange(&mut messages, &prompt.tool_calls, &prompt.tool_results);
    }

    messages
}

fn build_from_conversation(prompt: &Prompt) -> Vec<Message> {
    let mut messages = Vec::new();

    for msg in &prompt.messages {
        match msg.role {
            llm_core::Role::User => {
                messages.push(Message {
                    role: "user".into(),
                    content: MessageContent::Text(msg.content.clone()),
                });
            }
            llm_core::Role::Assistant => {
                if msg.tool_calls.is_empty() {
                    messages.push(Message {
                        role: "assistant".into(),
                        content: MessageContent::Text(msg.content.clone()),
                    });
                } else {
                    let mut blocks = Vec::new();
                    if !msg.content.is_empty() {
                        blocks.push(ContentBlock {
                            block_type: "text".into(),
                            text: Some(msg.content.clone()),
                            id: None,
                            name: None,
                            input: None,
                            tool_use_id: None,
                            content: None,
                            is_error: None,
                            source: None,
                        });
                    }
                    for tc in &msg.tool_calls {
                        blocks.push(map_tool_use(tc));
                    }
                    messages.push(Message {
                        role: "assistant".into(),
                        content: MessageContent::Blocks(blocks),
                    });
                }
            }
            llm_core::Role::Tool => {
                // Anthropic requires tool results in a "user" role message
                let blocks = msg
                    .tool_results
                    .iter()
                    .map(map_tool_result)
                    .collect();
                messages.push(Message {
                    role: "user".into(),
                    content: MessageContent::Blocks(blocks),
                });
            }
        }
    }

    // Inject attachments into the last user message
    if !prompt.attachments.is_empty() {
        inject_attachments_into_last_user_message(&mut messages, &prompt.attachments);
    }

    messages
}

/// Inject image attachments into the last user-role message in the conversation.
///
/// If the last user message was `Text(s)`, it becomes `Blocks([images..., text])`.
/// If it was already `Blocks(bs)`, image blocks are prepended.
fn inject_attachments_into_last_user_message(
    messages: &mut [Message],
    attachments: &[Attachment],
) {
    // Find the last user message
    let last_user = messages
        .iter_mut()
        .rev()
        .find(|m| m.role == "user");

    let Some(msg) = last_user else { return };

    let mut image_blocks: Vec<ContentBlock> = Vec::new();
    for att in attachments {
        match &att.source {
            AttachmentSource::Url(url) => {
                image_blocks.push(ContentBlock::image_url(url));
            }
            _ => {
                if let Ok(resolved) = resolve_to_base64(att) {
                    image_blocks.push(ContentBlock::image_base64(
                        &resolved.media_type,
                        &resolved.base64_data,
                    ));
                }
            }
        }
    }

    if image_blocks.is_empty() {
        return;
    }

    match &msg.content {
        MessageContent::Text(text) => {
            let mut blocks = image_blocks;
            if !text.is_empty() {
                blocks.push(ContentBlock {
                    block_type: "text".into(),
                    text: Some(text.clone()),
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: None,
                });
            }
            msg.content = MessageContent::Blocks(blocks);
        }
        MessageContent::Blocks(existing) => {
            image_blocks.extend(existing.iter().cloned());
            msg.content = MessageContent::Blocks(image_blocks);
        }
    }
}

fn map_tool_use(tc: &llm_core::ToolCall) -> ContentBlock {
    ContentBlock {
        block_type: "tool_use".into(),
        text: None,
        id: tc.tool_call_id.clone(),
        name: Some(tc.name.clone()),
        input: Some(tc.arguments.clone()),
        tool_use_id: None,
        content: None,
        is_error: None,
        source: None,
    }
}

fn map_tool_result(tr: &llm_core::ToolResult) -> ContentBlock {
    ContentBlock {
        block_type: "tool_result".into(),
        text: None,
        id: None,
        name: None,
        input: None,
        tool_use_id: tr.tool_call_id.clone(),
        content: Some(tr.output.clone()),
        is_error: tr.error.as_ref().map(|_| true),
        source: None,
    }
}

fn append_tool_exchange(
    messages: &mut Vec<Message>,
    tool_calls: &[llm_core::ToolCall],
    tool_results: &[llm_core::ToolResult],
) {
    let tool_use_blocks: Vec<ContentBlock> = tool_calls.iter().map(map_tool_use).collect();

    messages.push(Message {
        role: "assistant".into(),
        content: MessageContent::Blocks(tool_use_blocks),
    });

    let tool_result_blocks: Vec<ContentBlock> = tool_results.iter().map(map_tool_result).collect();

    messages.push(Message {
        role: "user".into(),
        content: MessageContent::Blocks(tool_result_blocks),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_user_only() {
        let prompt = Prompt::new("Hello");
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        if let MessageContent::Text(t) = &messages[0].content {
            assert_eq!(t, "Hello");
        } else {
            panic!("expected Text content");
        }
    }

    #[test]
    fn build_messages_with_system_does_not_add_system_message() {
        // Anthropic system prompt goes to top-level field, not in messages
        let prompt = Prompt::new("Hello").with_system("Be brief.");
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn build_messages_empty_system() {
        let prompt = Prompt::new("Hello").with_system("");
        let messages = build_messages(&prompt);
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
                tool_call_id: Some("toolu_1".into()),
            }])
            .with_tool_results(vec![ToolResult {
                name: "get_weather".into(),
                output: "Sunny, 22C".into(),
                tool_call_id: Some("toolu_1".into()),
                error: None,
            }]);

        let messages = build_messages(&prompt);
        // user(0) + assistant(1) + user tool_result(2) = 3
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        if let MessageContent::Blocks(blocks) = &messages[1].content {
            assert_eq!(blocks[0].block_type, "tool_use");
            assert_eq!(blocks[0].name.as_deref(), Some("get_weather"));
            assert_eq!(blocks[0].id.as_deref(), Some("toolu_1"));
        } else {
            panic!("expected Blocks content for assistant");
        }
        assert_eq!(messages[2].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[2].content {
            assert_eq!(blocks[0].block_type, "tool_result");
            assert_eq!(blocks[0].tool_use_id.as_deref(), Some("toolu_1"));
            assert_eq!(blocks[0].content.as_deref(), Some("Sunny, 22C"));
        } else {
            panic!("expected Blocks content for tool_result");
        }
    }

    #[test]
    fn build_messages_without_tool_results_unchanged() {
        let prompt = Prompt::new("Hello");
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn build_messages_multi_turn_conversation() {
        use llm_core::Message as CoreMessage;

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("Hello"),
                CoreMessage::assistant("Hi!"),
                CoreMessage::user("How are you?"),
            ]);

        let messages = build_messages(&prompt);
        // No system msg for Anthropic (it's top-level). 3 conversation messages.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        if let MessageContent::Text(t) = &messages[0].content {
            assert_eq!(t, "Hello");
        } else {
            panic!("expected Text");
        }
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[2].role, "user");
    }

    #[test]
    fn build_messages_multi_turn_with_tool_calls() {
        use llm_core::{Message as CoreMessage, ToolCall, ToolResult};

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("What time is it?"),
                CoreMessage::assistant_with_tool_calls(
                    "Let me check",
                    vec![ToolCall {
                        name: "get_time".into(),
                        arguments: serde_json::json!({}),
                        tool_call_id: Some("toolu_1".into()),
                    }],
                ),
                CoreMessage::tool_results(vec![ToolResult {
                    name: "get_time".into(),
                    output: "12:00 PM".into(),
                    tool_call_id: Some("toolu_1".into()),
                    error: None,
                }]),
                CoreMessage::assistant("It's 12:00 PM."),
                CoreMessage::user("Thanks!"),
            ]);

        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, "user");
        // Assistant with tool calls should have blocks
        assert_eq!(messages[1].role, "assistant");
        if let MessageContent::Blocks(blocks) = &messages[1].content {
            // text + tool_use
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].block_type, "text");
            assert_eq!(blocks[0].text.as_deref(), Some("Let me check"));
            assert_eq!(blocks[1].block_type, "tool_use");
            assert_eq!(blocks[1].name.as_deref(), Some("get_time"));
        } else {
            panic!("expected Blocks for assistant with tools");
        }
        // Tool results wrapped in user role
        assert_eq!(messages[2].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[2].content {
            assert_eq!(blocks[0].block_type, "tool_result");
            assert_eq!(blocks[0].tool_use_id.as_deref(), Some("toolu_1"));
        } else {
            panic!("expected Blocks for tool results");
        }
        assert_eq!(messages[3].role, "assistant");
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

        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // Should be Blocks, not Text
        if let MessageContent::Blocks(blocks) = &messages[0].content {
            // Image block first (Anthropic convention)
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].block_type, "image");
            let source = blocks[0].source.as_ref().unwrap();
            assert_eq!(source.source_type, "base64");
            assert_eq!(source.media_type.as_deref(), Some("image/png"));
            assert!(source.data.is_some());
            // Text block last
            assert_eq!(blocks[1].block_type, "text");
            assert_eq!(blocks[1].text.as_deref(), Some("Describe this image"));
        } else {
            panic!("expected Blocks content when attachments present");
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

        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 3);

        // First two messages unchanged
        if let MessageContent::Text(t) = &messages[0].content {
            assert_eq!(t, "Hello");
        } else {
            panic!("first message should be Text");
        }
        if let MessageContent::Text(t) = &messages[1].content {
            assert_eq!(t, "Hi! How can I help?");
        } else {
            panic!("second message should be Text");
        }

        // Last user message should have attachments injected
        assert_eq!(messages[2].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[2].content {
            assert_eq!(blocks.len(), 2);
            // Image first
            assert_eq!(blocks[0].block_type, "image");
            let source = blocks[0].source.as_ref().unwrap();
            assert_eq!(source.source_type, "base64");
            assert_eq!(source.media_type.as_deref(), Some("image/jpeg"));
            // Text last
            assert_eq!(blocks[1].block_type, "text");
            assert_eq!(blocks[1].text.as_deref(), Some("What is in this image?"));
        } else {
            panic!("last user message should be Blocks with attachments");
        }
    }

    #[test]
    fn without_attachments_unchanged() {
        // Verify that prompts without attachments produce the same output as before
        let prompt = Prompt::new("Hello");
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        if let MessageContent::Text(t) = &messages[0].content {
            assert_eq!(t, "Hello");
        } else {
            panic!("expected Text content when no attachments");
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

        let messages = build_messages(&prompt);
        if let MessageContent::Blocks(blocks) = &messages[0].content {
            assert_eq!(blocks[0].block_type, "image");
            let source = blocks[0].source.as_ref().unwrap();
            assert_eq!(source.source_type, "url");
            assert_eq!(source.url.as_deref(), Some("https://example.com/cat.jpg"));
            assert_eq!(blocks[1].block_type, "text");
        } else {
            panic!("expected Blocks");
        }
    }
}
