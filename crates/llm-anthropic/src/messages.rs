use crate::types::{ContentBlock, Message, MessageContent};
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

/// Map resolved image blocks to Anthropic `ContentBlock`s.
fn resolved_to_blocks(resolved: Vec<ResolvedImageBlock>) -> Vec<ContentBlock> {
    resolved
        .into_iter()
        .map(|r| match r {
            ResolvedImageBlock::Base64 { media_type, base64_data } => {
                ContentBlock::image_base64(media_type, base64_data)
            }
            ResolvedImageBlock::Url { url } => ContentBlock::image_url(url),
        })
        .collect()
}

/// Build a `MessageContent` for a user message, incorporating image attachments.
///
/// When attachments are present, produces `Blocks` with image blocks BEFORE the
/// text block (Anthropic convention: images first, text after).
/// When attachments are empty, produces `Text(text)`.
fn build_user_content(text: &str, attachments: &[Attachment]) -> llm_core::Result<MessageContent> {
    if attachments.is_empty() {
        return Ok(MessageContent::Text(text.to_string()));
    }

    let resolved = resolve_attachments(attachments)?;
    // Image blocks first (Anthropic convention)
    let mut blocks = resolved_to_blocks(resolved);

    // Text block last
    if !text.is_empty() {
        blocks.push(ContentBlock::text(text));
    }

    // Guard: Anthropic rejects an empty content array with 400.
    if blocks.is_empty() {
        return Err(llm_core::LlmError::Provider(
            "empty content: no text and no image blocks were produced from attachments".into(),
        ));
    }

    Ok(MessageContent::Blocks(blocks))
}

fn build_single_turn(prompt: &Prompt) -> llm_core::Result<Vec<Message>> {
    let content = build_user_content(&prompt.text, &prompt.attachments)?;

    let mut messages = vec![Message {
        role: "user".into(),
        content,
    }];

    // If there are tool calls and tool results, add assistant + user tool_result messages
    if !prompt.tool_calls.is_empty() && !prompt.tool_results.is_empty() {
        append_tool_exchange(&mut messages, &prompt.tool_calls, &prompt.tool_results);
    }

    Ok(messages)
}

fn build_from_conversation(prompt: &Prompt) -> llm_core::Result<Vec<Message>> {
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
                        blocks.push(ContentBlock::text(&msg.content));
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
        inject_attachments_into_last_user_message(&mut messages, &prompt.attachments)?;
    }

    Ok(messages)
}

/// Returns `true` if the message contains tool_result content blocks.
///
/// In Anthropic's wire format, tool results are sent as `role: "user"` messages
/// with `tool_result` blocks. We must distinguish these from genuine user-text
/// messages when injecting image attachments.
fn is_tool_result_message(msg: &Message) -> bool {
    match &msg.content {
        MessageContent::Blocks(blocks) => blocks.iter().any(|b| b.block_type == "tool_result"),
        MessageContent::Text(_) => false,
    }
}

/// Inject image attachments into the last user-role message in the conversation.
///
/// If the last user message was `Text(s)`, it becomes `Blocks([images..., text])`.
/// If it was already `Blocks(bs)`, image blocks are prepended.
fn inject_attachments_into_last_user_message(
    messages: &mut [Message],
    attachments: &[Attachment],
) -> llm_core::Result<()> {
    // Find the last genuine user-text message, skipping tool_result containers
    // which also have role "user" in Anthropic's wire format.
    let last_user = messages
        .iter_mut()
        .rev()
        .find(|m| m.role == "user" && !is_tool_result_message(m));

    let Some(msg) = last_user else { return Ok(()) };

    let resolved = resolve_attachments(attachments)?;
    let image_blocks = resolved_to_blocks(resolved);

    if image_blocks.is_empty() {
        return Ok(());
    }

    // Move existing content out to avoid cloning large payloads.
    match std::mem::replace(&mut msg.content, MessageContent::Text(String::new())) {
        MessageContent::Text(text) => {
            let mut blocks = image_blocks;
            if !text.is_empty() {
                blocks.push(ContentBlock::text(text));
            }
            // Guard: Anthropic rejects an empty content array with 400.
            if blocks.is_empty() {
                return Err(llm_core::LlmError::Provider(
                    "empty content: no text and no image blocks were produced from attachments"
                        .into(),
                ));
            }
            msg.content = MessageContent::Blocks(blocks);
        }
        MessageContent::Blocks(mut existing) => {
            let mut all_blocks = image_blocks;
            all_blocks.append(&mut existing);
            msg.content = MessageContent::Blocks(all_blocks);
        }
    }

    Ok(())
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
        let messages = build_messages(&prompt).unwrap();
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
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn build_messages_empty_system() {
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
                tool_call_id: Some("toolu_1".into()),
            }])
            .with_tool_results(vec![ToolResult {
                name: "get_weather".into(),
                output: "Sunny, 22C".into(),
                tool_call_id: Some("toolu_1".into()),
                error: None,
            }]);

        let messages = build_messages(&prompt).unwrap();
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
        let messages = build_messages(&prompt).unwrap();
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

        let messages = build_messages(&prompt).unwrap();
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

        let messages = build_messages(&prompt).unwrap();
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

        let messages = build_messages(&prompt).unwrap();
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

        let messages = build_messages(&prompt).unwrap();
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
        let messages = build_messages(&prompt).unwrap();
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

        let messages = build_messages(&prompt).unwrap();
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

    #[test]
    fn is_tool_result_message_identifies_correctly() {
        // tool_result message -> true
        let tr_msg = Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlock {
                block_type: "tool_result".into(),
                text: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: Some("toolu_1".into()),
                content: Some("result".into()),
                is_error: None,
                source: None,
            }]),
        };
        assert!(is_tool_result_message(&tr_msg));

        // Text message -> false
        let text_msg = Message {
            role: "user".into(),
            content: MessageContent::Text("hello".into()),
        };
        assert!(!is_tool_result_message(&text_msg));

        // Blocks with only text blocks -> false
        let text_blocks_msg = Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlock {
                block_type: "text".into(),
                text: Some("hello".into()),
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                source: None,
            }]),
        };
        assert!(!is_tool_result_message(&text_blocks_msg));
    }

    #[test]
    fn attachments_skip_tool_result_message() {
        use llm_core::types::{Attachment, AttachmentSource};
        use llm_core::{Message as CoreMessage, ToolCall, ToolResult};

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("Describe this image"),
                CoreMessage::assistant_with_tool_calls(
                    "Let me use a tool first",
                    vec![ToolCall {
                        name: "analyze".into(),
                        arguments: serde_json::json!({}),
                        tool_call_id: Some("toolu_1".into()),
                    }],
                ),
                CoreMessage::tool_results(vec![ToolResult {
                    name: "analyze".into(),
                    output: "analysis done".into(),
                    tool_call_id: Some("toolu_1".into()),
                    error: None,
                }]),
            ])
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Bytes(vec![0x89, 0x50, 0x4e, 0x47]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        // messages[0] = user "Describe this image"
        // messages[1] = assistant with tool_use
        // messages[2] = user with tool_result blocks
        assert_eq!(messages.len(), 3);

        // The tool_result message (messages[2]) should NOT contain image blocks
        assert_eq!(messages[2].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[2].content {
            for block in blocks {
                assert_eq!(
                    block.block_type, "tool_result",
                    "tool_result message should only contain tool_result blocks, found: {}",
                    block.block_type
                );
            }
        } else {
            panic!("tool_result message should be Blocks");
        }

        // The genuine user message (messages[0]) SHOULD contain the injected image
        assert_eq!(messages[0].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[0].content {
            assert_eq!(blocks.len(), 2); // image + text
            assert_eq!(blocks[0].block_type, "image");
            assert_eq!(blocks[1].block_type, "text");
            assert_eq!(blocks[1].text.as_deref(), Some("Describe this image"));
        } else {
            panic!("genuine user message should have been upgraded to Blocks with image");
        }
    }

    #[test]
    fn attachments_no_genuine_user_message_skips_injection() {
        use llm_core::types::{Attachment, AttachmentSource};

        // Construct Anthropic wire-format messages directly to simulate a
        // conversation with only tool_result "user" messages.
        let mut messages = vec![Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlock {
                block_type: "tool_result".into(),
                text: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: Some("toolu_1".into()),
                content: Some("result".into()),
                is_error: None,
                source: None,
            }]),
        }];

        let attachments = vec![Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Bytes(vec![0x89, 0x50, 0x4e, 0x47]),
        }];

        inject_attachments_into_last_user_message(&mut messages, &attachments).unwrap();

        // Message should be unchanged -- still only a tool_result block
        if let MessageContent::Blocks(blocks) = &messages[0].content {
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].block_type, "tool_result");
        } else {
            panic!("message should remain Blocks with tool_result");
        }
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
    fn empty_text_with_valid_attachment_produces_image_only_blocks() {
        use llm_core::types::{Attachment, AttachmentSource};

        let prompt = Prompt::new("")
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Bytes(vec![0x89, 0x50, 0x4e, 0x47]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // Should be Blocks with only image block (no text block since text is empty)
        if let MessageContent::Blocks(blocks) = &messages[0].content {
            assert_eq!(blocks.len(), 1, "expected exactly 1 image block, no text block");
            assert_eq!(blocks[0].block_type, "image");
            let source = blocks[0].source.as_ref().unwrap();
            assert_eq!(source.source_type, "base64");
            assert_eq!(source.media_type.as_deref(), Some("image/png"));
        } else {
            panic!("expected Blocks content when attachments present");
        }
    }

    #[test]
    fn empty_text_no_attachments_produces_text_not_blocks() {
        let prompt = Prompt::new("");
        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // Should be Text(""), NOT Blocks([])
        if let MessageContent::Text(t) = &messages[0].content {
            assert_eq!(t, "");
        } else {
            panic!("expected Text content for empty text with no attachments, not Blocks");
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
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Bytes(vec![0x89, 0x50, 0x4e, 0x47]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        assert_eq!(messages.len(), 3);

        // Last user message should have image block only, no text block
        assert_eq!(messages[2].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[2].content {
            assert_eq!(blocks.len(), 1, "expected 1 image block, no text block for empty text");
            assert_eq!(blocks[0].block_type, "image");
        } else {
            panic!("expected Blocks content on last user message after injection");
        }
    }

    #[test]
    fn attachments_with_multiple_tool_rounds_finds_correct_user_message() {
        use llm_core::types::{Attachment, AttachmentSource};
        use llm_core::{Message as CoreMessage, ToolCall, ToolResult};

        let prompt = Prompt::new("")
            .with_messages(vec![
                CoreMessage::user("First question"),
                CoreMessage::assistant("Answer 1"),
                CoreMessage::user("Analyze this with tools"),
                CoreMessage::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        name: "tool_a".into(),
                        arguments: serde_json::json!({}),
                        tool_call_id: Some("toolu_1".into()),
                    }],
                ),
                CoreMessage::tool_results(vec![ToolResult {
                    name: "tool_a".into(),
                    output: "result_a".into(),
                    tool_call_id: Some("toolu_1".into()),
                    error: None,
                }]),
                CoreMessage::assistant_with_tool_calls(
                    "Need another tool",
                    vec![ToolCall {
                        name: "tool_b".into(),
                        arguments: serde_json::json!({}),
                        tool_call_id: Some("toolu_2".into()),
                    }],
                ),
                CoreMessage::tool_results(vec![ToolResult {
                    name: "tool_b".into(),
                    output: "result_b".into(),
                    tool_call_id: Some("toolu_2".into()),
                    error: None,
                }]),
            ])
            .with_attachments(vec![Attachment {
                mime_type: Some("image/jpeg".into()),
                source: AttachmentSource::Bytes(vec![0xFF, 0xD8, 0xFF]),
            }]);

        let messages = build_messages(&prompt).unwrap();
        // Wire: user(0), assistant(1), user(2="Analyze..."), assistant(3), user(4=tool_result),
        //       assistant(5), user(6=tool_result)
        assert_eq!(messages.len(), 7);

        // messages[2] is "Analyze this with tools" -- should get the image
        assert_eq!(messages[2].role, "user");
        if let MessageContent::Blocks(blocks) = &messages[2].content {
            assert!(
                blocks.iter().any(|b| b.block_type == "image"),
                "genuine user message should have image injected"
            );
            assert!(
                blocks.iter().any(|b| b.block_type == "text"
                    && b.text.as_deref() == Some("Analyze this with tools")),
                "genuine user message should retain its text"
            );
        } else {
            panic!("expected Blocks after injection");
        }

        // messages[4] and messages[6] are tool_result -- should NOT have images
        for idx in [4, 6] {
            if let MessageContent::Blocks(blocks) = &messages[idx].content {
                assert!(
                    blocks.iter().all(|b| b.block_type == "tool_result"),
                    "tool_result message at index {} should not contain image blocks",
                    idx
                );
            }
        }
    }
}
