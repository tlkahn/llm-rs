use crate::types::{ContentBlock, Message, MessageContent};
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> Vec<Message> {
    if prompt.messages.is_empty() {
        build_single_turn(prompt)
    } else {
        build_from_conversation(prompt)
    }
}

fn build_single_turn(prompt: &Prompt) -> Vec<Message> {
    let mut messages = vec![Message {
        role: "user".into(),
        content: MessageContent::Text(prompt.text.clone()),
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

    messages
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
}
