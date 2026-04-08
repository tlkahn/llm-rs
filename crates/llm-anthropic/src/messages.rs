use crate::types::{ContentBlock, Message, MessageContent};
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> Vec<Message> {
    let mut messages = vec![Message {
        role: "user".into(),
        content: MessageContent::Text(prompt.text.clone()),
    }];

    // If there are tool calls and tool results, add assistant + user tool_result messages
    if !prompt.tool_calls.is_empty() && !prompt.tool_results.is_empty() {
        let tool_use_blocks: Vec<ContentBlock> = prompt
            .tool_calls
            .iter()
            .map(|tc| ContentBlock {
                block_type: "tool_use".into(),
                text: None,
                id: tc.tool_call_id.clone(),
                name: Some(tc.name.clone()),
                input: Some(tc.arguments.clone()),
                tool_use_id: None,
                content: None,
                is_error: None,
            })
            .collect();

        messages.push(Message {
            role: "assistant".into(),
            content: MessageContent::Blocks(tool_use_blocks),
        });

        let tool_result_blocks: Vec<ContentBlock> = prompt
            .tool_results
            .iter()
            .map(|tr| ContentBlock {
                block_type: "tool_result".into(),
                text: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: tr.tool_call_id.clone(),
                content: Some(tr.output.clone()),
                is_error: tr.error.as_ref().map(|_| true),
            })
            .collect();

        messages.push(Message {
            role: "user".into(),
            content: MessageContent::Blocks(tool_result_blocks),
        });
    }

    messages
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
}
