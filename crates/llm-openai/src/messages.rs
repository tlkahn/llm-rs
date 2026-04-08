use crate::types::{Message, MessageToolCall, MessageToolCallFunction};
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> Vec<Message> {
    if prompt.messages.is_empty() {
        build_single_turn(prompt)
    } else {
        build_from_conversation(prompt)
    }
}

fn build_single_turn(prompt: &Prompt) -> Vec<Message> {
    let mut messages = Vec::new();

    if let Some(system) = &prompt.system
        && !system.is_empty()
    {
        messages.push(Message {
            role: "system".into(),
            content: Some(system.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages.push(Message {
        role: "user".into(),
        content: Some(prompt.text.clone()),
        tool_calls: None,
        tool_call_id: None,
    });

    // If there are tool calls and tool results, add assistant + tool messages
    if !prompt.tool_calls.is_empty() && !prompt.tool_results.is_empty() {
        append_tool_exchange(&mut messages, &prompt.tool_calls, &prompt.tool_results);
    }

    messages
}

fn build_from_conversation(prompt: &Prompt) -> Vec<Message> {
    let mut messages = Vec::new();

    // System prompt as first message
    if let Some(system) = &prompt.system
        && !system.is_empty()
    {
        messages.push(Message {
            role: "system".into(),
            content: Some(system.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for msg in &prompt.messages {
        match msg.role {
            llm_core::Role::User => {
                messages.push(Message {
                    role: "user".into(),
                    content: Some(msg.content.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            llm_core::Role::Assistant => {
                if msg.tool_calls.is_empty() {
                    messages.push(Message {
                        role: "assistant".into(),
                        content: Some(msg.content.clone()),
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
                            Some(msg.content.clone())
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
                        content: Some(result.output.clone()),
                        tool_calls: None,
                        tool_call_id: result.tool_call_id.clone(),
                    });
                }
            }
        }
    }

    messages
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
            content: Some(result.output.clone()),
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
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn build_messages_with_system() {
        let prompt = Prompt::new("Hello").with_system("Be brief.");
        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content.as_deref(), Some("Be brief."));
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn build_messages_empty_system_is_skipped() {
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
                tool_call_id: Some("call_1".into()),
            }])
            .with_tool_results(vec![ToolResult {
                name: "get_weather".into(),
                output: "Sunny, 22C".into(),
                tool_call_id: Some("call_1".into()),
                error: None,
            }]);

        let messages = build_messages(&prompt);
        // system(0) + user(1) + assistant(2) + tool(3) = 3 messages (no system)
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].tool_calls.is_some());
        let tcs = messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].function.name, "get_weather");
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content.as_deref(), Some("Sunny, 22C"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
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
            .with_system("Be helpful")
            .with_messages(vec![
                CoreMessage::user("Hello"),
                CoreMessage::assistant("Hi!"),
                CoreMessage::user("How are you?"),
            ]);

        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 4); // system + 3 conversation
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("Hello"));
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].content.as_deref(), Some("Hi!"));
        assert_eq!(messages[3].role, "user");
        assert_eq!(messages[3].content.as_deref(), Some("How are you?"));
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

        let messages = build_messages(&prompt);
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content.as_deref(), Some("12:00 PM"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[3].role, "assistant");
        assert_eq!(messages[3].content.as_deref(), Some("It's 12:00 PM."));
        assert_eq!(messages[4].role, "user");
    }
}
