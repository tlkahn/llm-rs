use crate::types::Message;
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> Vec<Message> {
    let mut messages = Vec::new();

    if let Some(system) = &prompt.system {
        if !system.is_empty() {
            messages.push(Message {
                role: "system".into(),
                content: Some(system.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    messages.push(Message {
        role: "user".into(),
        content: Some(prompt.text.clone()),
        tool_calls: None,
        tool_call_id: None,
    });

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
}
