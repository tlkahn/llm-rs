use crate::types::{Message, MessageContent};
use llm_core::Prompt;

pub fn build_messages(prompt: &Prompt) -> Vec<Message> {
    vec![Message {
        role: "user".into(),
        content: MessageContent::Text(prompt.text.clone()),
    }]
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
}
