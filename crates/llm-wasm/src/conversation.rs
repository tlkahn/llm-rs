use llm_core::types::Message;
use wasm_bindgen::prelude::*;

use crate::LlmClient;

/// A multi-turn conversation backed by an in-memory message history.
///
/// Construct via `LlmClient.conversation()` to share the client's registered
/// tools and provider.
#[wasm_bindgen]
pub struct Conversation {
    client: LlmClient,
    messages: Vec<Message>,
    system: Option<String>,
}

#[wasm_bindgen]
impl Conversation {
    /// Create a fresh conversation. Prefer `client.conversation()` from JS.
    #[wasm_bindgen(constructor)]
    pub fn new(client: &LlmClient, system: Option<String>) -> Self {
        Self {
            client: client.clone(),
            messages: Vec::new(),
            system,
        }
    }

    /// Send a user turn and return the assistant's reply text.
    pub async fn send(&mut self, text: &str) -> Result<String, JsError> {
        self.messages.push(Message::user(text));
        let messages = std::mem::take(&mut self.messages);
        let (reply, updated) = self
            .client
            .send_messages(messages, self.system.clone(), None)
            .await?;
        self.messages = updated;
        Ok(reply)
    }

    /// Send a user turn and stream text chunks via the provided callback.
    /// Returns the full assistant reply text.
    #[wasm_bindgen(js_name = "sendStreaming")]
    pub async fn send_streaming(
        &mut self,
        text: &str,
        callback: js_sys::Function,
    ) -> Result<String, JsError> {
        self.messages.push(Message::user(text));
        let messages = std::mem::take(&mut self.messages);
        let (reply, updated) = self
            .client
            .send_messages(messages, self.system.clone(), Some(callback))
            .await?;
        self.messages = updated;
        Ok(reply)
    }

    /// Return the current message history as a JS-compatible value.
    #[wasm_bindgen(getter)]
    pub fn messages(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&self.messages).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Clear the message history.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Number of messages currently in history.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.messages.len()
    }
}
