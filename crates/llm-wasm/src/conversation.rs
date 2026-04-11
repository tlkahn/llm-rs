use llm_core::types::{Message, Options, Response};
use wasm_bindgen::prelude::*;

use crate::store::build_response_wasm;
use crate::LlmClient;

/// A multi-turn conversation backed by an in-memory message history.
///
/// Construct via `LlmClient.conversation()` to share the client's registered
/// tools and provider. If the client has a `ConversationStore` attached, each
/// turn is appended to the log automatically.
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

    /// Load a persisted conversation by ID. Returns a new `Conversation`
    /// seeded with the reconstructed messages and the system prompt from
    /// the first stored response.
    #[wasm_bindgen(js_name = "load")]
    pub async fn load(client: &LlmClient, cid: &str) -> Result<Conversation, JsError> {
        let store = client
            .store()
            .ok_or_else(|| JsError::new("client has no conversation store attached"))?;
        let (_, responses) = llm_store::ConversationStore::read_conversation(&*store, cid)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let messages = reconstruct_messages_wasm(&responses);
        let system = responses.first().and_then(|r| r.system.clone());
        // Seed the client's active cid so future auto-logs append to this
        // conversation.
        client.set_conversation_id(Some(cid.to_string()));
        Ok(Self {
            client: client.clone(),
            messages,
            system,
        })
    }

    /// Send a user turn and return the assistant's reply text.
    pub async fn send(&mut self, text: &str) -> Result<String, JsError> {
        self.messages.push(Message::user(text));
        let messages = std::mem::take(&mut self.messages);
        let turn = self
            .client
            .send_messages(messages, self.system.clone(), None)
            .await?;
        self.messages = turn.messages.clone();
        self.log_turn(text, &turn).await?;
        Ok(turn.text)
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
        let turn = self
            .client
            .send_messages(messages, self.system.clone(), Some(callback))
            .await?;
        self.messages = turn.messages.clone();
        self.log_turn(text, &turn).await?;
        Ok(turn.text)
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

impl Conversation {
    async fn log_turn(&self, user_text: &str, turn: &crate::WasmTurnOutput) -> Result<(), JsError> {
        if self.client.store().is_none() {
            return Ok(());
        }
        let response: Response = build_response_wasm(
            self.client.model(),
            user_text,
            self.system.as_deref(),
            Options::new(),
            turn.text.clone(),
            turn.total_usage.clone(),
            turn.tool_calls.clone(),
            turn.tool_results.clone(),
            None,
            None,
            turn.duration_ms,
        );
        self.client.log_external(&response).await
    }
}

/// wasm32-local copy of `llm_store::reconstruct_messages` — the original
/// lives in the native-only `logs` module.
fn reconstruct_messages_wasm(responses: &[Response]) -> Vec<Message> {
    let mut messages = Vec::new();
    for response in responses {
        messages.push(Message::user(&response.prompt));
        if response.tool_calls.is_empty() {
            messages.push(Message::assistant(&response.response));
        } else {
            messages.push(Message::assistant_with_tool_calls(
                &response.response,
                response.tool_calls.clone(),
            ));
            if !response.tool_results.is_empty() {
                messages.push(Message::tool_results(response.tool_results.clone()));
            }
        }
    }
    messages
}
