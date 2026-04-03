use futures::StreamExt;
use llm_core::stream::Chunk;
use llm_core::types::Prompt;
use llm_core::Provider;
use llm_openai::provider::OpenAiProvider;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LlmClient {
    provider: OpenAiProvider,
    model: String,
    api_key: String,
}

#[wasm_bindgen]
impl LlmClient {
    #[wasm_bindgen(constructor)]
    pub fn new(api_key: &str, model: &str) -> Self {
        Self::new_with_base_url(api_key, model, "https://api.openai.com")
    }

    #[wasm_bindgen(js_name = "newWithBaseUrl")]
    pub fn new_with_base_url(api_key: &str, model: &str, base_url: &str) -> Self {
        Self {
            provider: OpenAiProvider::new(base_url),
            model: model.to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Send a prompt and return the full response text.
    /// Returns a JS Promise that resolves to a string.
    pub async fn prompt(&self, text: &str) -> Result<String, JsError> {
        self.prompt_with_system(text, None).await
    }

    /// Send a prompt with a system message and return the full response text.
    #[wasm_bindgen(js_name = "promptWithSystem")]
    pub async fn prompt_with_system(
        &self,
        text: &str,
        system: Option<String>,
    ) -> Result<String, JsError> {
        let mut p = Prompt::new(text);
        if let Some(sys) = system {
            p = p.with_system(&sys);
        }

        let stream = self
            .provider
            .execute(&self.model, &p, Some(&self.api_key), false)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let mut stream = std::pin::pin!(stream);
        let mut text = String::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(Chunk::Text(t)) => text.push_str(&t),
                Ok(Chunk::Done) => break,
                Err(e) => return Err(JsError::new(&e.to_string())),
                _ => {}
            }
        }
        Ok(text)
    }

    /// Send a prompt with streaming. Calls the callback for each text chunk.
    /// Returns the full concatenated response text.
    #[wasm_bindgen(js_name = "promptStreaming")]
    pub async fn prompt_streaming(
        &self,
        text: &str,
        callback: &js_sys::Function,
    ) -> Result<String, JsError> {
        self.prompt_streaming_with_system(text, None, callback).await
    }

    /// Send a prompt with system message and streaming.
    #[wasm_bindgen(js_name = "promptStreamingWithSystem")]
    pub async fn prompt_streaming_with_system(
        &self,
        text: &str,
        system: Option<String>,
        callback: &js_sys::Function,
    ) -> Result<String, JsError> {
        let mut p = Prompt::new(text);
        if let Some(sys) = system {
            p = p.with_system(&sys);
        }

        let stream = self
            .provider
            .execute(&self.model, &p, Some(&self.api_key), true)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let mut stream = std::pin::pin!(stream);
        let mut full_text = String::new();
        let this = JsValue::NULL;

        while let Some(result) = stream.next().await {
            match result {
                Ok(Chunk::Text(t)) => {
                    let _ = callback.call1(&this, &JsValue::from_str(&t));
                    full_text.push_str(&t);
                }
                Ok(Chunk::Done) => break,
                Err(e) => return Err(JsError::new(&e.to_string())),
                _ => {}
            }
        }
        Ok(full_text)
    }

    /// Send a prompt with options (JSON string: {"temperature": 0.7, "max_tokens": 1000}).
    #[wasm_bindgen(js_name = "promptWithOptions")]
    pub async fn prompt_with_options(
        &self,
        text: &str,
        system: Option<String>,
        options_json: &str,
    ) -> Result<String, JsError> {
        let mut p = Prompt::new(text);
        if let Some(sys) = system {
            p = p.with_system(&sys);
        }
        let options: HashMap<String, serde_json::Value> =
            serde_json::from_str(options_json).map_err(|e| JsError::new(&e.to_string()))?;
        for (k, v) in options {
            p = p.with_option(&k, v);
        }

        let stream = self
            .provider
            .execute(&self.model, &p, Some(&self.api_key), false)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let mut stream = std::pin::pin!(stream);
        let mut text = String::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(Chunk::Text(t)) => text.push_str(&t),
                Ok(Chunk::Done) => break,
                Err(e) => return Err(JsError::new(&e.to_string())),
                _ => {}
            }
        }
        Ok(text)
    }

    /// Send a prompt with options and streaming.
    #[wasm_bindgen(js_name = "promptStreamingWithOptions")]
    pub async fn prompt_streaming_with_options(
        &self,
        text: &str,
        system: Option<String>,
        options_json: &str,
        callback: &js_sys::Function,
    ) -> Result<String, JsError> {
        let mut p = Prompt::new(text);
        if let Some(sys) = system {
            p = p.with_system(&sys);
        }
        let options: HashMap<String, serde_json::Value> =
            serde_json::from_str(options_json).map_err(|e| JsError::new(&e.to_string()))?;
        for (k, v) in options {
            p = p.with_option(&k, v);
        }

        let stream = self
            .provider
            .execute(&self.model, &p, Some(&self.api_key), true)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let mut stream = std::pin::pin!(stream);
        let mut full_text = String::new();
        let this = JsValue::NULL;

        while let Some(result) = stream.next().await {
            match result {
                Ok(Chunk::Text(t)) => {
                    let _ = callback.call1(&this, &JsValue::from_str(&t));
                    full_text.push_str(&t);
                }
                Ok(Chunk::Done) => break,
                Err(e) => return Err(JsError::new(&e.to_string())),
                _ => {}
            }
        }
        Ok(full_text)
    }
}
