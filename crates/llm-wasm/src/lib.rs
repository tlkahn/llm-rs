mod conversation;
mod tools;

use std::collections::HashMap;
use std::rc::Rc;

use async_trait::async_trait;
use futures::StreamExt;
use llm_anthropic::provider::AnthropicProvider;
use llm_core::stream::Chunk;
use llm_core::types::{Message, ModelInfo, Prompt};
use llm_core::{
    chain, multi_schema, parse_schema_dsl as core_parse_schema_dsl, ChainResult, ParallelConfig,
    Provider,
};
use llm_openai::provider::OpenAiProvider;
use serde_json::Value;
use wasm_bindgen::prelude::*;

pub use crate::conversation::Conversation;
use crate::tools::WasmToolRegistry;

const WASM_PKG_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_CHAIN_LIMIT: usize = 5;

enum ProviderImpl {
    OpenAi(OpenAiProvider),
    Anthropic(AnthropicProvider),
}

#[async_trait(?Send)]
impl Provider for ProviderImpl {
    fn id(&self) -> &str {
        match self {
            ProviderImpl::OpenAi(p) => p.id(),
            ProviderImpl::Anthropic(p) => p.id(),
        }
    }

    fn models(&self) -> Vec<ModelInfo> {
        match self {
            ProviderImpl::OpenAi(p) => p.models(),
            ProviderImpl::Anthropic(p) => p.models(),
        }
    }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> llm_core::Result<llm_core::stream::ResponseStream> {
        match self {
            ProviderImpl::OpenAi(p) => p.execute(model, prompt, key, stream).await,
            ProviderImpl::Anthropic(p) => p.execute(model, prompt, key, stream).await,
        }
    }
}

pub(crate) struct LlmClientInner {
    pub provider: ProviderImpl,
    pub model: String,
    pub api_key: String,
    pub registry: WasmToolRegistry,
    pub chain_limit: std::cell::Cell<usize>,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct LlmClient {
    pub(crate) inner: Rc<LlmClientInner>,
}

#[wasm_bindgen]
impl LlmClient {
    #[wasm_bindgen(constructor)]
    pub fn new(api_key: &str, model: &str) -> Self {
        if model.starts_with("claude") {
            Self::new_anthropic(api_key, model)
        } else {
            Self::new_with_base_url(api_key, model, "https://api.openai.com")
        }
    }

    #[wasm_bindgen(js_name = "newWithBaseUrl")]
    pub fn new_with_base_url(api_key: &str, model: &str, base_url: &str) -> Self {
        Self {
            inner: Rc::new(LlmClientInner {
                provider: ProviderImpl::OpenAi(OpenAiProvider::new(base_url)),
                model: model.to_string(),
                api_key: api_key.to_string(),
                registry: WasmToolRegistry::new(),
                chain_limit: std::cell::Cell::new(DEFAULT_CHAIN_LIMIT),
            }),
        }
    }

    #[wasm_bindgen(js_name = "newAnthropic")]
    pub fn new_anthropic(api_key: &str, model: &str) -> Self {
        Self::new_anthropic_with_base_url(api_key, model, "https://api.anthropic.com")
    }

    #[wasm_bindgen(js_name = "newAnthropicWithBaseUrl")]
    pub fn new_anthropic_with_base_url(api_key: &str, model: &str, base_url: &str) -> Self {
        Self {
            inner: Rc::new(LlmClientInner {
                provider: ProviderImpl::Anthropic(AnthropicProvider::new(base_url)),
                model: model.to_string(),
                api_key: api_key.to_string(),
                registry: WasmToolRegistry::new(),
                chain_limit: std::cell::Cell::new(DEFAULT_CHAIN_LIMIT),
            }),
        }
    }

    /// Register a tool from a JS object: `{ name, description, inputSchema, execute }`.
    /// `execute` may be sync or return a Promise.
    #[wasm_bindgen(js_name = "registerTool")]
    pub fn register_tool(&self, spec: JsValue) -> Result<(), JsError> {
        self.inner.registry.register(spec)
    }

    /// Enable the built-in tools (`llm_version`, `llm_time`).
    #[wasm_bindgen(js_name = "enableBuiltinTools")]
    pub fn enable_builtin_tools(&self) {
        self.inner.registry.enable_builtins(WASM_PKG_VERSION);
    }

    /// Set the maximum number of chain iterations (default 5).
    #[wasm_bindgen(js_name = "setChainLimit")]
    pub fn set_chain_limit(&self, limit: usize) {
        self.inner.chain_limit.set(limit);
    }

    /// Construct a fresh `Conversation` that shares this client's provider,
    /// model, key, and tool registry.
    pub fn conversation(&self, system: Option<String>) -> Conversation {
        Conversation::new(self, system)
    }

    /// Send a prompt and return the full response text.
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
        let prompt = self.build_prompt(text, system.as_deref(), None);
        self.run(prompt, false, None).await
    }

    /// Send a prompt with streaming. Calls the callback for each text chunk.
    #[wasm_bindgen(js_name = "promptStreaming")]
    pub async fn prompt_streaming(
        &self,
        text: &str,
        callback: js_sys::Function,
    ) -> Result<String, JsError> {
        self.prompt_streaming_with_system(text, None, callback).await
    }

    /// Send a prompt with system message and streaming.
    #[wasm_bindgen(js_name = "promptStreamingWithSystem")]
    pub async fn prompt_streaming_with_system(
        &self,
        text: &str,
        system: Option<String>,
        callback: js_sys::Function,
    ) -> Result<String, JsError> {
        let prompt = self.build_prompt(text, system.as_deref(), None);
        self.run(prompt, true, Some(callback)).await
    }

    /// Send a prompt with options (JSON string: `{"temperature": 0.7, ...}`).
    #[wasm_bindgen(js_name = "promptWithOptions")]
    pub async fn prompt_with_options(
        &self,
        text: &str,
        system: Option<String>,
        options_json: &str,
    ) -> Result<String, JsError> {
        let mut prompt = self.build_prompt(text, system.as_deref(), None);
        let options: HashMap<String, Value> =
            serde_json::from_str(options_json).map_err(|e| JsError::new(&e.to_string()))?;
        for (k, v) in options {
            prompt = prompt.with_option(&k, v);
        }
        self.run(prompt, false, None).await
    }

    /// Send a prompt with options and streaming.
    #[wasm_bindgen(js_name = "promptStreamingWithOptions")]
    pub async fn prompt_streaming_with_options(
        &self,
        text: &str,
        system: Option<String>,
        options_json: &str,
        callback: js_sys::Function,
    ) -> Result<String, JsError> {
        let mut prompt = self.build_prompt(text, system.as_deref(), None);
        let options: HashMap<String, Value> =
            serde_json::from_str(options_json).map_err(|e| JsError::new(&e.to_string()))?;
        for (k, v) in options {
            prompt = prompt.with_option(&k, v);
        }
        self.run(prompt, true, Some(callback)).await
    }

    /// Send a prompt with a JSON-Schema-validated structured output.
    /// `schema_or_dsl` accepts either a schema DSL string (`"name str, age int"`)
    /// or a JS object representing JSON Schema directly.
    #[wasm_bindgen(js_name = "promptWithSchema")]
    pub async fn prompt_with_schema(
        &self,
        text: &str,
        system: Option<String>,
        schema_or_dsl: JsValue,
        multi: bool,
    ) -> Result<String, JsError> {
        let schema = build_schema(schema_or_dsl, multi)?;
        let prompt = self.build_prompt(text, system.as_deref(), Some(schema));
        self.run(prompt, false, None).await
    }
}

impl LlmClient {
    fn build_prompt(
        &self,
        text: &str,
        system: Option<&str>,
        schema: Option<Value>,
    ) -> Prompt {
        let mut p = Prompt::new(text).with_tools(self.inner.registry.list_tools());
        if let Some(sys) = system {
            p = p.with_system(sys);
        }
        if let Some(s) = schema {
            p = p.with_schema(s);
        }
        p
    }

    async fn run(
        &self,
        prompt: Prompt,
        stream: bool,
        callback: Option<js_sys::Function>,
    ) -> Result<String, JsError> {
        if self.inner.registry.has_any() {
            let result = self.run_chain(prompt, stream, callback.as_ref()).await?;
            Ok(result)
        } else {
            self.run_direct(prompt, stream, callback.as_ref()).await
        }
    }

    async fn run_direct(
        &self,
        prompt: Prompt,
        stream: bool,
        callback: Option<&js_sys::Function>,
    ) -> Result<String, JsError> {
        let response = self
            .inner
            .provider
            .execute(&self.inner.model, &prompt, Some(&self.inner.api_key), stream)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let mut pinned = std::pin::pin!(response);
        let mut text = String::new();
        let this = JsValue::NULL;
        while let Some(item) = pinned.next().await {
            match item {
                Ok(Chunk::Text(t)) => {
                    if let Some(cb) = callback {
                        let _ = cb.call1(&this, &JsValue::from_str(&t));
                    }
                    text.push_str(&t);
                }
                Ok(Chunk::Done) => break,
                Err(e) => return Err(JsError::new(&e.to_string())),
                _ => {}
            }
        }
        Ok(text)
    }

    async fn run_chain(
        &self,
        prompt: Prompt,
        stream: bool,
        callback: Option<&js_sys::Function>,
    ) -> Result<String, JsError> {
        let cb = callback.cloned();
        let this = JsValue::NULL;
        let mut on_chunk = move |chunk: &Chunk| {
            if let (Chunk::Text(t), Some(c)) = (chunk, &cb) {
                let _ = c.call1(&this, &JsValue::from_str(t));
            }
        };
        let result = chain(
            &self.inner.provider,
            &self.inner.model,
            prompt,
            Some(&self.inner.api_key),
            stream,
            &self.inner.registry,
            self.inner.chain_limit.get(),
            &mut on_chunk,
            None,
            None,
            ParallelConfig::default(),
        )
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(llm_core::collect_text(&result.chunks))
    }

    /// Used by `Conversation`: run the chain seeded with `messages` and
    /// return (final_text, updated_messages).
    pub(crate) async fn send_messages(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        callback: Option<js_sys::Function>,
    ) -> Result<(String, Vec<Message>), JsError> {
        let mut p = Prompt::new("")
            .with_tools(self.inner.registry.list_tools())
            .with_messages(messages);
        if let Some(s) = system {
            p = p.with_system(&s);
        }

        let cb = callback;
        let this = JsValue::NULL;
        let mut on_chunk = move |chunk: &Chunk| {
            if let (Chunk::Text(t), Some(c)) = (chunk, &cb) {
                let _ = c.call1(&this, &JsValue::from_str(t));
            }
        };

        let result: ChainResult = chain(
            &self.inner.provider,
            &self.inner.model,
            p,
            Some(&self.inner.api_key),
            true,
            &self.inner.registry,
            self.inner.chain_limit.get(),
            &mut on_chunk,
            None,
            None,
            ParallelConfig::default(),
        )
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;

        let text = llm_core::collect_text(&result.chunks);
        Ok((text, result.messages))
    }
}

fn build_schema(js_value: JsValue, multi: bool) -> Result<Value, JsError> {
    let value = if let Some(s) = js_value.as_string() {
        core_parse_schema_dsl(&s).map_err(|e| JsError::new(&e.to_string()))?
    } else {
        serde_wasm_bindgen::from_value::<Value>(js_value)
            .map_err(|e| JsError::new(&e.to_string()))?
    };
    Ok(if multi { multi_schema(value) } else { value })
}

/// Parse a schema DSL string into a JSON Schema as a JS object.
#[wasm_bindgen(js_name = "parseSchemaDsl")]
pub fn parse_schema_dsl(dsl: &str) -> Result<JsValue, JsError> {
    let value = core_parse_schema_dsl(dsl).map_err(|e| JsError::new(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&value).map_err(|e| JsError::new(&e.to_string()))
}
