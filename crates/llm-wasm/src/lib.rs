mod conversation;
mod retry;
mod tools;

use std::collections::HashMap;
use std::rc::Rc;

use async_trait::async_trait;
use futures::StreamExt;
use llm_anthropic::provider::AnthropicProvider;
use llm_core::retry::RetryConfig;
use llm_core::stream::Chunk;
use llm_core::types::{Message, ModelInfo, Prompt};
use llm_core::{
    chain, multi_schema, parse_schema_dsl as core_parse_schema_dsl, ChainEvent, ChainResult,
    ParallelConfig, Provider,
};
use llm_openai::provider::OpenAiProvider;
use serde_json::Value;
use wasm_bindgen::prelude::*;

pub use crate::conversation::Conversation;
use crate::retry::RetryProvider;
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
    pub retry_config: std::cell::RefCell<RetryConfig>,
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
                retry_config: std::cell::RefCell::new(RetryConfig {
                    max_retries: 0,
                    base_delay_ms: 1000,
                    max_delay_ms: 30_000,
                    jitter: true,
                }),
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
                retry_config: std::cell::RefCell::new(RetryConfig {
                    max_retries: 0,
                    base_delay_ms: 1000,
                    max_delay_ms: 30_000,
                    jitter: true,
                }),
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

    /// Configure retry policy for transient (429 / 5xx) errors.
    /// Set `max_retries` to 0 to disable retries entirely.
    #[wasm_bindgen(js_name = "setRetryConfig")]
    pub fn set_retry_config(
        &self,
        max_retries: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
        jitter: bool,
    ) {
        *self.inner.retry_config.borrow_mut() = RetryConfig {
            max_retries,
            base_delay_ms,
            max_delay_ms,
            jitter,
        };
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

    /// Run a chain loop with optional observability and budget enforcement.
    ///
    /// `options` is an optional JS object:
    /// `{ system?: string, chainLimit?: number, budget?: number, onEvent?: (evt) => void }`.
    ///
    /// Returns a `ChainResult`-shaped JS object:
    /// `{ text, toolCalls, totalUsage, budgetExhausted }`. Event dicts are
    /// type-tagged: `iteration_start`, `iteration_end`, `budget_exhausted`.
    #[wasm_bindgen(js_name = "chain")]
    pub async fn chain_js(
        &self,
        text: &str,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let opts = ChainOptions::parse(&options)?;
        let mut prompt = Prompt::new(text).with_tools(self.inner.registry.list_tools());
        if let Some(s) = &opts.system {
            prompt = prompt.with_system(s);
        }
        let limit = opts.chain_limit.unwrap_or(self.inner.chain_limit.get());

        let on_event_cb = opts.on_event.clone();
        let this = JsValue::NULL;
        let mut on_chunk = |_: &Chunk| {};
        let mut on_event = move |ev: &ChainEvent| {
            if let Some(cb) = &on_event_cb {
                let v = chain_event_to_value(ev);
                if let Ok(obj) = serde_wasm_bindgen::to_value(&v) {
                    let _ = cb.call1(&this, &obj);
                }
            }
        };

        let retry_cfg = self.inner.retry_config.borrow().clone();
        let retry = RetryProvider::new(&self.inner.provider, retry_cfg);
        let result = chain(
            &retry,
            &self.inner.model,
            prompt,
            Some(&self.inner.api_key),
            false,
            &self.inner.registry,
            limit,
            &mut on_chunk,
            Some(&mut on_event),
            opts.budget,
            ParallelConfig::default(),
        )
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;

        build_chain_result_js(result)
    }

    /// Streaming variant of `chain`. Calls the provided `callback` with each
    /// text chunk (`{type: 'text', content}`) and each chain event, interleaved
    /// in order.
    #[wasm_bindgen(js_name = "chainStreaming")]
    pub async fn chain_streaming(
        &self,
        text: &str,
        callback: js_sys::Function,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let opts = ChainOptions::parse(&options)?;
        let mut prompt = Prompt::new(text).with_tools(self.inner.registry.list_tools());
        if let Some(s) = &opts.system {
            prompt = prompt.with_system(s);
        }
        let limit = opts.chain_limit.unwrap_or(self.inner.chain_limit.get());

        let cb_text = callback.clone();
        let this_text = JsValue::NULL;
        let mut on_chunk = move |chunk: &Chunk| {
            if let Chunk::Text(t) = chunk {
                let v = serde_json::json!({"type": "text", "content": t});
                if let Ok(obj) = serde_wasm_bindgen::to_value(&v) {
                    let _ = cb_text.call1(&this_text, &obj);
                }
            }
        };
        let cb_evt = callback.clone();
        let this_evt = JsValue::NULL;
        let mut on_event = move |ev: &ChainEvent| {
            let v = chain_event_to_value(ev);
            if let Ok(obj) = serde_wasm_bindgen::to_value(&v) {
                let _ = cb_evt.call1(&this_evt, &obj);
            }
        };

        let retry_cfg = self.inner.retry_config.borrow().clone();
        let retry = RetryProvider::new(&self.inner.provider, retry_cfg);
        let result = chain(
            &retry,
            &self.inner.model,
            prompt,
            Some(&self.inner.api_key),
            true,
            &self.inner.registry,
            limit,
            &mut on_chunk,
            Some(&mut on_event),
            opts.budget,
            ParallelConfig::default(),
        )
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;

        build_chain_result_js(result)
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
        let retry_cfg = self.inner.retry_config.borrow().clone();
        let retry = RetryProvider::new(&self.inner.provider, retry_cfg);
        let response = retry
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
        let retry_cfg = self.inner.retry_config.borrow().clone();
        let retry = RetryProvider::new(&self.inner.provider, retry_cfg);
        let result = chain(
            &retry,
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

        let retry_cfg = self.inner.retry_config.borrow().clone();
        let retry = RetryProvider::new(&self.inner.provider, retry_cfg);
        let result: ChainResult = chain(
            &retry,
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

struct ChainOptions {
    system: Option<String>,
    chain_limit: Option<usize>,
    budget: Option<u64>,
    on_event: Option<js_sys::Function>,
}

impl ChainOptions {
    fn parse(value: &JsValue) -> Result<Self, JsError> {
        if value.is_undefined() || value.is_null() {
            return Ok(Self {
                system: None,
                chain_limit: None,
                budget: None,
                on_event: None,
            });
        }
        let get = |k: &str| -> JsValue {
            js_sys::Reflect::get(value, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED)
        };
        let system = get("system").as_string();
        let chain_limit = get("chainLimit").as_f64().map(|n| n as usize);
        let budget = get("budget").as_f64().map(|n| n as u64);
        let on_event_val = get("onEvent");
        let on_event = if on_event_val.is_function() {
            Some(on_event_val.unchecked_into::<js_sys::Function>())
        } else {
            None
        };
        Ok(Self {
            system,
            chain_limit,
            budget,
            on_event,
        })
    }
}

fn chain_event_to_value(ev: &ChainEvent) -> Value {
    match ev {
        ChainEvent::IterationStart {
            iteration,
            limit,
            messages,
        } => serde_json::json!({
            "type": "iteration_start",
            "iteration": iteration,
            "limit": limit,
            "messages": messages,
        }),
        ChainEvent::IterationEnd {
            iteration,
            usage,
            cumulative_usage,
            tool_calls,
        } => serde_json::json!({
            "type": "iteration_end",
            "iteration": iteration,
            "usage": usage,
            "cumulative_usage": cumulative_usage,
            "tool_calls": tool_calls,
        }),
        ChainEvent::BudgetExhausted {
            cumulative_usage,
            budget,
        } => serde_json::json!({
            "type": "budget_exhausted",
            "cumulative_usage": cumulative_usage,
            "budget": budget,
        }),
    }
}

fn build_chain_result_js(r: ChainResult) -> Result<JsValue, JsError> {
    let text = llm_core::collect_text(&r.chunks);
    let tool_calls = llm_core::collect_tool_calls(&r.chunks);
    let v = serde_json::json!({
        "text": text,
        "toolCalls": tool_calls,
        "totalUsage": r.total_usage,
        "budgetExhausted": r.budget_exhausted,
    });
    serde_wasm_bindgen::to_value(&v).map_err(|e| JsError::new(&e.to_string()))
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
