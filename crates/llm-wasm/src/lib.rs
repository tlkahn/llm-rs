mod agent;
mod conversation;
mod retry;
mod store;
mod tools;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use async_trait::async_trait;
use futures::StreamExt;
use llm_anthropic::provider::AnthropicProvider;
use llm_core::retry::RetryConfig;
use llm_core::stream::Chunk;
use llm_core::types::{Message, ModelInfo, Options, Prompt, Response, ToolCall, ToolResult, Usage};
use llm_core::{
    chain, multi_schema, parse_schema_dsl as core_parse_schema_dsl, ChainEvent, ChainResult,
    ParallelConfig, Provider,
};
use llm_openai::provider::OpenAiProvider;
use llm_store::ConversationStore;
use serde_json::Value;
use wasm_bindgen::prelude::*;

pub use crate::agent::WasmAgentConfig;
pub use crate::conversation::Conversation;
use crate::retry::RetryProvider;
use crate::store::JsConversationStore;
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
    pub log_store: RefCell<Option<Rc<JsConversationStore>>>,
    pub conversation_id: RefCell<Option<String>>,
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
                log_store: RefCell::new(None),
                conversation_id: RefCell::new(None),
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
                log_store: RefCell::new(None),
                conversation_id: RefCell::new(None),
            }),
        }
    }

    /// Attach a JS-backed conversation store. `spec` must be an object with
    /// four function fields: `logResponse(cid, model, response)`,
    /// `readConversation(id)`, `listConversations(limit)`,
    /// `latestConversationId()`. Each is called with JSON-serializable Rust
    /// values and may return a Promise. Once set, every top-level `prompt`
    /// (and `Conversation.send`) call appends a `Response` to the store.
    #[wasm_bindgen(js_name = "setConversationStore")]
    pub fn set_conversation_store(&self, spec: JsValue) -> Result<(), JsError> {
        let js_store = JsConversationStore::from_spec(spec)?;
        *self.inner.log_store.borrow_mut() = Some(Rc::new(js_store));
        *self.inner.conversation_id.borrow_mut() = None;
        Ok(())
    }

    /// Clear any attached conversation store.
    #[wasm_bindgen(js_name = "clearConversationStore")]
    pub fn clear_conversation_store(&self) {
        *self.inner.log_store.borrow_mut() = None;
        *self.inner.conversation_id.borrow_mut() = None;
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
        let prompt_options = prompt.options.clone();

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

        let start = js_sys::Date::now();
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

        let duration_ms = (js_sys::Date::now() - start).max(0.0) as u64;
        let response_text = llm_core::collect_text(&result.chunks);
        let tool_calls = llm_core::collect_tool_calls(&result.chunks);
        self.auto_log(
            text,
            opts.system.as_deref(),
            &prompt_options,
            response_text,
            tool_calls,
            result.tool_results.clone(),
            result.total_usage.clone(),
            None,
            duration_ms,
        )
        .await?;

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
        let prompt_options = prompt.options.clone();

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

        let start = js_sys::Date::now();
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

        let duration_ms = (js_sys::Date::now() - start).max(0.0) as u64;
        let response_text = llm_core::collect_text(&result.chunks);
        let tool_calls = llm_core::collect_tool_calls(&result.chunks);
        self.auto_log(
            text,
            opts.system.as_deref(),
            &prompt_options,
            response_text,
            tool_calls,
            result.tool_results.clone(),
            result.total_usage.clone(),
            None,
            duration_ms,
        )
        .await?;

        build_chain_result_js(result)
    }

    /// Run a configured agent with CLI-parity precedence.
    ///
    /// `config` is a `AgentConfig`; `text` is the user prompt. `options`
    /// is an optional JS object: `{ system?: string, retries?: number }`.
    /// See the Python `run_agent` docstring for the full precedence rules.
    /// Returns a `ChainResult`-shaped JS object.
    #[wasm_bindgen(js_name = "runAgent")]
    pub async fn run_agent(
        &self,
        config: &WasmAgentConfig,
        text: &str,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let (arg_system, arg_retries) = parse_run_agent_options(&options);

        let agent_cfg = &config.inner;
        let model = llm_core::resolve_agent_model(agent_cfg, &self.inner.model).to_string();
        let effective_system = llm_core::resolve_agent_system(arg_system.as_deref(), agent_cfg)
            .map(|s| s.to_string());
        let client_retry = self.inner.retry_config.borrow().clone();
        let effective_retry =
            llm_core::resolve_agent_retry(arg_retries, agent_cfg, &client_retry);
        let budget = llm_core::resolve_agent_budget(agent_cfg);

        let registry_tools = self.inner.registry.list_tools();
        let tools = llm_core::resolve_agent_tools(agent_cfg, &registry_tools)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let mut p = Prompt::new(text).with_tools(tools);
        if let Some(s) = &effective_system {
            p = p.with_system(s);
        }
        for (k, v) in &agent_cfg.options {
            p = p.with_option(k, v.clone());
        }
        let prompt_options = p.options.clone();

        let parallel = ParallelConfig {
            enabled: agent_cfg.parallel_tools,
            max_concurrent: agent_cfg.max_parallel_tools,
        };
        let limit = agent_cfg.chain_limit;

        let mut on_chunk = |_: &Chunk| {};
        let start = js_sys::Date::now();
        let retry = RetryProvider::new(&self.inner.provider, effective_retry);
        let result = chain(
            &retry,
            &model,
            p,
            Some(&self.inner.api_key),
            false,
            &self.inner.registry,
            limit,
            &mut on_chunk,
            None,
            budget,
            parallel,
        )
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;

        let duration_ms = (js_sys::Date::now() - start).max(0.0) as u64;
        let response_text = llm_core::collect_text(&result.chunks);
        let tool_calls = llm_core::collect_tool_calls(&result.chunks);
        self.auto_log(
            text,
            effective_system.as_deref(),
            &prompt_options,
            response_text,
            tool_calls,
            result.tool_results.clone(),
            result.total_usage.clone(),
            None,
            duration_ms,
        )
        .await?;

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
        // The prompt text/system fields haven't been set for plain `prompt()`
        // calls; pull them off the Prompt for auto-logging.
        let user_text = prompt.text.clone();
        let system = prompt.system.clone();
        let options = prompt.options.clone();
        let schema = prompt.schema.clone();
        if self.inner.registry.has_any() {
            self.run_chain_logging(
                prompt,
                stream,
                callback.as_ref(),
                &user_text,
                system.as_deref(),
                &options,
                schema,
            )
            .await
        } else {
            self.run_direct_logging(
                prompt,
                stream,
                callback.as_ref(),
                &user_text,
                system.as_deref(),
                &options,
                schema,
            )
            .await
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_direct_logging(
        &self,
        prompt: Prompt,
        stream: bool,
        callback: Option<&js_sys::Function>,
        user_text: &str,
        system: Option<&str>,
        options: &Options,
        schema: Option<Value>,
    ) -> Result<String, JsError> {
        let start = js_sys::Date::now();
        let retry_cfg = self.inner.retry_config.borrow().clone();
        let retry = RetryProvider::new(&self.inner.provider, retry_cfg);
        let response = retry
            .execute(&self.inner.model, &prompt, Some(&self.inner.api_key), stream)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        let mut pinned = std::pin::pin!(response);
        let mut text = String::new();
        let mut chunks: Vec<Chunk> = Vec::new();
        let this = JsValue::NULL;
        while let Some(item) = pinned.next().await {
            match item {
                Ok(Chunk::Text(t)) => {
                    if let Some(cb) = callback {
                        let _ = cb.call1(&this, &JsValue::from_str(&t));
                    }
                    text.push_str(&t);
                    chunks.push(Chunk::Text(t));
                }
                Ok(Chunk::Done) => break,
                Ok(other) => chunks.push(other),
                Err(e) => return Err(JsError::new(&e.to_string())),
            }
        }
        let duration_ms = (js_sys::Date::now() - start).max(0.0) as u64;
        let usage = llm_core::collect_usage(&chunks);
        self.auto_log(
            user_text,
            system,
            options,
            text.clone(),
            Vec::new(),
            Vec::new(),
            usage,
            schema,
            duration_ms,
        )
        .await?;
        Ok(text)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_chain_logging(
        &self,
        prompt: Prompt,
        stream: bool,
        callback: Option<&js_sys::Function>,
        user_text: &str,
        system: Option<&str>,
        options: &Options,
        schema: Option<Value>,
    ) -> Result<String, JsError> {
        let cb = callback.cloned();
        let this = JsValue::NULL;
        let mut on_chunk = move |chunk: &Chunk| {
            if let (Chunk::Text(t), Some(c)) = (chunk, &cb) {
                let _ = c.call1(&this, &JsValue::from_str(t));
            }
        };
        let start = js_sys::Date::now();
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
        let duration_ms = (js_sys::Date::now() - start).max(0.0) as u64;
        let text = llm_core::collect_text(&result.chunks);
        let tool_calls = llm_core::collect_tool_calls(&result.chunks);
        self.auto_log(
            user_text,
            system,
            options,
            text.clone(),
            tool_calls,
            result.tool_results,
            result.total_usage,
            schema,
            duration_ms,
        )
        .await?;
        Ok(text)
    }

    /// Used by `Conversation`: run the chain seeded with `messages` and
    /// return `(final_text, updated_messages, turn_data_for_logging)`.
    pub(crate) async fn send_messages(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        callback: Option<js_sys::Function>,
    ) -> Result<WasmTurnOutput, JsError> {
        let mut p = Prompt::new("")
            .with_tools(self.inner.registry.list_tools())
            .with_messages(messages);
        if let Some(s) = &system {
            p = p.with_system(s);
        }

        let cb = callback;
        let this = JsValue::NULL;
        let mut on_chunk = move |chunk: &Chunk| {
            if let (Chunk::Text(t), Some(c)) = (chunk, &cb) {
                let _ = c.call1(&this, &JsValue::from_str(t));
            }
        };

        let start = js_sys::Date::now();
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

        let duration_ms = (js_sys::Date::now() - start).max(0.0) as u64;
        let text = llm_core::collect_text(&result.chunks);
        let tool_calls = llm_core::collect_tool_calls(&result.chunks);
        Ok(WasmTurnOutput {
            text,
            tool_calls,
            tool_results: result.tool_results,
            total_usage: result.total_usage,
            duration_ms,
            messages: result.messages,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn auto_log(
        &self,
        user_text: &str,
        system: Option<&str>,
        options: &Options,
        text: String,
        tool_calls: Vec<ToolCall>,
        tool_results: Vec<ToolResult>,
        total_usage: Option<Usage>,
        schema: Option<Value>,
        duration_ms: u64,
    ) -> Result<(), JsError> {
        let Some(store) = self.inner.log_store.borrow().clone() else {
            return Ok(());
        };
        let response: Response = crate::store::build_response_wasm(
            &self.inner.model,
            user_text,
            system,
            options.clone(),
            text,
            total_usage,
            tool_calls,
            tool_results,
            schema,
            None,
            duration_ms,
        );
        let prev_cid = self.inner.conversation_id.borrow().clone();
        let cid = store
            .log_response(prev_cid.as_deref(), &self.inner.model, &response)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        *self.inner.conversation_id.borrow_mut() = Some(cid);
        Ok(())
    }

    /// Log an externally-constructed Response. Used by `Conversation.send`.
    pub(crate) async fn log_external(&self, response: &Response) -> Result<(), JsError> {
        let Some(store) = self.inner.log_store.borrow().clone() else {
            return Ok(());
        };
        let prev_cid = self.inner.conversation_id.borrow().clone();
        let cid = store
            .log_response(prev_cid.as_deref(), &self.inner.model, response)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        *self.inner.conversation_id.borrow_mut() = Some(cid);
        Ok(())
    }

    pub(crate) fn store(&self) -> Option<Rc<JsConversationStore>> {
        self.inner.log_store.borrow().clone()
    }

    pub(crate) fn model(&self) -> &str {
        &self.inner.model
    }

    pub(crate) fn set_conversation_id(&self, id: Option<String>) {
        *self.inner.conversation_id.borrow_mut() = id;
    }
}

/// Per-turn output used by `Conversation` to update history and optionally
/// auto-log the response.
pub(crate) struct WasmTurnOutput {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub total_usage: Option<Usage>,
    pub duration_ms: u64,
    pub messages: Vec<Message>,
}

/// Parse `{ system?, retries? }` from a JS object.
fn parse_run_agent_options(value: &JsValue) -> (Option<String>, Option<u32>) {
    if value.is_undefined() || value.is_null() {
        return (None, None);
    }
    let get = |k: &str| -> JsValue {
        js_sys::Reflect::get(value, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED)
    };
    let system = get("system").as_string();
    let retries = get("retries").as_f64().map(|n| n as u32);
    (system, retries)
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
