mod agent;
mod conversation;
mod log_store_py;
mod response_build;
mod retry;
mod tools;

use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use futures::StreamExt;
use llm_anthropic::provider::AnthropicProvider;
use llm_core::retry::RetryConfig;
use llm_core::stream::Chunk;
use llm_core::types::{Message, Options, Prompt, Response, ToolCall, ToolResult, Usage};
use llm_core::{
    chain, multi_schema, parse_schema_dsl as core_parse_schema_dsl, ChainEvent, ChainResult,
    ParallelConfig, Provider,
};
use llm_openai::provider::OpenAiProvider;
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde_json::Value;

use crate::agent::PyAgentConfig;
use crate::conversation::Conversation;
use crate::log_store_py::PyLogStore;
use crate::response_build::{synthesize_response, ResponseInputs};
use crate::retry::RetryProvider;
use crate::tools::{PyToolRegistry, ToolDecorator};

const PY_PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

enum ProviderImpl {
    OpenAi(OpenAiProvider),
    Anthropic(AnthropicProvider),
}

impl ProviderImpl {
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

#[async_trait::async_trait]
impl Provider for ProviderImpl {
    fn id(&self) -> &str {
        match self {
            ProviderImpl::OpenAi(p) => p.id(),
            ProviderImpl::Anthropic(p) => p.id(),
        }
    }

    fn models(&self) -> Vec<llm_core::types::ModelInfo> {
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
        ProviderImpl::execute(self, model, prompt, key, stream).await
    }
}

#[pyclass]
pub struct LlmClient {
    runtime: tokio::runtime::Runtime,
    provider: ProviderImpl,
    model: String,
    api_key: String,
    registry: PyToolRegistry,
    chain_limit: usize,
    retry_config: RetryConfig,
    pub(crate) log_store: Option<Arc<llm_store::LogStore>>,
    /// Conversation id used for auto-logging successive top-level `prompt()`
    /// calls. Created lazily on the first log and reused until the client
    /// is dropped.
    conversation_id: Mutex<Option<String>>,
}

#[pymethods]
impl LlmClient {
    #[new]
    #[pyo3(signature = (api_key, model="gpt-4o-mini", *, provider=None, base_url=None, log_dir=None, log_store=None, chain_limit=5, retries=0, retry_base_delay_ms=1000, retry_max_delay_ms=30000, retry_jitter=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        api_key: &str,
        model: &str,
        provider: Option<&str>,
        base_url: Option<&str>,
        log_dir: Option<&str>,
        log_store: Option<&PyLogStore>,
        chain_limit: usize,
        retries: u32,
        retry_base_delay_ms: u64,
        retry_max_delay_ms: u64,
        retry_jitter: bool,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let is_anthropic = match provider {
            Some("anthropic") => true,
            Some("openai") => false,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown provider: {other}. Use 'openai' or 'anthropic'."
                )));
            }
            None => model.starts_with("claude"),
        };

        let provider_impl = if is_anthropic {
            let base = base_url.unwrap_or("https://api.anthropic.com");
            ProviderImpl::Anthropic(AnthropicProvider::new(base))
        } else {
            let base = base_url.unwrap_or("https://api.openai.com");
            ProviderImpl::OpenAi(OpenAiProvider::new(base))
        };

        let log_store_arc: Option<Arc<llm_store::LogStore>> = if let Some(ls) = log_store {
            Some(ls.inner.clone())
        } else if let Some(dir) = log_dir {
            let store = llm_store::LogStore::open(std::path::Path::new(dir))
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            Some(Arc::new(store))
        } else {
            None
        };

        Ok(Self {
            runtime,
            provider: provider_impl,
            model: model.to_string(),
            api_key: api_key.to_string(),
            registry: PyToolRegistry::new(),
            chain_limit,
            retry_config: RetryConfig {
                max_retries: retries,
                base_delay_ms: retry_base_delay_ms,
                max_delay_ms: retry_max_delay_ms,
                jitter: retry_jitter,
            },
            log_store: log_store_arc,
            conversation_id: Mutex::new(None),
        })
    }

    /// Decorator factory for registering Python functions as tools.
    ///
    /// Usage:
    /// ```python
    /// @client.tool(description="Add two numbers")
    /// def add(a: int, b: int) -> int:
    ///     return a + b
    /// ```
    #[pyo3(signature = (*, name=None, description=None, schema=None))]
    fn tool(
        &self,
        name: Option<String>,
        description: Option<String>,
        schema: Option<Py<PyAny>>,
    ) -> ToolDecorator {
        ToolDecorator {
            registry: self.registry.clone(),
            description,
            schema,
            name,
        }
    }

    /// Enable the built-in tools (`llm_version`, `llm_time`).
    fn enable_builtin_tools(&self) {
        self.registry.enable_builtins(PY_PKG_VERSION);
    }

    /// Send a prompt and return the response text.
    ///
    /// If `schema` is a string it is parsed as schema DSL; if it is a dict
    /// it is used verbatim as JSON Schema. `schema_multi=True` wraps the
    /// schema in an items array. If the client was constructed with a
    /// `log_store`, the completed response is appended to the client's
    /// rolling conversation automatically.
    #[pyo3(signature = (text, *, system=None, schema=None, schema_multi=false))]
    fn prompt(
        &self,
        py: Python<'_>,
        text: &str,
        system: Option<&str>,
        schema: Option<Py<PyAny>>,
        schema_multi: bool,
    ) -> PyResult<String> {
        let schema_value = build_schema(py, schema.as_ref(), schema_multi)?;
        let _ = py;
        let prompt_obj = self.build_prompt(text, system, schema_value.clone());
        if self.registry.has_any() {
            self.run_chain_logging(prompt_obj, text, system, schema_value)
        } else {
            self.run_direct_logging(prompt_obj, text, system, schema_value)
        }
    }

    /// Run a chain loop with optional observability and budget enforcement.
    ///
    /// Returns a `ChainResult` exposing `text`, `tool_calls`, `total_usage`,
    /// and `budget_exhausted`. If `on_event` is provided it is called with a
    /// dict tagged by `type` (`iteration_start`, `iteration_end`,
    /// `budget_exhausted`).
    #[pyo3(signature = (text, *, system=None, chain_limit=None, budget=None, on_event=None))]
    fn chain(
        &self,
        py: Python<'_>,
        text: &str,
        system: Option<&str>,
        chain_limit: Option<usize>,
        budget: Option<u64>,
        on_event: Option<Py<PyAny>>,
    ) -> PyResult<PyChainResult> {
        let prompt = self.build_prompt(text, system, None);
        let limit = chain_limit.unwrap_or(self.chain_limit);
        let cb_opt = on_event;

        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |_: &Chunk| {};
            let mut on_event_cb = |ev: &ChainEvent| {
                if let Some(cb) = &cb_opt {
                    let v = chain_event_to_value(ev);
                    if let Ok(obj) = pythonize(py, &v) {
                        let _ = cb.call1(py, (obj,));
                    }
                }
            };
            let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
            chain(
                &retry,
                &self.model,
                prompt,
                Some(&self.api_key),
                false,
                &self.registry,
                limit,
                &mut on_chunk,
                Some(&mut on_event_cb),
                budget,
                ParallelConfig::default(),
            )
            .await
        });
        let r = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        build_chain_result(py, r)
    }

    /// Run a configured agent on `prompt`. Honors CLI-parity precedence:
    ///
    /// - **Model**: `config.model` if set, else the client's constructor
    ///   model. The model must belong to the client's provider — cross-
    ///   provider switching is out of scope.
    /// - **System**: arg > `config.system_prompt` > none.
    /// - **Retry**: arg `retries` > `config.retry` > client default.
    /// - **Chain limit**: `config.chain_limit` (default 10).
    /// - **Budget**: `config.budget.max_tokens` if set.
    /// - **Tools**: `config.tools` is a whitelist against the client's
    ///   registry. Unknown tool names error with
    ///   `unknown tool in agent config: {name}`.
    ///
    /// Returns a `ChainResult`.
    #[pyo3(signature = (config, prompt, *, system=None, retries=None))]
    fn run_agent(
        &self,
        py: Python<'_>,
        config: &PyAgentConfig,
        prompt: &str,
        system: Option<&str>,
        retries: Option<u32>,
    ) -> PyResult<PyChainResult> {
        let agent_cfg = &config.inner;
        let model = llm_core::resolve_agent_model(agent_cfg, &self.model).to_string();
        let effective_system = llm_core::resolve_agent_system(system, agent_cfg);
        let effective_retry =
            llm_core::resolve_agent_retry(retries, agent_cfg, &self.retry_config);
        let budget = llm_core::resolve_agent_budget(agent_cfg);

        let registry_tools = self.registry.list_tools();
        let tools = llm_core::resolve_agent_tools(agent_cfg, &registry_tools)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let mut p = Prompt::new(prompt).with_tools(tools);
        if let Some(s) = effective_system {
            p = p.with_system(s);
        }
        for (k, v) in &agent_cfg.options {
            p = p.with_option(k, v.clone());
        }

        let parallel = ParallelConfig {
            enabled: agent_cfg.parallel_tools,
            max_concurrent: agent_cfg.max_parallel_tools,
        };
        let limit = agent_cfg.chain_limit;

        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |_: &Chunk| {};
            let retry = RetryProvider::new(&self.provider, effective_retry);
            chain(
                &retry,
                &model,
                p,
                Some(&self.api_key),
                false,
                &self.registry,
                limit,
                &mut on_chunk,
                None,
                budget,
                parallel,
            )
            .await
        });
        let r =
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        build_chain_result(py, r)
    }

    /// Streaming variant: invokes `callback` with type-tagged dicts for both
    /// text chunks (`{"type": "text", "content": str}`) and events, as they
    /// are emitted. Returns the final `ChainResult`.
    #[pyo3(signature = (text, callback, *, system=None, chain_limit=None, budget=None))]
    fn chain_stream(
        &self,
        py: Python<'_>,
        text: &str,
        callback: Py<PyAny>,
        system: Option<&str>,
        chain_limit: Option<usize>,
        budget: Option<u64>,
    ) -> PyResult<PyChainResult> {
        let prompt = self.build_prompt(text, system, None);
        let limit = chain_limit.unwrap_or(self.chain_limit);
        let cb = callback;

        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |chunk: &Chunk| {
                if let Chunk::Text(t) = chunk {
                    let v = serde_json::json!({"type": "text", "content": t});
                    if let Ok(obj) = pythonize(py, &v) {
                        let _ = cb.call1(py, (obj,));
                    }
                }
            };
            let mut on_event_cb = |ev: &ChainEvent| {
                let v = chain_event_to_value(ev);
                if let Ok(obj) = pythonize(py, &v) {
                    let _ = cb.call1(py, (obj,));
                }
            };
            let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
            chain(
                &retry,
                &self.model,
                prompt,
                Some(&self.api_key),
                true,
                &self.registry,
                limit,
                &mut on_chunk,
                Some(&mut on_event_cb),
                budget,
                ParallelConfig::default(),
            )
            .await
        });
        let r = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        build_chain_result(py, r)
    }

    /// Send a prompt and stream text chunks. Returns an iterator yielding strings.
    #[pyo3(signature = (text, *, system=None, schema=None, schema_multi=false))]
    fn prompt_stream(
        &self,
        py: Python<'_>,
        text: &str,
        system: Option<&str>,
        schema: Option<Py<PyAny>>,
        schema_multi: bool,
    ) -> PyResult<ChunkIterator> {
        let schema_value = build_schema(py, schema.as_ref(), schema_multi)?;
        let prompt = self.build_prompt(text, system, schema_value);

        let (tx, rx) = mpsc::channel::<Option<String>>();

        if self.registry.has_any() {
            // Tool path: run chain on the runtime, forward text via channel.
            let tx_clone = tx.clone();
            let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
                let mut on_chunk = move |chunk: &Chunk| {
                    if let Chunk::Text(t) = chunk {
                        let _ = tx_clone.send(Some(t.clone()));
                    }
                };
                let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
                chain(
                    &retry,
                    &self.model,
                    prompt,
                    Some(&self.api_key),
                    true,
                    &self.registry,
                    self.chain_limit,
                    &mut on_chunk,
                    None,
                    None,
                    ParallelConfig::default(),
                )
                .await
            });
            let _ = tx.send(None);
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        } else {
            // No-tool path: stream directly via spawned task.
            let stream_res = self.runtime.block_on(async {
                let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
                retry
                    .execute(&self.model, &prompt, Some(&self.api_key), true)
                    .await
            });
            let response_stream = stream_res
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let tx_clone = tx.clone();
            self.runtime.spawn(async move {
                let mut stream = std::pin::pin!(response_stream);
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(Chunk::Text(t)) => {
                            if tx_clone.send(Some(t)).is_err() {
                                break;
                            }
                        }
                        Ok(Chunk::Done) => break,
                        Err(_) => break,
                        _ => {}
                    }
                }
                let _ = tx_clone.send(None);
            });
        }

        Ok(ChunkIterator {
            receiver: Mutex::new(rx),
        })
    }
}

impl LlmClient {
    fn build_prompt(
        &self,
        text: &str,
        system: Option<&str>,
        schema: Option<Value>,
    ) -> Prompt {
        let mut p = Prompt::new(text).with_tools(self.registry.list_tools());
        if let Some(sys) = system {
            p = p.with_system(sys);
        }
        if let Some(s) = schema {
            p = p.with_schema(s);
        }
        p
    }

    #[allow(clippy::too_many_arguments)]
    fn auto_log(
        &self,
        user_text: &str,
        system: Option<&str>,
        options: &Options,
        chunks: &[Chunk],
        tool_calls: Vec<ToolCall>,
        tool_results: Vec<ToolResult>,
        total_usage: Option<Usage>,
        schema: Option<Value>,
        duration_ms: u64,
    ) -> PyResult<()> {
        let Some(store) = self.log_store.as_ref() else {
            return Ok(());
        };
        let response = synthesize_response(ResponseInputs {
            model: &self.model,
            prompt: user_text,
            system,
            options: options.clone(),
            chunks,
            tool_calls,
            tool_results,
            total_usage,
            schema,
            schema_id: None,
            duration_ms,
        });
        let prev_cid = self.conversation_id.lock().unwrap().clone();
        let new_cid = store
            .log_response(prev_cid.as_deref(), &self.model, &response)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        *self.conversation_id.lock().unwrap() = Some(new_cid);
        Ok(())
    }

    /// Log a completed `Response` against `store` using `cid` as the active
    /// conversation (or creating a new one). Returns the resulting cid.
    /// Used by `Conversation` when it holds its own store.
    pub(crate) fn log_response_external(
        store: &Arc<llm_store::LogStore>,
        cid: Option<&str>,
        model: &str,
        response: &Response,
    ) -> PyResult<String> {
        store
            .log_response(cid, model, response)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    pub(crate) fn model_name(&self) -> &str {
        &self.model
    }

    fn run_direct_logging(
        &self,
        prompt: Prompt,
        user_text: &str,
        system: Option<&str>,
        schema: Option<Value>,
    ) -> PyResult<String> {
        let start = Instant::now();
        let result: llm_core::Result<(Vec<Chunk>, Option<Usage>)> =
            self.runtime.block_on(async {
                let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
                let stream = retry
                    .execute(&self.model, &prompt, Some(&self.api_key), false)
                    .await?;
                let mut pinned = std::pin::pin!(stream);
                let mut chunks = Vec::new();
                while let Some(item) = pinned.next().await {
                    let chunk = item?;
                    if matches!(chunk, Chunk::Done) {
                        break;
                    }
                    chunks.push(chunk);
                }
                let usage = llm_core::collect_usage(&chunks);
                Ok((chunks, usage))
            });
        let (chunks, usage) =
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let duration_ms = start.elapsed().as_millis() as u64;
        let text = llm_core::collect_text(&chunks);
        self.auto_log(
            user_text,
            system,
            &prompt.options,
            &chunks,
            Vec::new(),
            Vec::new(),
            usage,
            schema,
            duration_ms,
        )?;
        Ok(text)
    }

    fn run_chain_logging(
        &self,
        prompt: Prompt,
        user_text: &str,
        system: Option<&str>,
        schema: Option<Value>,
    ) -> PyResult<String> {
        let start = Instant::now();
        let options = prompt.options.clone();
        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |_: &Chunk| {};
            let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
            chain(
                &retry,
                &self.model,
                prompt,
                Some(&self.api_key),
                false,
                &self.registry,
                self.chain_limit,
                &mut on_chunk,
                None,
                None,
                ParallelConfig::default(),
            )
            .await
        });
        let chain_result =
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let duration_ms = start.elapsed().as_millis() as u64;
        let text = llm_core::collect_text(&chain_result.chunks);
        let tool_calls = llm_core::collect_tool_calls(&chain_result.chunks);
        self.auto_log(
            user_text,
            system,
            &options,
            &chain_result.chunks,
            tool_calls,
            chain_result.tool_results,
            chain_result.total_usage,
            schema,
            duration_ms,
        )?;
        Ok(text)
    }

    /// Used by `Conversation`: run a multi-turn chain seeded with `messages`
    /// and return a `TurnOutput` with everything the caller needs to update
    /// its history and optionally auto-log the turn.
    pub(crate) fn send_messages(
        &self,
        messages: &[Message],
        system: Option<&str>,
    ) -> PyResult<TurnOutput> {
        let mut p = Prompt::new("")
            .with_tools(self.registry.list_tools())
            .with_messages(messages.to_vec());
        if let Some(s) = system {
            p = p.with_system(s);
        }

        let start = Instant::now();
        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |_: &Chunk| {};
            let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
            chain(
                &retry,
                &self.model,
                p,
                Some(&self.api_key),
                false,
                &self.registry,
                self.chain_limit,
                &mut on_chunk,
                None,
                None,
                ParallelConfig::default(),
            )
            .await
        });
        let chain_result =
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let duration_ms = start.elapsed().as_millis() as u64;
        Ok(TurnOutput::from_chain(chain_result, duration_ms))
    }

    pub(crate) fn send_messages_streaming(
        &self,
        messages: &[Message],
        system: Option<&str>,
        py: Python<'_>,
        callback: Py<PyAny>,
    ) -> PyResult<TurnOutput> {
        let mut p = Prompt::new("")
            .with_tools(self.registry.list_tools())
            .with_messages(messages.to_vec());
        if let Some(s) = system {
            p = p.with_system(s);
        }

        let cb = callback;
        let start = Instant::now();
        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |chunk: &Chunk| {
                if let Chunk::Text(t) = chunk {
                    let _ = cb.call1(py, (t.clone(),));
                }
            };
            let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
            chain(
                &retry,
                &self.model,
                p,
                Some(&self.api_key),
                true,
                &self.registry,
                self.chain_limit,
                &mut on_chunk,
                None,
                None,
                ParallelConfig::default(),
            )
            .await
        });
        let chain_result =
            result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let duration_ms = start.elapsed().as_millis() as u64;
        Ok(TurnOutput::from_chain(chain_result, duration_ms))
    }
}

/// One turn worth of data returned by `send_messages{_streaming}` so that
/// `Conversation` can both update its history and, if it has an attached
/// store, synthesize a `Response` and append it to the log.
pub(crate) struct TurnOutput {
    pub text: String,
    pub chunks: Vec<Chunk>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub total_usage: Option<Usage>,
    pub duration_ms: u64,
    pub messages: Vec<Message>,
}

impl TurnOutput {
    fn from_chain(r: ChainResult, duration_ms: u64) -> Self {
        let text = llm_core::collect_text(&r.chunks);
        let tool_calls = llm_core::collect_tool_calls(&r.chunks);
        Self {
            text,
            chunks: r.chunks,
            tool_calls,
            tool_results: r.tool_results,
            total_usage: r.total_usage,
            duration_ms,
            messages: r.messages,
        }
    }
}

fn build_schema(
    py: Python<'_>,
    schema: Option<&Py<PyAny>>,
    multi: bool,
) -> PyResult<Option<Value>> {
    let Some(s) = schema else {
        return Ok(None);
    };
    let bound = s.bind(py);
    let value: Value = if let Ok(text) = bound.extract::<String>() {
        core_parse_schema_dsl(&text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        depythonize(bound)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    Ok(Some(if multi { multi_schema(value) } else { value }))
}

/// Result of `LlmClient.chain()` / `chain_stream()`.
#[pyclass(name = "ChainResult")]
pub struct PyChainResult {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    tool_calls: Py<PyAny>,
    #[pyo3(get)]
    total_usage: Py<PyAny>,
    #[pyo3(get)]
    budget_exhausted: bool,
}

fn build_chain_result(py: Python<'_>, r: ChainResult) -> PyResult<PyChainResult> {
    let text = llm_core::collect_text(&r.chunks);
    let tool_calls = llm_core::collect_tool_calls(&r.chunks);
    let tc_obj = pythonize(py, &tool_calls)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .unbind();
    let usage_obj = match &r.total_usage {
        Some(u) => pythonize(py, u)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .unbind(),
        None => py.None(),
    };
    Ok(PyChainResult {
        text,
        tool_calls: tc_obj,
        total_usage: usage_obj,
        budget_exhausted: r.budget_exhausted,
    })
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

#[pyclass]
pub struct ChunkIterator {
    receiver: Mutex<mpsc::Receiver<Option<String>>>,
}

#[pymethods]
impl ChunkIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> Option<String> {
        let rx = self.receiver.lock().unwrap();
        match rx.recv() {
            Ok(Some(text)) => Some(text),
            Ok(None) => None,
            Err(_) => None,
        }
    }
}

/// Parse a schema DSL string into a JSON Schema dict.
#[pyfunction]
fn parse_schema_dsl(py: Python<'_>, dsl: &str) -> PyResult<PyObject> {
    let value = core_parse_schema_dsl(dsl)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let bound =
        pythonize(py, &value).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(bound.unbind())
}

#[pymodule]
fn llm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LlmClient>()?;
    m.add_class::<ChunkIterator>()?;
    m.add_class::<Conversation>()?;
    m.add_class::<ToolDecorator>()?;
    m.add_class::<PyChainResult>()?;
    m.add_class::<PyLogStore>()?;
    m.add_class::<PyAgentConfig>()?;
    m.add_function(wrap_pyfunction!(parse_schema_dsl, m)?)?;
    Ok(())
}
