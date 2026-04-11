mod conversation;
mod retry;
mod tools;

use std::sync::{mpsc, Mutex};

use futures::StreamExt;
use llm_anthropic::provider::AnthropicProvider;
use llm_core::retry::RetryConfig;
use llm_core::stream::Chunk;
use llm_core::types::{Message, Prompt};
use llm_core::{
    chain, multi_schema, parse_schema_dsl as core_parse_schema_dsl, ChainEvent, ChainResult,
    ParallelConfig, Provider,
};
use llm_openai::provider::OpenAiProvider;
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde_json::Value;

use crate::conversation::Conversation;
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
    #[allow(dead_code)]
    log_store: Option<llm_store::LogStore>,
}

#[pymethods]
impl LlmClient {
    #[new]
    #[pyo3(signature = (api_key, model="gpt-4o-mini", *, provider=None, base_url=None, log_dir=None, chain_limit=5, retries=0, retry_base_delay_ms=1000, retry_max_delay_ms=30000, retry_jitter=true))]
    fn new(
        api_key: &str,
        model: &str,
        provider: Option<&str>,
        base_url: Option<&str>,
        log_dir: Option<&str>,
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

        let log_store = log_dir
            .map(|d| llm_store::LogStore::open(std::path::Path::new(d)))
            .transpose()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

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
            log_store,
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
    /// schema in an items array.
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
        let prompt = self.build_prompt(text, system, schema_value);
        if self.registry.has_any() {
            self.run_chain(prompt)
        } else {
            self.run_direct(prompt)
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

    fn run_direct(&self, prompt: Prompt) -> PyResult<String> {
        let result: llm_core::Result<String> = self.runtime.block_on(async {
            let retry = RetryProvider::new(&self.provider, self.retry_config.clone());
            let stream = retry
                .execute(&self.model, &prompt, Some(&self.api_key), false)
                .await?;
            let mut pinned = std::pin::pin!(stream);
            let mut text = String::new();
            while let Some(item) = pinned.next().await {
                match item? {
                    Chunk::Text(t) => text.push_str(&t),
                    Chunk::Done => break,
                    _ => {}
                }
            }
            Ok(text)
        });
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn run_chain(&self, prompt: Prompt) -> PyResult<String> {
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
        Ok(llm_core::collect_text(&chain_result.chunks))
    }

    /// Used by `Conversation`: run a multi-turn chain seeded with `messages`
    /// and return (final_text, updated_messages).
    pub(crate) fn send_messages(
        &self,
        messages: &[Message],
        system: Option<&str>,
    ) -> PyResult<(String, Vec<Message>)> {
        let mut p = Prompt::new("")
            .with_tools(self.registry.list_tools())
            .with_messages(messages.to_vec());
        if let Some(s) = system {
            p = p.with_system(s);
        }

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
        let text = llm_core::collect_text(&chain_result.chunks);
        Ok((text, chain_result.messages))
    }

    pub(crate) fn send_messages_streaming(
        &self,
        messages: &[Message],
        system: Option<&str>,
        py: Python<'_>,
        callback: Py<PyAny>,
    ) -> PyResult<(String, Vec<Message>)> {
        let mut p = Prompt::new("")
            .with_tools(self.registry.list_tools())
            .with_messages(messages.to_vec());
        if let Some(s) = system {
            p = p.with_system(s);
        }

        let cb = callback;
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
        let text = llm_core::collect_text(&chain_result.chunks);
        Ok((text, chain_result.messages))
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
    m.add_function(wrap_pyfunction!(parse_schema_dsl, m)?)?;
    Ok(())
}
