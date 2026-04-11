mod conversation;
mod tools;

use std::sync::{mpsc, Mutex};

use futures::StreamExt;
use llm_anthropic::provider::AnthropicProvider;
use llm_core::stream::Chunk;
use llm_core::types::{Message, Prompt};
use llm_core::{
    chain, multi_schema, parse_schema_dsl as core_parse_schema_dsl, ChainResult, ParallelConfig,
    Provider,
};
use llm_openai::provider::OpenAiProvider;
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde_json::Value;

use crate::conversation::Conversation;
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
    #[allow(dead_code)]
    log_store: Option<llm_store::LogStore>,
}

#[pymethods]
impl LlmClient {
    #[new]
    #[pyo3(signature = (api_key, model="gpt-4o-mini", *, provider=None, base_url=None, log_dir=None, chain_limit=5))]
    fn new(
        api_key: &str,
        model: &str,
        provider: Option<&str>,
        base_url: Option<&str>,
        log_dir: Option<&str>,
        chain_limit: usize,
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
                chain(
                    &self.provider,
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
                self.provider
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
            let stream = self
                .provider
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
            chain(
                &self.provider,
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
            chain(
                &self.provider,
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

        // We hold the GIL throughout (this is a #[pymethods]-callable path).
        // The closure invokes the Python callback via the same `py` token.
        let cb = callback;
        let result: llm_core::Result<ChainResult> = self.runtime.block_on(async {
            let mut on_chunk = |chunk: &Chunk| {
                if let Chunk::Text(t) = chunk {
                    let _ = cb.call1(py, (t.clone(),));
                }
            };
            chain(
                &self.provider,
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
    m.add_function(wrap_pyfunction!(parse_schema_dsl, m)?)?;
    Ok(())
}
