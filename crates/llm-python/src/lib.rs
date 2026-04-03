use futures::StreamExt;
use llm_core::stream::Chunk;
use llm_core::types::Prompt;
use llm_core::Provider;
use llm_openai::provider::OpenAiProvider;
use pyo3::prelude::*;
use std::sync::{mpsc, Mutex};

#[pyclass]
struct LlmClient {
    runtime: tokio::runtime::Runtime,
    provider: OpenAiProvider,
    model: String,
    api_key: String,
    #[allow(dead_code)]
    log_store: Option<llm_store::LogStore>,
}

#[pymethods]
impl LlmClient {
    #[new]
    #[pyo3(signature = (api_key, model="gpt-4o-mini", *, base_url=None, log_dir=None))]
    fn new(
        api_key: &str,
        model: &str,
        base_url: Option<&str>,
        log_dir: Option<&str>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let base = base_url.unwrap_or("https://api.openai.com");
        let provider = OpenAiProvider::new(base);
        let log_store = log_dir
            .map(|d| llm_store::LogStore::open(std::path::Path::new(d)))
            .transpose()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            runtime,
            provider,
            model: model.to_string(),
            api_key: api_key.to_string(),
            log_store,
        })
    }

    /// Send a prompt and return the response text.
    #[pyo3(signature = (text, *, system=None))]
    fn prompt(&self, text: &str, system: Option<&str>) -> PyResult<String> {
        let mut p = Prompt::new(text);
        if let Some(sys) = system {
            p = p.with_system(sys);
        }

        let result = self.runtime.block_on(async {
            let stream = self
                .provider
                .execute(&self.model, &p, Some(&self.api_key), false)
                .await?;

            let mut stream = std::pin::pin!(stream);
            let mut text = String::new();
            while let Some(result) = stream.next().await {
                match result {
                    Ok(Chunk::Text(t)) => text.push_str(&t),
                    Ok(Chunk::Done) => break,
                    Err(e) => return Err(e),
                    _ => {}
                }
            }
            Ok(text)
        });

        result.map_err(|e: llm_core::LlmError| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Send a prompt and return an iterator that yields text chunks.
    #[pyo3(signature = (text, *, system=None))]
    fn prompt_stream(&self, text: &str, system: Option<&str>) -> PyResult<ChunkIterator> {
        let mut p = Prompt::new(text);
        if let Some(sys) = system {
            p = p.with_system(sys);
        }

        let (tx, rx) = mpsc::channel::<Option<String>>();
        let model = self.model.clone();
        let api_key = self.api_key.clone();

        // Get the stream on the current runtime
        let response_stream = self.runtime.block_on(async {
            self.provider
                .execute(&model, &p, Some(&api_key), true)
                .await
        });

        let response_stream = response_stream.map_err(|e: llm_core::LlmError| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;

        // Spawn a background task to consume the stream
        self.runtime.spawn(async move {
            let mut stream = std::pin::pin!(response_stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(Chunk::Text(t)) => {
                        if tx.send(Some(t)).is_err() {
                            break;
                        }
                    }
                    Ok(Chunk::Done) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
            let _ = tx.send(None);
        });

        Ok(ChunkIterator {
            receiver: Mutex::new(rx),
        })
    }
}

#[pyclass]
struct ChunkIterator {
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

#[pymodule]
fn llm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LlmClient>()?;
    m.add_class::<ChunkIterator>()?;
    Ok(())
}
