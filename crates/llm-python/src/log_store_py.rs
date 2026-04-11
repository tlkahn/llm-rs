//! Python wrapper around `llm_store::LogStore`.

use std::path::Path;
use std::sync::Arc;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pythonize::pythonize;

/// Python-facing persistent log store.
///
/// Wraps `llm_store::LogStore` in an `Arc` so it can be shared between a
/// client and any `Conversation` instances constructed from it, without
/// cloning the on-disk state.
#[pyclass(name = "LogStore")]
#[derive(Clone)]
pub struct PyLogStore {
    pub(crate) inner: Arc<llm_store::LogStore>,
}

#[pymethods]
impl PyLogStore {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = llm_store::LogStore::open(Path::new(path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Return the most recent conversation ID in this store (or `None`).
    fn latest_conversation_id(&self) -> PyResult<Option<String>> {
        // Resolve via the inner logs_dir. We can't call the async trait
        // method directly without a runtime, so use the underlying query.
        let summaries = self
            .list_summaries(1)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(summaries.into_iter().next().map(|s| s.id))
    }

    /// List recent conversations as a list of dicts with `id`, `model`,
    /// `name`, `created`.
    #[pyo3(signature = (limit=50))]
    fn list_conversations(&self, py: Python<'_>, limit: usize) -> PyResult<PyObject> {
        let summaries = self
            .list_summaries(limit)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let obj = pythonize(py, &summaries)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(obj.unbind())
    }

    /// Read a conversation by ID. Returns `(meta_dict, responses_list)`.
    fn read_conversation(&self, py: Python<'_>, id: &str) -> PyResult<(PyObject, PyObject)> {
        let (meta, responses) = self
            .inner
            .read_conversation(id)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let meta_obj = pythonize(py, &meta)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .unbind();
        let responses_obj = pythonize(py, &responses)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .unbind();
        Ok((meta_obj, responses_obj))
    }

    /// Directory path backing this store.
    fn path(&self) -> String {
        self.inner.logs_dir().display().to_string()
    }
}

impl PyLogStore {
    fn list_summaries(
        &self,
        limit: usize,
    ) -> llm_core::Result<Vec<llm_store::ConversationSummary>> {
        llm_store::list_conversations(self.inner.logs_dir(), limit)
    }
}
