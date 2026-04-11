use std::sync::{Arc, Mutex};

use llm_core::types::{Message, Options};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pythonize::pythonize;

use crate::log_store_py::PyLogStore;
use crate::response_build::{synthesize_response, ResponseInputs};
use crate::{LlmClient, TurnOutput};

/// A multi-turn conversation backed by an in-memory message history.
///
/// Construct from an `LlmClient` to share its registered tools and provider.
/// If the client was constructed with a `log_store`, the conversation
/// inherits it and each `send` call appends a `Response` to the log. The
/// conversation can be loaded from a persisted cid via `Conversation.load`
/// to resume a prior session.
#[pyclass]
pub struct Conversation {
    client: Py<LlmClient>,
    messages: Vec<Message>,
    system: Option<String>,
    log_store: Mutex<Option<Arc<llm_store::LogStore>>>,
    conversation_id: Mutex<Option<String>>,
}

#[pymethods]
impl Conversation {
    #[new]
    #[pyo3(signature = (client, *, system=None))]
    fn new(py: Python<'_>, client: Py<LlmClient>, system: Option<String>) -> Self {
        // Inherit the client's log store if it has one.
        let inherited = {
            let c = client.borrow(py);
            c.log_store.clone()
        };
        Self {
            client,
            messages: Vec::new(),
            system,
            log_store: Mutex::new(inherited),
            conversation_id: Mutex::new(None),
        }
    }

    /// Attach a `LogStore` to this conversation so future `send` calls are
    /// persisted. Errors if the conversation already has messages — to
    /// resume a persisted conversation, use `Conversation.load` instead.
    fn persist_to(&self, store: &PyLogStore) -> PyResult<()> {
        if !self.messages.is_empty() {
            return Err(PyValueError::new_err(
                "persist_to must be called before first send",
            ));
        }
        *self.log_store.lock().unwrap() = Some(store.inner.clone());
        *self.conversation_id.lock().unwrap() = None;
        Ok(())
    }

    /// Load a persisted conversation by ID.
    ///
    /// Reads the cid from `store`, reconstructs the message history via
    /// `reconstruct_messages`, and seeds the system prompt from the first
    /// stored response so it survives the reload. Future `send` calls on
    /// the returned conversation append to the same cid.
    ///
    /// Note: `reconstruct_messages` is lossy across multi-iteration chains
    /// (intermediate assistant reasoning text between tool turns is
    /// dropped).
    #[classmethod]
    fn load(
        _cls: &Bound<'_, pyo3::types::PyType>,
        client: Py<LlmClient>,
        store: &PyLogStore,
        cid: &str,
    ) -> PyResult<Self> {
        let (_, responses) = store
            .inner
            .read_conversation(cid)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let messages = llm_store::reconstruct_messages(&responses);
        let system = responses
            .first()
            .and_then(|r| r.system.clone());
        Ok(Self {
            client,
            messages,
            system,
            log_store: Mutex::new(Some(store.inner.clone())),
            conversation_id: Mutex::new(Some(cid.to_string())),
        })
    }

    /// Send a user turn and return the assistant's reply text. If the
    /// conversation has an attached log store, the turn is appended.
    fn send(&mut self, py: Python<'_>, text: &str) -> PyResult<String> {
        self.messages.push(Message::user(text));
        let client = self.client.borrow(py);
        let turn = client.send_messages(&self.messages, self.system.as_deref())?;
        self.messages = turn.messages.clone();
        self.log_turn(&client, text, &turn)?;
        Ok(turn.text)
    }

    /// Send a user turn and stream text chunks via the provided callback.
    /// Returns the full assistant reply text.
    #[pyo3(signature = (text, callback))]
    fn send_stream(
        &mut self,
        py: Python<'_>,
        text: &str,
        callback: Py<PyAny>,
    ) -> PyResult<String> {
        self.messages.push(Message::user(text));
        let client = self.client.borrow(py);
        let turn = client.send_messages_streaming(
            &self.messages,
            self.system.as_deref(),
            py,
            callback,
        )?;
        self.messages = turn.messages.clone();
        self.log_turn(&client, text, &turn)?;
        Ok(turn.text)
    }

    /// Return the current message history as a list of dicts.
    #[getter]
    fn messages(&self, py: Python<'_>) -> PyResult<PyObject> {
        let bound = pythonize(py, &self.messages)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(bound.unbind())
    }

    /// The active persisted conversation ID, if any.
    #[getter]
    fn conversation_id(&self) -> Option<String> {
        self.conversation_id.lock().unwrap().clone()
    }

    /// Clear the message history. Does not clear the attached log store.
    fn clear(&mut self) {
        self.messages.clear();
        *self.conversation_id.lock().unwrap() = None;
    }

    /// Number of messages currently in history.
    fn __len__(&self) -> usize {
        self.messages.len()
    }
}

impl Conversation {
    fn log_turn(
        &self,
        client: &LlmClient,
        user_text: &str,
        turn: &TurnOutput,
    ) -> PyResult<()> {
        let Some(store) = self.log_store.lock().unwrap().clone() else {
            return Ok(());
        };
        let response = synthesize_response(ResponseInputs {
            model: client.model_name(),
            prompt: user_text,
            system: self.system.as_deref(),
            options: Options::new(),
            chunks: &turn.chunks,
            tool_calls: turn.tool_calls.clone(),
            tool_results: turn.tool_results.clone(),
            total_usage: turn.total_usage.clone(),
            schema: None,
            schema_id: None,
            duration_ms: turn.duration_ms,
        });
        let prev_cid = self.conversation_id.lock().unwrap().clone();
        let new_cid = LlmClient::log_response_external(
            &store,
            prev_cid.as_deref(),
            client.model_name(),
            &response,
        )?;
        *self.conversation_id.lock().unwrap() = Some(new_cid);
        Ok(())
    }
}
