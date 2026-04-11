use llm_core::types::Message;
use pyo3::prelude::*;
use pythonize::pythonize;

use crate::LlmClient;

/// A multi-turn conversation backed by an in-memory message history.
///
/// Construct from an `LlmClient` to share its registered tools and provider.
#[pyclass]
pub struct Conversation {
    client: Py<LlmClient>,
    messages: Vec<Message>,
    system: Option<String>,
}

#[pymethods]
impl Conversation {
    #[new]
    #[pyo3(signature = (client, *, system=None))]
    fn new(client: Py<LlmClient>, system: Option<String>) -> Self {
        Self {
            client,
            messages: Vec::new(),
            system,
        }
    }

    /// Send a user turn and return the assistant's reply text.
    fn send(&mut self, py: Python<'_>, text: &str) -> PyResult<String> {
        self.messages.push(Message::user(text));
        let client = self.client.borrow(py);
        let (reply_text, new_messages) =
            client.send_messages(&self.messages, self.system.as_deref())?;
        // Replace history with whatever the chain produced (assistant turns + tool messages).
        self.messages = new_messages;
        Ok(reply_text)
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
        let (reply_text, new_messages) = client.send_messages_streaming(
            &self.messages,
            self.system.as_deref(),
            py,
            callback,
        )?;
        self.messages = new_messages;
        Ok(reply_text)
    }

    /// Return the current message history as a list of dicts.
    #[getter]
    fn messages(&self, py: Python<'_>) -> PyResult<PyObject> {
        let bound = pythonize(py, &self.messages)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(bound.unbind())
    }

    /// Clear the message history.
    fn clear(&mut self) {
        self.messages.clear();
    }

    /// Number of messages currently in history.
    fn __len__(&self) -> usize {
        self.messages.len()
    }
}
