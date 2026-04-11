//! Python bindings for programmatic agent configuration and execution.

use std::collections::HashMap;
use std::path::Path;

use llm_core::{AgentConfig, BudgetConfig, RetryConfig};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pythonize::depythonize;
use serde_json::Value;

/// Python-facing wrapper around `llm_core::AgentConfig`.
///
/// Construct explicitly with keyword arguments or load from a TOML file via
/// `AgentConfig.from_toml(path)`. All fields default to the same values as
/// the `[agent]` TOML section used by the CLI.
#[pyclass(name = "AgentConfig")]
#[derive(Clone)]
pub struct PyAgentConfig {
    pub(crate) inner: AgentConfig,
}

#[pymethods]
impl PyAgentConfig {
    #[new]
    #[pyo3(signature = (
        *,
        model=None,
        system_prompt=None,
        tools=None,
        chain_limit=10,
        options=None,
        budget=None,
        retry=None,
        parallel_tools=true,
        max_parallel_tools=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: Option<String>,
        system_prompt: Option<String>,
        tools: Option<Vec<String>>,
        chain_limit: usize,
        options: Option<Py<PyAny>>,
        budget: Option<u64>,
        retry: Option<Py<PyAny>>,
        parallel_tools: bool,
        max_parallel_tools: Option<usize>,
    ) -> PyResult<Self> {
        let options_map: HashMap<String, Value> = match options {
            Some(o) => depythonize(o.bind(py))
                .map_err(|e| PyValueError::new_err(format!("options: {e}")))?,
            None => HashMap::new(),
        };
        let retry_cfg: Option<RetryConfig> = match retry {
            Some(r) => {
                let v: Value = depythonize(r.bind(py))
                    .map_err(|e| PyValueError::new_err(format!("retry: {e}")))?;
                Some(
                    serde_json::from_value(v)
                        .map_err(|e| PyValueError::new_err(format!("retry: {e}")))?,
                )
            }
            None => None,
        };
        let budget_cfg = budget.map(|max_tokens| BudgetConfig {
            max_tokens: Some(max_tokens),
        });

        Ok(Self {
            inner: AgentConfig {
                model,
                system_prompt,
                tools: tools.unwrap_or_default(),
                chain_limit,
                options: options_map,
                budget: budget_cfg,
                retry: retry_cfg,
                parallel_tools,
                max_parallel_tools,
            },
        })
    }

    /// Load an agent config from a TOML file.
    #[classmethod]
    fn from_toml(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let config = AgentConfig::load(Path::new(path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: config })
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[getter]
    fn system_prompt(&self) -> Option<String> {
        self.inner.system_prompt.clone()
    }

    #[getter]
    fn tools(&self) -> Vec<String> {
        self.inner.tools.clone()
    }

    #[getter]
    fn chain_limit(&self) -> usize {
        self.inner.chain_limit
    }

    #[getter]
    fn parallel_tools(&self) -> bool {
        self.inner.parallel_tools
    }

    #[getter]
    fn max_parallel_tools(&self) -> Option<usize> {
        self.inner.max_parallel_tools
    }
}
