use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use llm_core::{BuiltinToolRegistry, Tool, ToolCall, ToolExecutor, ToolResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pythonize::{depythonize, pythonize};
use serde_json::Value;

struct PyToolEntry {
    func: Py<PyAny>,
    schema: Tool,
}

struct PyToolRegistryInner {
    tools: HashMap<String, PyToolEntry>,
    builtins: Option<BuiltinToolRegistry>,
}

/// Shared, mutable registry of Python tool callbacks plus optional builtins.
///
/// Cloning yields another handle to the same underlying registry — used to
/// share state between `LlmClient` and `Conversation`.
#[derive(Clone)]
pub(crate) struct PyToolRegistry {
    inner: Arc<RwLock<PyToolRegistryInner>>,
}

impl PyToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(PyToolRegistryInner {
                tools: HashMap::new(),
                builtins: None,
            })),
        }
    }

    pub fn enable_builtins(&self, version: &'static str) {
        let mut g = self.inner.write().unwrap();
        g.builtins = Some(BuiltinToolRegistry::new(version));
    }

    pub fn register(
        &self,
        name: String,
        func: Py<PyAny>,
        description: String,
        schema: Value,
    ) {
        let mut g = self.inner.write().unwrap();
        let tool = Tool {
            name: name.clone(),
            description,
            input_schema: schema,
        };
        g.tools.insert(name, PyToolEntry { func, schema: tool });
    }

    pub fn list_tools(&self) -> Vec<Tool> {
        let g = self.inner.read().unwrap();
        let mut out: Vec<Tool> = g.tools.values().map(|e| e.schema.clone()).collect();
        if let Some(b) = &g.builtins {
            out.extend(b.list().iter().cloned());
        }
        out
    }

    pub fn has_any(&self) -> bool {
        let g = self.inner.read().unwrap();
        !g.tools.is_empty() || g.builtins.is_some()
    }
}

#[async_trait]
impl ToolExecutor for PyToolRegistry {
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        // Try builtins first (no GIL needed).
        {
            let g = self.inner.read().unwrap();
            if let Some(b) = &g.builtins
                && b.contains(&call.name)
            {
                return b.execute_tool(call);
            }
        }

        Python::with_gil(|py| -> ToolResult {
            let func = {
                let g = self.inner.read().unwrap();
                match g.tools.get(&call.name) {
                    Some(e) => e.func.clone_ref(py),
                    None => {
                        return ToolResult {
                            name: call.name.clone(),
                            output: String::new(),
                            tool_call_id: call.tool_call_id.clone(),
                            error: Some(format!("unknown tool: {}", call.name)),
                        };
                    }
                }
            };

            // ToolCall.arguments is a JSON object whose keys map to function kwargs.
            let kwargs: Bound<'_, PyDict> = match &call.arguments {
                Value::Object(_) => match pythonize(py, &call.arguments) {
                    Ok(obj) => match obj.downcast_into::<PyDict>() {
                        Ok(d) => d,
                        Err(_) => PyDict::new(py),
                    },
                    Err(e) => {
                        return ToolResult {
                            name: call.name.clone(),
                            output: String::new(),
                            tool_call_id: call.tool_call_id.clone(),
                            error: Some(format!("argument conversion failed: {e}")),
                        };
                    }
                },
                _ => PyDict::new(py),
            };

            let args = PyTuple::empty(py);
            match func.call(py, &args, Some(&kwargs)) {
                Ok(val) => {
                    let bound = val.bind(py);
                    let output = if let Ok(s) = bound.extract::<String>() {
                        s
                    } else {
                        match depythonize::<Value>(bound) {
                            Ok(v) => match v {
                                Value::String(s) => s,
                                other => other.to_string(),
                            },
                            Err(_) => bound.str().map(|s| s.to_string()).unwrap_or_default(),
                        }
                    };
                    ToolResult {
                        name: call.name.clone(),
                        output,
                        tool_call_id: call.tool_call_id.clone(),
                        error: None,
                    }
                }
                Err(e) => ToolResult {
                    name: call.name.clone(),
                    output: String::new(),
                    tool_call_id: call.tool_call_id.clone(),
                    error: Some(format!("python tool raised: {e}")),
                },
            }
        })
    }
}

/// Infer a JSON-Schema-style input schema from a Python function's
/// type hints. Supports `str`, `int`, `float`, `bool`, `list`, `dict`.
/// Anything richer raises and forces the caller to pass `schema=` explicitly.
pub(crate) fn infer_schema(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<Value> {
    let inspect = py.import("inspect")?;
    let sig = inspect.call_method1("signature", (func,))?;
    let params: Bound<'_, PyAny> = sig.getattr("parameters")?;
    let items = params.call_method0("items")?;
    let iter = items.try_iter()?;

    let mut properties = serde_json::Map::new();
    let mut required: Vec<Value> = Vec::new();
    let empty = inspect.getattr("Parameter")?.getattr("empty")?;

    for item in iter {
        let item = item?;
        let tup: Bound<'_, PyTuple> = item.downcast_into()?;
        let name: String = tup.get_item(0)?.extract()?;
        let param = tup.get_item(1)?;
        let annotation = param.getattr("annotation")?;
        let default = param.getattr("default")?;

        let json_type = if annotation.is(&empty) {
            // No annotation — assume string and continue.
            "string"
        } else {
            map_py_type(py, &annotation)?
        };

        let mut prop = serde_json::Map::new();
        prop.insert("type".into(), Value::String(json_type.into()));
        properties.insert(name.clone(), Value::Object(prop));

        if default.is(&empty) {
            required.push(Value::String(name));
        }
    }

    Ok(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    }))
}

fn map_py_type(py: Python<'_>, annotation: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    let builtins = py.import("builtins")?;
    let str_t = builtins.getattr("str")?;
    let int_t = builtins.getattr("int")?;
    let float_t = builtins.getattr("float")?;
    let bool_t = builtins.getattr("bool")?;
    let list_t = builtins.getattr("list")?;
    let dict_t = builtins.getattr("dict")?;

    if annotation.is(&str_t) {
        Ok("string")
    } else if annotation.is(&bool_t) {
        // Must check bool before int — bool is a subclass of int in Python.
        Ok("boolean")
    } else if annotation.is(&int_t) {
        Ok("integer")
    } else if annotation.is(&float_t) {
        Ok("number")
    } else if annotation.is(&list_t) {
        Ok("array")
    } else if annotation.is(&dict_t) {
        Ok("object")
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "unsupported type hint: {annotation}. Pass schema= explicitly to override."
        )))
    }
}

/// Decorator object returned by `LlmClient.tool(...)`. Calling it on a
/// function registers that function as a tool and returns it unchanged.
#[pyclass]
pub(crate) struct ToolDecorator {
    pub(crate) registry: PyToolRegistry,
    pub(crate) description: Option<String>,
    pub(crate) schema: Option<Py<PyAny>>,
    pub(crate) name: Option<String>,
}

#[pymethods]
impl ToolDecorator {
    fn __call__(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let bound = func.bind(py);
        let name = match &self.name {
            Some(n) => n.clone(),
            None => bound
                .getattr("__name__")
                .ok()
                .and_then(|n| n.extract::<String>().ok())
                .unwrap_or_else(|| "tool".to_string()),
        };

        let description = self.description.clone().unwrap_or_else(|| {
            bound
                .getattr("__doc__")
                .ok()
                .and_then(|d| d.extract::<String>().ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| name.clone())
        });

        let schema = if let Some(s) = &self.schema {
            depythonize::<Value>(s.bind(py))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            infer_schema(py, bound)?
        };

        self.registry.register(name, func.clone_ref(py), description, schema);
        Ok(func)
    }
}
