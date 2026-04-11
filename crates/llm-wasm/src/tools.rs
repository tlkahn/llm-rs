use std::cell::RefCell;
use std::rc::Rc;

use async_trait::async_trait;
use llm_core::{BuiltinToolRegistry, Tool, ToolCall, ToolExecutor, ToolResult};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// One JS-side tool: schema + executor function.
struct JsTool {
    schema: Tool,
    func: js_sys::Function,
}

struct WasmToolRegistryInner {
    tools: Vec<JsTool>,
    builtins: Option<BuiltinToolRegistry>,
}

/// Shared, mutable registry of WASM tool callbacks.
///
/// Cloning yields another handle to the same underlying registry — used to
/// share state between `LlmClient` and `Conversation`. Single-threaded by
/// virtue of running in WebAssembly.
#[derive(Clone)]
pub(crate) struct WasmToolRegistry {
    inner: Rc<RefCell<WasmToolRegistryInner>>,
}

#[derive(Deserialize)]
struct JsToolSpec {
    name: String,
    description: String,
    #[serde(default, alias = "inputSchema", alias = "input_schema")]
    input_schema: serde_json::Value,
}

impl WasmToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(WasmToolRegistryInner {
                tools: Vec::new(),
                builtins: None,
            })),
        }
    }

    pub fn enable_builtins(&self, version: &'static str) {
        self.inner.borrow_mut().builtins =
            Some(BuiltinToolRegistry::new(version));
    }

    /// Register a tool from a JS object `{ name, description, inputSchema, execute }`.
    pub fn register(&self, spec: JsValue) -> Result<(), JsError> {
        // Pull out the function field first since it can't be deserialized via serde.
        let obj: js_sys::Object = spec
            .clone()
            .dyn_into()
            .map_err(|_| JsError::new("tool spec must be a plain object"))?;
        let func = js_sys::Reflect::get(&obj, &JsValue::from_str("execute"))
            .map_err(|_| JsError::new("tool spec missing 'execute' function"))?;
        let func: js_sys::Function = func
            .dyn_into()
            .map_err(|_| JsError::new("tool spec 'execute' must be a function"))?;

        let parsed: JsToolSpec = serde_wasm_bindgen::from_value(spec)
            .map_err(|e| JsError::new(&format!("invalid tool spec: {e}")))?;

        let input_schema = if parsed.input_schema.is_null() {
            serde_json::json!({"type": "object", "properties": {}})
        } else {
            parsed.input_schema
        };

        self.inner.borrow_mut().tools.push(JsTool {
            schema: Tool {
                name: parsed.name,
                description: parsed.description,
                input_schema,
            },
            func,
        });
        Ok(())
    }

    pub fn list_tools(&self) -> Vec<Tool> {
        let g = self.inner.borrow();
        let mut out: Vec<Tool> = g.tools.iter().map(|t| t.schema.clone()).collect();
        if let Some(b) = &g.builtins {
            out.extend(b.list().iter().cloned());
        }
        out
    }

    pub fn has_any(&self) -> bool {
        let g = self.inner.borrow();
        !g.tools.is_empty() || g.builtins.is_some()
    }
}

#[async_trait(?Send)]
impl ToolExecutor for WasmToolRegistry {
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        // Builtin first.
        {
            let g = self.inner.borrow();
            if let Some(b) = &g.builtins
                && b.contains(&call.name)
            {
                return b.execute_tool(call);
            }
        }

        // Look up the JS tool, clone the function out of the borrow, and
        // drop the borrow before awaiting (RefCell + await don't mix).
        let func = {
            let g = self.inner.borrow();
            match g.tools.iter().find(|t| t.schema.name == call.name) {
                Some(t) => t.func.clone(),
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

        // Convert arguments to a JS value.
        let args_js = match serde_wasm_bindgen::to_value(&call.arguments) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    name: call.name.clone(),
                    output: String::new(),
                    tool_call_id: call.tool_call_id.clone(),
                    error: Some(format!("argument conversion failed: {e}")),
                };
            }
        };

        let this = JsValue::NULL;
        let returned = match func.call1(&this, &args_js) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    name: call.name.clone(),
                    output: String::new(),
                    tool_call_id: call.tool_call_id.clone(),
                    error: Some(js_value_to_string(&e)),
                };
            }
        };

        // The JS executor may return either a value directly or a Promise.
        let resolved = if let Some(promise) = returned.dyn_ref::<js_sys::Promise>() {
            match JsFuture::from(promise.clone()).await {
                Ok(v) => v,
                Err(e) => {
                    return ToolResult {
                        name: call.name.clone(),
                        output: String::new(),
                        tool_call_id: call.tool_call_id.clone(),
                        error: Some(js_value_to_string(&e)),
                    };
                }
            }
        } else {
            returned
        };

        ToolResult {
            name: call.name.clone(),
            output: js_value_to_string(&resolved),
            tool_call_id: call.tool_call_id.clone(),
            error: None,
        }
    }
}

fn js_value_to_string(v: &JsValue) -> String {
    if let Some(s) = v.as_string() {
        return s;
    }
    // Try JSON.stringify for non-string values.
    match js_sys::JSON::stringify(v) {
        Ok(s) => s.as_string().unwrap_or_default(),
        Err(_) => String::new(),
    }
}
