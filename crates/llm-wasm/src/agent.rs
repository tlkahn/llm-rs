//! WASM bindings for programmatic agent configuration and execution.

use llm_core::AgentConfig;
use wasm_bindgen::prelude::*;

/// WASM-facing wrapper around `llm_core::AgentConfig`.
///
/// Construct from a JS object matching the TOML schema:
/// `{ model?, systemPrompt?, tools?, chainLimit?, options?, budget?, retry?, parallelTools?, maxParallelTools? }`.
///
/// The JS object is deserialized via `serde_wasm_bindgen`, so field names
/// use camelCase automatically via `#[serde(rename_all = "snake_case")]` on
/// `AgentConfig`... actually `AgentConfig` uses snake_case, so the JS
/// object must use snake_case field names too (same as TOML).
#[wasm_bindgen(js_name = "AgentConfig")]
#[derive(Clone)]
pub struct WasmAgentConfig {
    pub(crate) inner: AgentConfig,
}

#[wasm_bindgen(js_class = "AgentConfig")]
impl WasmAgentConfig {
    /// Build an `AgentConfig` from a JS object. Field names are snake_case,
    /// matching the TOML schema used by `llm agent run` on the CLI.
    #[wasm_bindgen(constructor)]
    pub fn new(spec: JsValue) -> Result<WasmAgentConfig, JsError> {
        // Allow an empty / undefined spec for an all-defaults config.
        if spec.is_undefined() || spec.is_null() {
            return Ok(Self {
                inner: AgentConfig::default(),
            });
        }
        let config: AgentConfig = serde_wasm_bindgen::from_value(spec)
            .map_err(|e| JsError::new(&format!("AgentConfig: {e}")))?;
        Ok(Self { inner: config })
    }
}
