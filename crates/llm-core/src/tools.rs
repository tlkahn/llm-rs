use crate::types::{Tool, ToolCall, ToolResult};

/// Registry of built-in tools (`llm_version`, `llm_time`).
///
/// Lifted out of `llm-cli` so the WASM and Python bindings can offer the
/// same builtins without pulling in CLI-only concerns. The version string
/// is taken at construction time so each caller reports its own crate
/// version (`env!("CARGO_PKG_VERSION")`).
pub struct BuiltinToolRegistry {
    tools: Vec<Tool>,
    version: &'static str,
}

impl BuiltinToolRegistry {
    #[must_use]
    pub fn new(version: &'static str) -> Self {
        Self {
            tools: vec![
                Tool {
                    name: "llm_version".into(),
                    description: "Returns the current LLM CLI version".into(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {},
                    }),
                },
                Tool {
                    name: "llm_time".into(),
                    description: "Returns the current date and time".into(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {},
                    }),
                },
            ],
            version,
        }
    }

    #[must_use]
    pub fn list(&self) -> &[Tool] {
        &self.tools
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Returns `true` if `name` is a known builtin tool.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Execute a builtin tool call. Returns an error result for unknown tools.
    #[must_use]
    pub fn execute_tool(&self, call: &ToolCall) -> ToolResult {
        let output = match call.name.as_str() {
            "llm_version" => self.version.to_string(),
            "llm_time" => {
                let utc = chrono::Utc::now();
                let local = chrono::Local::now();
                let tz = local.format("%Z").to_string();
                serde_json::json!({
                    "utc_time": utc.to_rfc3339(),
                    "local_time": local.to_rfc3339(),
                    "timezone": tz,
                })
                .to_string()
            }
            _ => {
                return ToolResult {
                    name: call.name.clone(),
                    output: String::new(),
                    tool_call_id: call.tool_call_id.clone(),
                    error: Some(format!("unknown tool: {}", call.name)),
                };
            }
        };

        ToolResult {
            name: call.name.clone(),
            output,
            tool_call_id: call.tool_call_id.clone(),
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> BuiltinToolRegistry {
        BuiltinToolRegistry::new("9.9.9-test")
    }

    #[test]
    fn registry_has_two_builtin_tools() {
        assert_eq!(registry().list().len(), 2);
    }

    #[test]
    fn llm_version_returns_constructor_version() {
        let call = ToolCall {
            name: "llm_version".into(),
            arguments: serde_json::json!({}),
            tool_call_id: Some("tc_1".into()),
        };
        let result = registry().execute_tool(&call);
        assert!(result.error.is_none());
        assert_eq!(result.output, "9.9.9-test");
    }

    #[test]
    fn llm_time_returns_time_info() {
        let call = ToolCall {
            name: "llm_time".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };
        let result = registry().execute_tool(&call);
        assert!(result.error.is_none());
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed.get("utc_time").is_some());
        assert!(parsed.get("local_time").is_some());
        assert!(parsed.get("timezone").is_some());
    }

    #[test]
    fn unknown_tool_returns_error_result() {
        let call = ToolCall {
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };
        let result = registry().execute_tool(&call);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("unknown tool"));
    }

    #[test]
    fn registry_get_finds_tool() {
        let r = registry();
        assert!(r.get("llm_version").is_some());
        assert!(r.get("llm_time").is_some());
        assert!(r.get("nonexistent").is_none());
        assert!(r.contains("llm_time"));
    }
}
