use async_trait::async_trait;
use clap::Subcommand;
use llm_core::{Tool, ToolCall, ToolExecutor, ToolResult};

#[derive(Subcommand)]
pub enum ToolsCommand {
    /// List available built-in tools
    List,
}

pub fn run(command: &ToolsCommand) -> llm_core::Result<()> {
    match command {
        ToolsCommand::List => {
            let registry = BuiltinToolRegistry::new();
            for tool in registry.list() {
                println!("{}: {}", tool.name, tool.description);
            }
            Ok(())
        }
    }
}

pub struct BuiltinToolRegistry {
    tools: Vec<Tool>,
}

impl BuiltinToolRegistry {
    pub fn new() -> Self {
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
        }
    }

    pub fn list(&self) -> &[Tool] {
        &self.tools
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    fn execute_tool(call: &ToolCall) -> ToolResult {
        let output = match call.name.as_str() {
            "llm_version" => env!("CARGO_PKG_VERSION").to_string(),
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

/// CLI tool executor that wraps BuiltinToolRegistry.
pub struct CliToolExecutor {
    pub debug: bool,
    pub approve: bool,
}

impl CliToolExecutor {
    pub fn new(debug: bool, approve: bool) -> Self {
        Self { debug, approve }
    }
}

#[async_trait]
impl ToolExecutor for CliToolExecutor {
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        if self.debug {
            eprintln!(
                "Tool call: {} (id: {})",
                call.name,
                call.tool_call_id.as_deref().unwrap_or("none")
            );
            eprintln!("Arguments: {}", call.arguments);
        }

        if self.approve {
            eprint!("Execute tool {}? [y/N] ", call.name);
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() {
                let input = input.trim().to_lowercase();
                if input != "y" && input != "yes" {
                    return ToolResult {
                        name: call.name.clone(),
                        output: String::new(),
                        tool_call_id: call.tool_call_id.clone(),
                        error: Some("user declined".into()),
                    };
                }
            }
        }

        let result = BuiltinToolRegistry::execute_tool(call);

        if self.debug {
            if let Some(err) = &result.error {
                eprintln!("Tool error: {err}");
            } else {
                eprintln!("Tool result: {}", result.output);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_two_builtin_tools() {
        let registry = BuiltinToolRegistry::new();
        assert_eq!(registry.list().len(), 2);
    }

    #[test]
    fn llm_version_returns_version_string() {
        let call = ToolCall {
            name: "llm_version".into(),
            arguments: serde_json::json!({}),
            tool_call_id: Some("tc_1".into()),
        };
        let result = BuiltinToolRegistry::execute_tool(&call);
        assert!(result.error.is_none());
        assert!(!result.output.is_empty());
        assert_eq!(result.output, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn llm_time_returns_time_info() {
        let call = ToolCall {
            name: "llm_time".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };
        let result = BuiltinToolRegistry::execute_tool(&call);
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
        let result = BuiltinToolRegistry::execute_tool(&call);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("unknown tool"));
    }

    #[test]
    fn registry_get_finds_tool() {
        let registry = BuiltinToolRegistry::new();
        assert!(registry.get("llm_version").is_some());
        assert!(registry.get("llm_time").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}
