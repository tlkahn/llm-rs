use async_trait::async_trait;
use clap::Subcommand;
use llm_core::{BuiltinToolRegistry, ToolCall, ToolExecutor, ToolResult};

use crate::subprocess::tool::ExternalToolExecutor;

#[derive(Subcommand)]
pub enum ToolsCommand {
    /// List available tools (built-in and external)
    List,
}

pub fn builtin_registry() -> BuiltinToolRegistry {
    BuiltinToolRegistry::new(env!("CARGO_PKG_VERSION"))
}

pub async fn run(command: &ToolsCommand) -> llm_core::Result<()> {
    match command {
        ToolsCommand::List => {
            let registry = builtin_registry();
            for tool in registry.list() {
                println!("{}: {}", tool.name, tool.description);
            }

            // Show external tools from PATH
            let external = ExternalToolExecutor::discover().await?;
            let mut ext_tools = external.list_tools();
            ext_tools.sort_by_key(|(name, _, _)| name.to_string());
            for (name, path, tool) in &ext_tools {
                println!("{name}: {} ({})", tool.description, path.display());
            }

            Ok(())
        }
    }
}

/// CLI tool executor that wraps `BuiltinToolRegistry` and optionally
/// delegates to external subprocess tools.
pub struct CliToolExecutor {
    pub debug: bool,
    pub approve: bool,
    pub external: Option<ExternalToolExecutor>,
    builtins: BuiltinToolRegistry,
}

impl CliToolExecutor {
    pub fn new(debug: bool, approve: bool) -> Self {
        Self {
            debug,
            approve,
            external: None,
            builtins: builtin_registry(),
        }
    }

    pub fn with_external(mut self, external: ExternalToolExecutor) -> Self {
        self.external = Some(external);
        self
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

        // Try builtin first
        let result = self.builtins.execute_tool(call);
        let result = if result.error.as_ref().is_some_and(|e| e.contains("unknown tool")) {
            // Not a builtin — try external
            if let Some(ext) = &self.external {
                ext.execute(call).await
            } else {
                result
            }
        } else {
            result
        };

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
