use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use llm_core::{Tool, ToolCall, ToolExecutor, ToolResult};

use super::discovery;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Executor for external subprocess tools discovered on PATH.
pub struct ExternalToolExecutor {
    /// tool name → (binary path, Tool schema)
    tools: HashMap<String, (PathBuf, Tool)>,
    timeout: Duration,
}

impl ExternalToolExecutor {
    /// Discover external tools on PATH and fetch their schemas.
    pub async fn discover() -> llm_core::Result<Self> {
        Self::discover_with_timeout(DEFAULT_TIMEOUT).await
    }

    /// Discover with a custom timeout.
    pub async fn discover_with_timeout(timeout: Duration) -> llm_core::Result<Self> {
        let binaries = discovery::discover_tools();
        let schemas = discovery::fetch_all_tool_schemas(&binaries, timeout).await;
        let mut tools = HashMap::new();
        for (path, tool) in schemas {
            tools.insert(tool.name.clone(), (path, tool));
        }
        Ok(Self { tools, timeout })
    }

    /// Create from a pre-built map (for testing).
    #[cfg(test)]
    pub fn from_map(tools: HashMap<String, (PathBuf, Tool)>, timeout: Duration) -> Self {
        Self { tools, timeout }
    }

    /// Get the binary path for a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&(PathBuf, Tool)> {
        self.tools.get(name)
    }

    /// List all discovered tools.
    pub fn list_tools(&self) -> Vec<(&str, &PathBuf, &Tool)> {
        self.tools
            .iter()
            .map(|(name, (path, tool))| (name.as_str(), path, tool))
            .collect()
    }

}

#[async_trait]
impl ToolExecutor for ExternalToolExecutor {
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let (binary, _) = match self.tools.get(&call.name) {
            Some(entry) => entry,
            None => {
                return ToolResult {
                    name: call.name.clone(),
                    output: String::new(),
                    tool_call_id: call.tool_call_id.clone(),
                    error: Some(format!("unknown external tool: {}", call.name)),
                };
            }
        };

        let stdin_data = call.arguments.to_string();

        let result = tokio::time::timeout(self.timeout, async {
            let mut child = match tokio::process::Command::new(binary)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    return ToolResult {
                        name: call.name.clone(),
                        output: String::new(),
                        tool_call_id: call.tool_call_id.clone(),
                        error: Some(format!("failed to spawn {}: {e}", binary.display())),
                    };
                }
            };

            // Write arguments JSON to stdin
            if let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                let _ = stdin.write_all(stdin_data.as_bytes()).await;
                drop(stdin);
            }

            match child.wait_with_output().await {
                Ok(output) => {
                    if output.status.success() {
                        ToolResult {
                            name: call.name.clone(),
                            output: String::from_utf8_lossy(&output.stdout).to_string(),
                            tool_call_id: call.tool_call_id.clone(),
                            error: None,
                        }
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                        ToolResult {
                            name: call.name.clone(),
                            output: String::new(),
                            tool_call_id: call.tool_call_id.clone(),
                            error: Some(if stderr.is_empty() {
                                format!("tool exited with {}", output.status)
                            } else {
                                stderr
                            }),
                        }
                    }
                }
                Err(e) => ToolResult {
                    name: call.name.clone(),
                    output: String::new(),
                    tool_call_id: call.tool_call_id.clone(),
                    error: Some(format!("tool execution error: {e}")),
                },
            }
        })
        .await;

        match result {
            Ok(tool_result) => tool_result,
            Err(_) => ToolResult {
                name: call.name.clone(),
                output: String::new(),
                tool_call_id: call.tool_call_id.clone(),
                error: Some(format!("tool {} timed out", call.name)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool_script(dir: &std::path::Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    fn make_upper_tool(dir: &std::path::Path) -> (PathBuf, Tool) {
        let path = make_tool_script(
            dir,
            "llm-tool-upper",
            r#"#!/bin/sh
if [ "$1" = "--schema" ]; then
    echo '{"name":"upper","description":"Uppercase text","input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}'
    exit 0
fi
read input
echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'].upper())"
"#,
        );
        let tool = Tool {
            name: "upper".into(),
            description: "Uppercase text".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }),
        };
        (path, tool)
    }

    #[tokio::test]
    async fn execute_tool_success() {
        let dir = tempfile::TempDir::new().unwrap();
        let (path, tool) = make_upper_tool(dir.path());
        let mut tools = HashMap::new();
        tools.insert("upper".into(), (path, tool));
        let executor = ExternalToolExecutor::from_map(tools, Duration::from_secs(10));

        let call = ToolCall {
            name: "upper".into(),
            arguments: serde_json::json!({"text": "hello"}),
            tool_call_id: Some("tc_1".into()),
        };

        let result = executor.execute(&call).await;
        assert!(result.error.is_none(), "got error: {:?}", result.error);
        assert_eq!(result.output.trim(), "HELLO");
        assert_eq!(result.tool_call_id.as_deref(), Some("tc_1"));
    }

    #[tokio::test]
    async fn execute_tool_nonzero_exit() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = make_tool_script(
            dir.path(),
            "llm-tool-fail",
            "#!/bin/sh\necho 'something went wrong' >&2\nexit 1",
        );
        let tool = Tool {
            name: "fail".into(),
            description: "Always fails".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let mut tools = HashMap::new();
        tools.insert("fail".into(), (path, tool));
        let executor = ExternalToolExecutor::from_map(tools, Duration::from_secs(10));

        let call = ToolCall {
            name: "fail".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };

        let result = executor.execute(&call).await;
        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("something went wrong"));
    }

    #[tokio::test]
    async fn execute_tool_timeout() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = make_tool_script(
            dir.path(),
            "llm-tool-slow",
            "#!/bin/sh\nsleep 10",
        );
        let tool = Tool {
            name: "slow".into(),
            description: "Slow tool".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let mut tools = HashMap::new();
        tools.insert("slow".into(), (path, tool));
        let executor = ExternalToolExecutor::from_map(tools, Duration::from_millis(100));

        let call = ToolCall {
            name: "slow".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };

        let result = executor.execute(&call).await;
        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn execute_tool_empty_stdout() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = make_tool_script(
            dir.path(),
            "llm-tool-empty",
            "#!/bin/sh\nexit 0",
        );
        let tool = Tool {
            name: "empty".into(),
            description: "Empty output".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let mut tools = HashMap::new();
        tools.insert("empty".into(), (path, tool));
        let executor = ExternalToolExecutor::from_map(tools, Duration::from_secs(10));

        let call = ToolCall {
            name: "empty".into(),
            arguments: serde_json::json!({}),
            tool_call_id: Some("tc_2".into()),
        };

        let result = executor.execute(&call).await;
        assert!(result.error.is_none());
        assert!(result.output.is_empty());
        assert_eq!(result.tool_call_id.as_deref(), Some("tc_2"));
    }

    #[tokio::test]
    async fn execute_unknown_tool() {
        let executor =
            ExternalToolExecutor::from_map(HashMap::new(), Duration::from_secs(10));

        let call = ToolCall {
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
            tool_call_id: None,
        };

        let result = executor.execute(&call).await;
        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("unknown external tool"));
    }
}
