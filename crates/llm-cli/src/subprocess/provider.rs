use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use llm_core::{Chunk, LlmError, ModelInfo, Prompt, Provider, ResponseStream};
use tokio::io::AsyncBufReadExt;

use super::discovery::{self, KeyRequirement};
use super::protocol::{ProtocolChunk, ProviderRequest, ProviderResponse};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// A provider backed by an external subprocess (`llm-provider-*`).
pub struct SubprocessProvider {
    binary: PathBuf,
    provider_id: String,
    model_list: Vec<ModelInfo>,
    key_requirement: KeyRequirement,
}

impl SubprocessProvider {
    /// Create a SubprocessProvider by fetching metadata from the binary.
    pub async fn from_binary(path: PathBuf) -> llm_core::Result<Self> {
        Self::from_binary_with_timeout(path, DEFAULT_TIMEOUT).await
    }

    /// Create with custom timeout for metadata fetching.
    pub async fn from_binary_with_timeout(
        path: PathBuf,
        timeout: Duration,
    ) -> llm_core::Result<Self> {
        let provider_id = discovery::fetch_provider_id(&path, timeout).await?;
        let model_list = discovery::fetch_provider_models(&path, timeout).await?;
        let key_requirement = discovery::fetch_provider_key_info(&path, timeout).await?;

        Ok(Self {
            binary: path,
            provider_id,
            model_list,
            key_requirement,
        })
    }

    /// Create from pre-fetched metadata (for testing).
    #[cfg(test)]
    pub fn new(
        binary: PathBuf,
        provider_id: String,
        model_list: Vec<ModelInfo>,
        key_requirement: KeyRequirement,
    ) -> Self {
        Self {
            binary,
            provider_id,
            model_list,
            key_requirement,
        }
    }
}

#[async_trait]
impl Provider for SubprocessProvider {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.model_list.clone()
    }

    fn needs_key(&self) -> Option<&str> {
        if self.key_requirement.needed {
            Some(self.provider_id.as_str())
        } else {
            None
        }
    }

    fn key_env_var(&self) -> Option<&str> {
        self.key_requirement.env_var.as_deref()
    }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> llm_core::Result<ResponseStream> {
        let request = ProviderRequest {
            model: model.to_string(),
            prompt: prompt.clone(),
            key: key.map(String::from),
            stream,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| LlmError::Provider(format!("failed to serialize request: {e}")))?;

        let mut child = tokio::process::Command::new(&self.binary)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                LlmError::Provider(format!(
                    "failed to spawn {}: {e}",
                    self.binary.display()
                ))
            })?;

        // Write request to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(request_json.as_bytes())
                .await
                .map_err(|e| LlmError::Provider(format!("failed to write to stdin: {e}")))?;
            drop(stdin);
        }

        if stream {
            // Streaming: read stdout line by line, parse each as ProtocolChunk
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| LlmError::Provider("no stdout from subprocess".into()))?;

            let reader = tokio::io::BufReader::new(stdout);
            let mut lines = reader.lines();

            let stream = async_stream::try_stream! {
                while let Some(line) = lines.next_line().await
                    .map_err(|e| LlmError::Provider(format!("failed to read stdout: {e}")))?
                {
                    let line = line.trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    let pc: ProtocolChunk = serde_json::from_str(&line)
                        .map_err(|e| LlmError::Provider(
                            format!("malformed JSONL from subprocess: {e}: {line}")
                        ))?;
                    yield Chunk::from(pc);
                }

                // Check exit status
                let status = child.wait().await
                    .map_err(|e| LlmError::Provider(format!("subprocess wait error: {e}")))?;
                if !status.success() {
                    // We already yielded chunks; just warn via a provider error if non-zero
                    // but only if we haven't yielded anything meaningful
                }
            };

            Ok(Box::pin(stream))
        } else {
            // Non-streaming: read all stdout, parse as ProviderResponse
            let output = child.wait_with_output().await.map_err(|e| {
                LlmError::Provider(format!("subprocess execution error: {e}"))
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                return Err(LlmError::Provider(format!(
                    "subprocess provider exited with {}: {}",
                    output.status,
                    if stderr.is_empty() {
                        "no error message"
                    } else {
                        &stderr
                    }
                )));
            }

            let resp: ProviderResponse =
                serde_json::from_slice(&output.stdout).map_err(|e| {
                    LlmError::Provider(format!("invalid response from subprocess: {e}"))
                })?;

            // Convert to chunks
            let mut chunks: Vec<Result<Chunk, LlmError>> = Vec::new();
            if !resp.text.is_empty() {
                chunks.push(Ok(Chunk::Text(resp.text)));
            }
            for tc in &resp.tool_calls {
                chunks.push(Ok(Chunk::ToolCallStart {
                    name: tc.name.clone(),
                    id: tc.tool_call_id.clone(),
                }));
                chunks.push(Ok(Chunk::ToolCallDelta {
                    content: tc.arguments.to_string(),
                }));
            }
            if let Some(usage) = resp.usage {
                chunks.push(Ok(Chunk::Usage(llm_core::Usage {
                    input: Some(usage.input),
                    output: Some(usage.output),
                    details: None,
                })));
            }
            chunks.push(Ok(Chunk::Done));

            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use llm_core::{collect_text, collect_tool_calls, collect_usage};

    fn make_provider_script(dir: &std::path::Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    fn make_echo_provider(dir: &std::path::Path) -> SubprocessProvider {
        let script = make_provider_script(
            dir,
            "llm-provider-echo",
            r#"#!/bin/sh
# Read the request from stdin
read request

# Extract stream flag and text
stream=$(echo "$request" | python3 -c "import sys,json; print(json.load(sys.stdin)['stream'])")
text=$(echo "$request" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt']['text'])")

if [ "$stream" = "True" ]; then
    echo "{\"type\":\"text\",\"content\":\"echo: $text\"}"
    echo "{\"type\":\"usage\",\"input\":5,\"output\":10}"
    echo "{\"type\":\"done\"}"
else
    echo "{\"text\":\"echo: $text\",\"tool_calls\":[],\"usage\":{\"input\":5,\"output\":10}}"
fi
"#,
        );

        SubprocessProvider::new(
            script,
            "echo".into(),
            vec![ModelInfo {
                id: "echo-model".into(),
                can_stream: true,
                supports_tools: false,
                supports_schema: false,
                attachment_types: Vec::new(),
            }],
            KeyRequirement {
                needed: false,
                env_var: None,
            },
        )
    }

    #[test]
    fn provider_id_returns_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider = make_echo_provider(dir.path());
        assert_eq!(provider.id(), "echo");
    }

    #[test]
    fn provider_models_returns_list() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider = make_echo_provider(dir.path());
        let models = provider.models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "echo-model");
    }

    #[test]
    fn provider_needs_key_reflects_requirement() {
        let dir = tempfile::TempDir::new().unwrap();
        let no_key = make_echo_provider(dir.path());
        assert_eq!(no_key.needs_key(), None);
        assert_eq!(no_key.key_env_var(), None);

        let with_key = SubprocessProvider::new(
            dir.path().join("dummy"),
            "test".into(),
            vec![],
            KeyRequirement {
                needed: true,
                env_var: Some("MY_KEY".into()),
            },
        );
        assert_eq!(with_key.needs_key(), Some("test"));
        assert_eq!(with_key.key_env_var(), Some("MY_KEY"));
    }

    #[tokio::test]
    async fn non_streaming_execution() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider = make_echo_provider(dir.path());
        let prompt = Prompt::new("hello");

        let stream = provider
            .execute("echo-model", &prompt, None, false)
            .await
            .unwrap();
        let chunks: Vec<Chunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text = collect_text(&chunks);
        assert_eq!(text, "echo: hello");
        let usage = collect_usage(&chunks).unwrap();
        assert_eq!(usage.input, Some(5));
        assert_eq!(usage.output, Some(10));
    }

    #[tokio::test]
    async fn streaming_execution() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider = make_echo_provider(dir.path());
        let prompt = Prompt::new("world");

        let stream = provider
            .execute("echo-model", &prompt, None, true)
            .await
            .unwrap();
        let chunks: Vec<Chunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text = collect_text(&chunks);
        assert_eq!(text, "echo: world");
        let usage = collect_usage(&chunks).unwrap();
        assert_eq!(usage.input, Some(5));
    }

    #[tokio::test]
    async fn non_streaming_error_on_nonzero_exit() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = make_provider_script(
            dir.path(),
            "llm-provider-fail",
            "#!/bin/sh\necho 'provider error' >&2\nexit 1",
        );
        let provider = SubprocessProvider::new(
            script,
            "fail".into(),
            vec![ModelInfo::new("fail-model")],
            KeyRequirement {
                needed: false,
                env_var: None,
            },
        );

        let result = provider
            .execute("fail-model", &Prompt::new("test"), None, false)
            .await;
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(err.contains("provider error"), "got: {err}");
    }

    #[tokio::test]
    async fn streaming_malformed_jsonl() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = make_provider_script(
            dir.path(),
            "llm-provider-bad",
            "#!/bin/sh\nread _\necho 'not json'",
        );
        let provider = SubprocessProvider::new(
            script,
            "bad".into(),
            vec![ModelInfo::new("bad-model")],
            KeyRequirement {
                needed: false,
                env_var: None,
            },
        );

        let stream = provider
            .execute("bad-model", &Prompt::new("test"), None, true)
            .await
            .unwrap();
        let results: Vec<_> = stream.collect().await;
        // Should have at least one error
        assert!(results.iter().any(|r| r.is_err()));
    }

    #[tokio::test]
    async fn non_streaming_with_tool_calls() {
        let dir = tempfile::TempDir::new().unwrap();
        let script = make_provider_script(
            dir.path(),
            "llm-provider-tools",
            r#"#!/bin/sh
read _
echo '{"text":"Let me search","tool_calls":[{"name":"search","arguments":{"query":"rust"},"tool_call_id":"tc_1"}],"usage":{"input":10,"output":20}}'
"#,
        );
        let provider = SubprocessProvider::new(
            script,
            "tools".into(),
            vec![ModelInfo {
                id: "tools-model".into(),
                can_stream: true,
                supports_tools: true,
                supports_schema: false,
                attachment_types: Vec::new(),
            }],
            KeyRequirement {
                needed: false,
                env_var: None,
            },
        );

        let stream = provider
            .execute("tools-model", &Prompt::new("search rust"), None, false)
            .await
            .unwrap();
        let chunks: Vec<Chunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let text = collect_text(&chunks);
        assert_eq!(text, "Let me search");
        let tool_calls = collect_tool_calls(&chunks);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "search");
        assert_eq!(tool_calls[0].tool_call_id.as_deref(), Some("tc_1"));
    }
}
