use std::path::PathBuf;

/// Scan PATH for executables matching the given prefix (e.g. "llm-tool-").
fn scan_path(prefix: &str) -> Vec<PathBuf> {
    let path_var = match std::env::var_os("PATH") {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();

    for dir in std::env::split_paths(&path_var) {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if !name.starts_with(prefix) {
                continue;
            }
            // Skip directories
            if entry.file_type().is_ok_and(|ft| ft.is_dir()) {
                continue;
            }
            // Skip non-executable on unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(meta) = entry.metadata()
                    && meta.permissions().mode() & 0o111 == 0
                {
                    continue;
                }
            }
            // Dedup: first occurrence in PATH wins
            let name_string = name.to_string();
            if seen.contains(&name_string) {
                continue;
            }
            seen.insert(name_string);
            results.push(entry.path());
        }
    }

    results
}

/// Discover external tool binaries (`llm-tool-*`) on PATH.
pub fn discover_tools() -> Vec<PathBuf> {
    scan_path("llm-tool-")
}

/// Discover external provider binaries (`llm-provider-*`) on PATH.
pub fn discover_providers() -> Vec<PathBuf> {
    scan_path("llm-provider-")
}

/// Fetch tool schema by running `binary --schema` and parsing the JSON output.
pub async fn fetch_tool_schema(
    binary: &std::path::Path,
    timeout: std::time::Duration,
) -> llm_core::Result<llm_core::Tool> {
    let result = tokio::time::timeout(timeout, async {
        let output = tokio::process::Command::new(binary)
            .arg("--schema")
            .output()
            .await
            .map_err(|e| llm_core::LlmError::Provider(format!("failed to run {}: {e}", binary.display())))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(llm_core::LlmError::Provider(format!(
                "{} --schema exited with {}: {}",
                binary.display(),
                output.status,
                stderr.trim()
            )));
        }

        let tool: llm_core::Tool = serde_json::from_slice(&output.stdout).map_err(|e| {
            llm_core::LlmError::Provider(format!(
                "invalid schema JSON from {}: {e}",
                binary.display()
            ))
        })?;
        Ok(tool)
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(llm_core::LlmError::Provider(format!(
            "{} --schema timed out",
            binary.display()
        ))),
    }
}

/// Fetch schemas from multiple tool binaries, skipping failures.
pub async fn fetch_all_tool_schemas(
    binaries: &[PathBuf],
    timeout: std::time::Duration,
) -> Vec<(PathBuf, llm_core::Tool)> {
    let mut results = Vec::new();
    for binary in binaries {
        match fetch_tool_schema(binary, timeout).await {
            Ok(tool) => results.push((binary.clone(), tool)),
            Err(e) => eprintln!("warning: skipping tool {}: {e}", binary.display()),
        }
    }
    results
}

/// Fetch provider ID by running `binary --id`.
pub async fn fetch_provider_id(
    binary: &std::path::Path,
    timeout: std::time::Duration,
) -> llm_core::Result<String> {
    let result = tokio::time::timeout(timeout, async {
        let output = tokio::process::Command::new(binary)
            .arg("--id")
            .output()
            .await
            .map_err(|e| llm_core::LlmError::Provider(format!("failed to run {}: {e}", binary.display())))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(llm_core::LlmError::Provider(format!(
                "{} --id failed: {}",
                binary.display(),
                stderr.trim()
            )));
        }

        let id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(id)
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(llm_core::LlmError::Provider(format!(
            "{} --id timed out",
            binary.display()
        ))),
    }
}

/// Fetch model list by running `binary --models`.
pub async fn fetch_provider_models(
    binary: &std::path::Path,
    timeout: std::time::Duration,
) -> llm_core::Result<Vec<llm_core::ModelInfo>> {
    let result = tokio::time::timeout(timeout, async {
        let output = tokio::process::Command::new(binary)
            .arg("--models")
            .output()
            .await
            .map_err(|e| llm_core::LlmError::Provider(format!("failed to run {}: {e}", binary.display())))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(llm_core::LlmError::Provider(format!(
                "{} --models failed: {}",
                binary.display(),
                stderr.trim()
            )));
        }

        let models: Vec<llm_core::ModelInfo> = serde_json::from_slice(&output.stdout).map_err(|e| {
            llm_core::LlmError::Provider(format!(
                "invalid models JSON from {}: {e}",
                binary.display()
            ))
        })?;
        Ok(models)
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(llm_core::LlmError::Provider(format!(
            "{} --models timed out",
            binary.display()
        ))),
    }
}

/// Key requirement from a subprocess provider.
#[derive(serde::Deserialize, Debug)]
pub struct KeyRequirement {
    pub needed: bool,
    pub env_var: Option<String>,
}

/// Fetch key requirement by running `binary --needs-key`.
pub async fn fetch_provider_key_info(
    binary: &std::path::Path,
    timeout: std::time::Duration,
) -> llm_core::Result<KeyRequirement> {
    let result = tokio::time::timeout(timeout, async {
        let output = tokio::process::Command::new(binary)
            .arg("--needs-key")
            .output()
            .await
            .map_err(|e| llm_core::LlmError::Provider(format!("failed to run {}: {e}", binary.display())))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(llm_core::LlmError::Provider(format!(
                "{} --needs-key failed: {}",
                binary.display(),
                stderr.trim()
            )));
        }

        let info: KeyRequirement = serde_json::from_slice(&output.stdout).map_err(|e| {
            llm_core::LlmError::Provider(format!(
                "invalid key info JSON from {}: {e}",
                binary.display()
            ))
        })?;
        Ok(info)
    })
    .await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(llm_core::LlmError::Provider(format!(
            "{} --needs-key timed out",
            binary.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_executable(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, "#!/bin/sh\necho test").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    fn make_non_executable(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, "not executable").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        path
    }

    #[test]
    fn discover_tools_finds_matching_binaries() {
        let dir = TempDir::new().unwrap();
        make_executable(dir.path(), "llm-tool-foo");
        make_executable(dir.path(), "llm-tool-bar");
        make_executable(dir.path(), "other-binary");

        temp_env::with_var("PATH", Some(dir.path().to_str().unwrap()), || {
            let tools = discover_tools();
            let names: Vec<String> = tools
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect();
            assert!(names.contains(&"llm-tool-foo".to_string()));
            assert!(names.contains(&"llm-tool-bar".to_string()));
            assert!(!names.contains(&"other-binary".to_string()));
        });
    }

    #[test]
    fn discover_tools_skips_non_executable() {
        let dir = TempDir::new().unwrap();
        make_non_executable(dir.path(), "llm-tool-noexec");
        make_executable(dir.path(), "llm-tool-exec");

        temp_env::with_var("PATH", Some(dir.path().to_str().unwrap()), || {
            let tools = discover_tools();
            let names: Vec<String> = tools
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect();
            assert!(names.contains(&"llm-tool-exec".to_string()));
            assert!(!names.contains(&"llm-tool-noexec".to_string()));
        });
    }

    #[test]
    fn discover_tools_skips_directories() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("llm-tool-dir")).unwrap();
        make_executable(dir.path(), "llm-tool-real");

        temp_env::with_var("PATH", Some(dir.path().to_str().unwrap()), || {
            let tools = discover_tools();
            let names: Vec<String> = tools
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect();
            assert!(names.contains(&"llm-tool-real".to_string()));
            assert!(!names.contains(&"llm-tool-dir".to_string()));
        });
    }

    #[test]
    fn discover_providers_finds_matching_binaries() {
        let dir = TempDir::new().unwrap();
        make_executable(dir.path(), "llm-provider-ollama");
        make_executable(dir.path(), "llm-tool-foo");

        temp_env::with_var("PATH", Some(dir.path().to_str().unwrap()), || {
            let providers = discover_providers();
            let names: Vec<String> = providers
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect();
            assert!(names.contains(&"llm-provider-ollama".to_string()));
            assert!(!names.contains(&"llm-tool-foo".to_string()));
        });
    }

    #[test]
    fn discover_deduplicates_across_path_dirs() {
        let dir1 = TempDir::new().unwrap();
        let dir2 = TempDir::new().unwrap();
        make_executable(dir1.path(), "llm-tool-dup");
        make_executable(dir2.path(), "llm-tool-dup");

        let path = format!(
            "{}:{}",
            dir1.path().display(),
            dir2.path().display()
        );
        temp_env::with_var("PATH", Some(&path), || {
            let tools = discover_tools();
            let names: Vec<String> = tools
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect();
            // Should appear exactly once (first wins)
            assert_eq!(names.iter().filter(|n| *n == "llm-tool-dup").count(), 1);
            // Should be from dir1
            assert!(tools[0].starts_with(dir1.path()));
        });
    }

    #[test]
    fn discover_handles_empty_path() {
        temp_env::with_var("PATH", Some(""), || {
            let tools = discover_tools();
            assert!(tools.is_empty());
        });
    }

    #[tokio::test]
    async fn fetch_tool_schema_parses_valid_output() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-tool-upper");
        std::fs::write(
            &script,
            r#"#!/bin/sh
echo '{"name":"upper","description":"Uppercase text","input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}'
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let tool = fetch_tool_schema(&script, std::time::Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(tool.name, "upper");
        assert_eq!(tool.description, "Uppercase text");
        assert_eq!(tool.input_schema["type"], "object");
    }

    #[tokio::test]
    async fn fetch_tool_schema_error_on_invalid_json() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-tool-bad");
        std::fs::write(&script, "#!/bin/sh\necho 'not json'").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let result = fetch_tool_schema(&script, std::time::Duration::from_secs(5)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn fetch_tool_schema_error_on_nonzero_exit() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-tool-fail");
        std::fs::write(&script, "#!/bin/sh\nexit 1").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let result = fetch_tool_schema(&script, std::time::Duration::from_secs(5)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn fetch_tool_schema_error_on_timeout() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-tool-slow");
        std::fs::write(&script, "#!/bin/sh\nsleep 10").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let result =
            fetch_tool_schema(&script, std::time::Duration::from_millis(100)).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timed out"), "got: {err}");
    }

    #[tokio::test]
    async fn fetch_provider_id_returns_trimmed_string() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-provider-test");
        std::fs::write(&script, "#!/bin/sh\necho '  ollama  '").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let id = fetch_provider_id(&script, std::time::Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(id, "ollama");
    }

    #[tokio::test]
    async fn fetch_provider_models_parses_json_array() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-provider-test");
        std::fs::write(
            &script,
            r#"#!/bin/sh
echo '[{"id":"llama3","can_stream":true,"supports_tools":false,"supports_schema":false,"attachment_types":[]}]'
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let models = fetch_provider_models(&script, std::time::Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "llama3");
        assert!(models[0].can_stream);
    }

    #[tokio::test]
    async fn fetch_provider_key_info_parses() {
        let dir = TempDir::new().unwrap();
        let script = dir.path().join("llm-provider-test");
        std::fs::write(
            &script,
            r#"#!/bin/sh
echo '{"needed":true,"env_var":"OLLAMA_KEY"}'
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let info = fetch_provider_key_info(&script, std::time::Duration::from_secs(5))
            .await
            .unwrap();
        assert!(info.needed);
        assert_eq!(info.env_var.as_deref(), Some("OLLAMA_KEY"));
    }
}
