use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{LlmError, Result};
use crate::retry::RetryConfig;
use crate::types::Tool;

// ---------------------------------------------------------------------------
// AgentConfig
// ---------------------------------------------------------------------------

fn default_chain_limit() -> usize {
    10
}

fn default_parallel_tools() -> bool {
    true
}

/// Configuration for an agent, loaded from a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Model to use (falls back to global default_model if None).
    #[serde(default)]
    pub model: Option<String>,

    /// System prompt for the agent.
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Tool names the agent should use.
    #[serde(default)]
    pub tools: Vec<String>,

    /// Maximum chain loop iterations (default: 10).
    #[serde(default = "default_chain_limit")]
    pub chain_limit: usize,

    /// Model options (temperature, max_tokens, etc.).
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,

    /// Budget configuration (max_tokens enforced by chain loop).
    #[serde(default)]
    pub budget: Option<BudgetConfig>,

    /// Retry configuration for transient HTTP errors.
    #[serde(default)]
    pub retry: Option<RetryConfig>,

    /// Dispatch tool calls in parallel within a single chain iteration.
    /// Default: true. Set to false for tools with ordering side-effects.
    #[serde(default = "default_parallel_tools")]
    pub parallel_tools: bool,

    /// Optional cap on parallel tool dispatch. `None` = unlimited.
    #[serde(default)]
    pub max_parallel_tools: Option<usize>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: None,
            system_prompt: None,
            tools: Vec::new(),
            chain_limit: default_chain_limit(),
            options: HashMap::new(),
            budget: None,
            retry: None,
            parallel_tools: default_parallel_tools(),
            max_parallel_tools: None,
        }
    }
}

impl AgentConfig {
    /// Load an agent config from a TOML file.
    /// Unlike `Config::load()`, this returns an error if the file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                LlmError::Config(format!("agent config not found: {}", path.display()))
            } else {
                LlmError::Io(e)
            }
        })?;
        toml::from_str(&contents).map_err(|e| LlmError::Config(e.to_string()))
    }
}

/// Budget configuration. `max_tokens` is passed to `chain()` for enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Programmatic resolution helpers
// ---------------------------------------------------------------------------
//
// Shared by `llm-python::run_agent` and `llm-wasm::runAgent`. Pure functions
// so they can be unit-tested without any binding machinery.

/// Resolve the model for an agent run: `config.model` if set, else the
/// client's default.
pub fn resolve_agent_model<'a>(config: &'a AgentConfig, client_default: &'a str) -> &'a str {
    config.model.as_deref().unwrap_or(client_default)
}

/// Resolve the system prompt: arg > `config.system_prompt` > None.
/// Mirrors the CLI precedence at `llm-cli/src/commands/agent.rs`.
pub fn resolve_agent_system<'a>(
    arg: Option<&'a str>,
    config: &'a AgentConfig,
) -> Option<&'a str> {
    arg.or(config.system_prompt.as_deref())
}

/// Resolve the retry config: CLI arg > agent TOML > client default.
pub fn resolve_agent_retry(
    cli_arg: Option<u32>,
    config: &AgentConfig,
    client_default: &RetryConfig,
) -> RetryConfig {
    if let Some(n) = cli_arg {
        let mut cfg = client_default.clone();
        cfg.max_retries = n;
        return cfg;
    }
    if let Some(agent_retry) = &config.retry {
        return agent_retry.clone();
    }
    client_default.clone()
}

/// Filter an agent's tool whitelist against the set of tools registered on
/// the host. Returns a `Vec<Tool>` in the order the agent requested them.
/// Errors with the exact CLI message if any name is unknown.
pub fn resolve_agent_tools(
    config: &AgentConfig,
    registry_tools: &[Tool],
) -> Result<Vec<Tool>> {
    let mut out = Vec::with_capacity(config.tools.len());
    for name in &config.tools {
        match registry_tools.iter().find(|t| t.name == *name) {
            Some(t) => out.push(t.clone()),
            None => {
                return Err(LlmError::Config(format!(
                    "unknown tool in agent config: {name}"
                )));
            }
        }
    }
    Ok(out)
}

/// Extract the budget (`max_tokens`) from an agent config, if set.
pub fn resolve_agent_budget(config: &AgentConfig) -> Option<u64> {
    config.budget.as_ref().and_then(|b| b.max_tokens)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Where an agent was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSource {
    Global,
    Local,
}

impl std::fmt::Display for AgentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentSource::Global => write!(f, "global"),
            AgentSource::Local => write!(f, "local"),
        }
    }
}

/// Metadata about a discovered agent.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub path: PathBuf,
    pub source: AgentSource,
}

/// Discover agents from global and optional local directories.
///
/// Local agents shadow global ones with the same name.
/// Results are sorted alphabetically by name.
pub fn discover_agents(
    global_dir: &Path,
    local_dir: Option<&Path>,
) -> Result<Vec<AgentInfo>> {
    let mut agents: HashMap<String, AgentInfo> = HashMap::new();

    // Scan global directory first
    scan_agents_dir(global_dir, AgentSource::Global, &mut agents)?;

    // Local shadows global
    if let Some(local) = local_dir {
        scan_agents_dir(local, AgentSource::Local, &mut agents)?;
    }

    let mut result: Vec<AgentInfo> = agents.into_values().collect();
    result.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

/// Resolve an agent by name: find it, load its config, return both.
pub fn resolve_agent(
    name: &str,
    global_dir: &Path,
    local_dir: Option<&Path>,
) -> Result<(AgentConfig, PathBuf)> {
    // Check local first (local shadows global)
    if let Some(local) = local_dir {
        let path = local.join(format!("{name}.toml"));
        if path.exists() {
            let config = AgentConfig::load(&path)?;
            return Ok((config, path));
        }
    }

    // Then global
    let path = global_dir.join(format!("{name}.toml"));
    if path.exists() {
        let config = AgentConfig::load(&path)?;
        return Ok((config, path));
    }

    Err(LlmError::Config(format!("agent not found: {name}")))
}

fn scan_agents_dir(
    dir: &Path,
    source: AgentSource,
    agents: &mut HashMap<String, AgentInfo>,
) -> Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(LlmError::Io(e)),
    };

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            agents.insert(
                stem.to_string(),
                AgentInfo {
                    name: stem.to_string(),
                    path,
                    source: source.clone(),
                },
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Cycle 1: AgentConfig type + TOML parsing ---

    #[test]
    fn agent_config_default() {
        let config = AgentConfig::default();
        assert!(config.model.is_none());
        assert!(config.system_prompt.is_none());
        assert!(config.tools.is_empty());
        assert_eq!(config.chain_limit, 10);
        assert!(config.options.is_empty());
        assert!(config.budget.is_none());
        assert!(config.retry.is_none());
        assert!(config.parallel_tools);
        assert!(config.max_parallel_tools.is_none());
    }

    #[test]
    fn agent_config_load_full_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("reviewer.toml");
        std::fs::write(
            &path,
            r#"
model = "claude-sonnet-4-20250514"
system_prompt = "You are a code reviewer."
tools = ["ripgrep", "read_file", "llm_time"]
chain_limit = 20

[options]
temperature = 0

[budget]
max_tokens = 50000
"#,
        )
        .unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.model.as_deref(), Some("claude-sonnet-4-20250514"));
        assert_eq!(config.system_prompt.as_deref(), Some("You are a code reviewer."));
        assert_eq!(config.tools, vec!["ripgrep", "read_file", "llm_time"]);
        assert_eq!(config.chain_limit, 20);
        assert_eq!(config.options["temperature"], serde_json::json!(0));

        let budget = config.budget.unwrap();
        assert_eq!(budget.max_tokens, Some(50000));
    }

    #[test]
    fn agent_config_load_minimal_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("minimal.toml");
        std::fs::write(&path, "model = \"gpt-4o-mini\"\n").unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.model.as_deref(), Some("gpt-4o-mini"));
        assert!(config.system_prompt.is_none());
        assert!(config.tools.is_empty());
        assert_eq!(config.chain_limit, 10); // default
        assert!(config.options.is_empty());
    }

    #[test]
    fn agent_config_load_with_options() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("opts.toml");
        std::fs::write(
            &path,
            r#"
[options]
temperature = 0.7
max_tokens = 200
"#,
        )
        .unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.options["temperature"], serde_json::json!(0.7));
        assert_eq!(config.options["max_tokens"], serde_json::json!(200));
    }

    #[test]
    fn agent_config_load_missing_file() {
        let result = AgentConfig::load(Path::new("/nonexistent/agent.toml"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LlmError::Config(_)));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn agent_config_load_invalid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.toml");
        std::fs::write(&path, "not valid {{{{ toml").unwrap();

        let result = AgentConfig::load(&path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::Config(_)));
    }

    #[test]
    fn agent_config_chain_limit_default() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.toml");
        std::fs::write(&path, "").unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.chain_limit, 10);
    }

    // --- Cycle 3: Discovery ---

    #[test]
    fn discover_agents_empty_dirs() {
        let global = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();

        let agents = discover_agents(global.path(), Some(local.path())).unwrap();
        assert!(agents.is_empty());
    }

    #[test]
    fn discover_agents_global_only() {
        let global = tempfile::tempdir().unwrap();
        std::fs::write(
            global.path().join("reviewer.toml"),
            "model = \"gpt-4o\"\n",
        )
        .unwrap();

        let agents = discover_agents(global.path(), None).unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "reviewer");
        assert_eq!(agents[0].source, AgentSource::Global);
    }

    #[test]
    fn discover_agents_local_only() {
        let global = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        std::fs::write(
            local.path().join("helper.toml"),
            "model = \"gpt-4o-mini\"\n",
        )
        .unwrap();

        let agents = discover_agents(global.path(), Some(local.path())).unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "helper");
        assert_eq!(agents[0].source, AgentSource::Local);
    }

    #[test]
    fn discover_agents_local_shadows_global() {
        let global = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        std::fs::write(
            global.path().join("reviewer.toml"),
            "model = \"gpt-4o\"\n",
        )
        .unwrap();
        std::fs::write(
            local.path().join("reviewer.toml"),
            "model = \"gpt-4o-mini\"\n",
        )
        .unwrap();

        let agents = discover_agents(global.path(), Some(local.path())).unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "reviewer");
        assert_eq!(agents[0].source, AgentSource::Local);
    }

    #[test]
    fn discover_agents_sorted() {
        let global = tempfile::tempdir().unwrap();
        std::fs::write(global.path().join("zebra.toml"), "").unwrap();
        std::fs::write(global.path().join("alpha.toml"), "").unwrap();
        std::fs::write(global.path().join("mid.toml"), "").unwrap();

        let agents = discover_agents(global.path(), None).unwrap();
        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mid", "zebra"]);
    }

    #[test]
    fn discover_agents_non_toml_ignored() {
        let global = tempfile::tempdir().unwrap();
        std::fs::write(global.path().join("agent.toml"), "").unwrap();
        std::fs::write(global.path().join("readme.md"), "# agents").unwrap();
        std::fs::write(global.path().join("notes.txt"), "some notes").unwrap();

        let agents = discover_agents(global.path(), None).unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "agent");
    }

    #[test]
    fn discover_agents_nonexistent_dirs() {
        let agents = discover_agents(
            Path::new("/nonexistent/global"),
            Some(Path::new("/nonexistent/local")),
        )
        .unwrap();
        assert!(agents.is_empty());
    }

    #[test]
    fn resolve_agent_found() {
        let global = tempfile::tempdir().unwrap();
        std::fs::write(
            global.path().join("reviewer.toml"),
            "model = \"gpt-4o\"\nsystem_prompt = \"Review code.\"\n",
        )
        .unwrap();

        let (config, path) = resolve_agent("reviewer", global.path(), None).unwrap();
        assert_eq!(config.model.as_deref(), Some("gpt-4o"));
        assert_eq!(config.system_prompt.as_deref(), Some("Review code."));
        assert_eq!(path, global.path().join("reviewer.toml"));
    }

    #[test]
    fn resolve_agent_not_found() {
        let global = tempfile::tempdir().unwrap();
        let result = resolve_agent("nonexistent", global.path(), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LlmError::Config(_)));
        assert!(err.to_string().contains("agent not found"));
    }

    // --- Retry config tests ---

    #[test]
    fn agent_config_parses_retry() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("retry.toml");
        std::fs::write(
            &path,
            r#"
[retry]
max_retries = 5
base_delay_ms = 500
"#,
        )
        .unwrap();

        let config = AgentConfig::load(&path).unwrap();
        let retry = config.retry.unwrap();
        assert_eq!(retry.max_retries, 5);
        assert_eq!(retry.base_delay_ms, 500);
        // Defaults for unspecified fields
        assert_eq!(retry.max_delay_ms, 30_000);
        assert!(retry.jitter);
    }

    #[test]
    fn agent_config_retry_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("retry_defaults.toml");
        std::fs::write(&path, "[retry]\n").unwrap();

        let config = AgentConfig::load(&path).unwrap();
        let retry = config.retry.unwrap();
        assert_eq!(retry.max_retries, 3);
        assert_eq!(retry.base_delay_ms, 1000);
        assert_eq!(retry.max_delay_ms, 30_000);
        assert!(retry.jitter);
    }

    // --- Parallel tool dispatch config tests ---

    #[test]
    fn agent_config_parses_parallel_tools_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("parallel.toml");
        std::fs::write(
            &path,
            r#"
parallel_tools = false
max_parallel_tools = 3
"#,
        )
        .unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert!(!config.parallel_tools);
        assert_eq!(config.max_parallel_tools, Some(3));
    }

    #[test]
    fn agent_config_parallel_tools_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("defaults.toml");
        std::fs::write(&path, "model = \"gpt-4o-mini\"\n").unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert!(config.parallel_tools, "parallel_tools should default to true");
        assert_eq!(config.max_parallel_tools, None);
    }

    #[test]
    fn agent_config_no_retry() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("no_retry.toml");
        std::fs::write(&path, "model = \"gpt-4o-mini\"\n").unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert!(config.retry.is_none());
    }

    // --- Programmatic resolution helpers (shared by Python/WASM bindings) ---

    fn tool(name: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: format!("{name} tool"),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn resolve_model_uses_config_when_set() {
        let mut cfg = AgentConfig::default();
        cfg.model = Some("gpt-4o".into());
        assert_eq!(resolve_agent_model(&cfg, "gpt-4o-mini"), "gpt-4o");
    }

    #[test]
    fn resolve_model_falls_back_to_client_default() {
        let cfg = AgentConfig::default();
        assert_eq!(resolve_agent_model(&cfg, "gpt-4o-mini"), "gpt-4o-mini");
    }

    #[test]
    fn resolve_system_prefers_arg_over_config() {
        let mut cfg = AgentConfig::default();
        cfg.system_prompt = Some("from config".into());
        assert_eq!(
            resolve_agent_system(Some("from arg"), &cfg),
            Some("from arg")
        );
    }

    #[test]
    fn resolve_system_uses_config_when_no_arg() {
        let mut cfg = AgentConfig::default();
        cfg.system_prompt = Some("from config".into());
        assert_eq!(resolve_agent_system(None, &cfg), Some("from config"));
    }

    #[test]
    fn resolve_system_none_when_neither() {
        let cfg = AgentConfig::default();
        assert_eq!(resolve_agent_system(None, &cfg), None);
    }

    #[test]
    fn resolve_retry_cli_arg_wins() {
        let cfg = AgentConfig::default();
        let client = RetryConfig::default();
        let out = resolve_agent_retry(Some(7), &cfg, &client);
        assert_eq!(out.max_retries, 7);
    }

    #[test]
    fn resolve_retry_agent_config_wins_over_default() {
        let mut cfg = AgentConfig::default();
        cfg.retry = Some(RetryConfig {
            max_retries: 5,
            base_delay_ms: 123,
            max_delay_ms: 456,
            jitter: false,
        });
        let client = RetryConfig::default();
        let out = resolve_agent_retry(None, &cfg, &client);
        assert_eq!(out.max_retries, 5);
        assert_eq!(out.base_delay_ms, 123);
    }

    #[test]
    fn resolve_retry_falls_back_to_client() {
        let cfg = AgentConfig::default();
        let client = RetryConfig {
            max_retries: 2,
            base_delay_ms: 1000,
            max_delay_ms: 30_000,
            jitter: true,
        };
        let out = resolve_agent_retry(None, &cfg, &client);
        assert_eq!(out.max_retries, 2);
    }

    #[test]
    fn resolve_tools_filters_to_agent_whitelist() {
        let mut cfg = AgentConfig::default();
        cfg.tools = vec!["read_file".into(), "llm_time".into()];
        let registry = vec![tool("read_file"), tool("ripgrep"), tool("llm_time")];

        let out = resolve_agent_tools(&cfg, &registry).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[1].name, "llm_time");
    }

    #[test]
    fn resolve_tools_errors_on_unknown_with_cli_format() {
        let mut cfg = AgentConfig::default();
        cfg.tools = vec!["missing".into()];
        let registry = vec![tool("read_file")];

        let err = resolve_agent_tools(&cfg, &registry).unwrap_err();
        let msg = err.to_string();
        // Byte-identical to the CLI error at
        // llm-cli/src/commands/agent.rs:331-333.
        assert!(
            msg.contains("unknown tool in agent config: missing"),
            "got: {msg}"
        );
    }

    #[test]
    fn resolve_tools_empty_config_returns_empty() {
        let cfg = AgentConfig::default();
        let registry = vec![tool("read_file")];
        let out = resolve_agent_tools(&cfg, &registry).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn resolve_budget_extracts_max_tokens() {
        let mut cfg = AgentConfig::default();
        cfg.budget = Some(BudgetConfig {
            max_tokens: Some(5000),
        });
        assert_eq!(resolve_agent_budget(&cfg), Some(5000));
    }

    #[test]
    fn resolve_budget_none_when_unset() {
        let cfg = AgentConfig::default();
        assert_eq!(resolve_agent_budget(&cfg), None);
    }

    #[test]
    fn resolve_agent_local_wins() {
        let global = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        std::fs::write(
            global.path().join("reviewer.toml"),
            "model = \"gpt-4o\"\n",
        )
        .unwrap();
        std::fs::write(
            local.path().join("reviewer.toml"),
            "model = \"claude-sonnet-4-20250514\"\n",
        )
        .unwrap();

        let (config, path) = resolve_agent("reviewer", global.path(), Some(local.path())).unwrap();
        assert_eq!(config.model.as_deref(), Some("claude-sonnet-4-20250514"));
        assert_eq!(path, local.path().join("reviewer.toml"));
    }
}
