use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{LlmError, Result};

// ---------------------------------------------------------------------------
// AgentConfig
// ---------------------------------------------------------------------------

fn default_chain_limit() -> usize {
    10
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

    /// Sub-agent names (Tier 3 stub — parsed but not wired up).
    #[serde(default)]
    pub sub_agents: Vec<String>,

    /// Memory configuration (Tier 3 stub).
    #[serde(default)]
    pub memory: Option<MemoryConfig>,

    /// Budget configuration (max_tokens enforced by chain loop).
    #[serde(default)]
    pub budget: Option<BudgetConfig>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: None,
            system_prompt: None,
            tools: Vec::new(),
            chain_limit: default_chain_limit(),
            options: HashMap::new(),
            sub_agents: Vec::new(),
            memory: None,
            budget: None,
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

/// Memory configuration (Tier 3 stub — parsed but not wired up).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub last_n: Option<usize>,
}

/// Budget configuration. `max_tokens` is passed to `chain()` for enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    #[serde(default)]
    pub max_tokens: Option<u64>,
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
        assert!(config.sub_agents.is_empty());
        assert!(config.memory.is_none());
        assert!(config.budget.is_none());
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
sub_agents = ["security-checker"]

[options]
temperature = 0

[memory]
enabled = true
last_n = 10

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
        assert_eq!(config.sub_agents, vec!["security-checker"]);

        let mem = config.memory.unwrap();
        assert!(mem.enabled);
        assert_eq!(mem.last_n, Some(10));

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
    fn agent_config_load_with_stub_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("stubs.toml");
        std::fs::write(
            &path,
            r#"
sub_agents = ["helper-a", "helper-b"]

[memory]
enabled = false

[budget]
max_tokens = 100000
"#,
        )
        .unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.sub_agents, vec!["helper-a", "helper-b"]);
        let mem = config.memory.unwrap();
        assert!(!mem.enabled);
        assert!(mem.last_n.is_none());
        let budget = config.budget.unwrap();
        assert_eq!(budget.max_tokens, Some(100000));
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
