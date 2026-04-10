use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{LlmError, Result};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Resolved filesystem paths for config and data directories.
#[derive(Debug, Clone)]
pub struct Paths {
    config_dir: PathBuf,
    data_dir: PathBuf,
}

impl Paths {
    /// Resolve paths from environment variables (pure XDG).
    ///
    /// Priority:
    /// 1. `LLM_USER_PATH` → flat layout (both dirs point there)
    /// 2. `$XDG_CONFIG_HOME/llm` / `$XDG_DATA_HOME/llm`
    /// 3. `$HOME/.config/llm` / `$HOME/.local/share/llm`
    pub fn resolve() -> Result<Self> {
        if let Ok(user_path) = std::env::var("LLM_USER_PATH") {
            return Ok(Self::from_dir(Path::new(&user_path)));
        }

        let home = std::env::var("HOME")
            .map_err(|_| LlmError::Config("$HOME is not set".into()))?;
        let home = PathBuf::from(home);

        let config_dir = match std::env::var("XDG_CONFIG_HOME") {
            Ok(val) if !val.is_empty() => PathBuf::from(val).join("llm"),
            _ => home.join(".config").join("llm"),
        };

        let data_dir = match std::env::var("XDG_DATA_HOME") {
            Ok(val) if !val.is_empty() => PathBuf::from(val).join("llm"),
            _ => home.join(".local").join("share").join("llm"),
        };

        Ok(Self { config_dir, data_dir })
    }

    /// Both dirs point to `dir`. Used for testing and `LLM_USER_PATH` override.
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            config_dir: dir.to_path_buf(),
            data_dir: dir.to_path_buf(),
        }
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    pub fn keys_file(&self) -> PathBuf {
        self.config_dir.join("keys.toml")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }

    pub fn agents_dir(&self) -> PathBuf {
        self.config_dir.join("agents")
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn default_model() -> String {
    "gpt-4o-mini".into()
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub default_model: String,
    #[serde(default = "default_true")]
    pub logging: bool,
    #[serde(default)]
    pub aliases: HashMap<String, String>,
    #[serde(default)]
    pub options: HashMap<String, HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub providers: HashMap<String, serde_json::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            logging: true,
            aliases: HashMap::new(),
            options: HashMap::new(),
            providers: HashMap::new(),
        }
    }
}

impl Config {
    /// Load config from a TOML file. Returns defaults if file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(contents) => {
                toml::from_str(&contents).map_err(|e| LlmError::Config(e.to_string()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(LlmError::Io(e)),
        }
    }

    /// Returns the effective default model, checking `LLM_DEFAULT_MODEL` env var first.
    pub fn default_model(&self) -> &str {
        // Can't return a reference to an env var, so this method checks at call time.
        // The caller should use `effective_default_model()` for the owned version.
        &self.default_model
    }

    /// Returns the effective default model (env var `LLM_DEFAULT_MODEL` takes priority).
    pub fn effective_default_model(&self) -> String {
        match std::env::var("LLM_DEFAULT_MODEL") {
            Ok(val) if !val.is_empty() => val,
            _ => self.default_model.clone(),
        }
    }

    /// Resolve a model name through aliases. Returns the alias target if found,
    /// otherwise returns the input unchanged.
    pub fn resolve_model<'a>(&'a self, input: &'a str) -> &'a str {
        self.aliases.get(input).map(|s| s.as_str()).unwrap_or(input)
    }

    /// Save config to a TOML file, creating parent directories if needed.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)
            .map_err(|e| LlmError::Config(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Returns a clone of all options for a given model (empty if none set).
    pub fn model_options(&self, model: &str) -> HashMap<String, serde_json::Value> {
        self.options.get(model).cloned().unwrap_or_default()
    }

    /// Set a single option for a model.
    pub fn set_option(&mut self, model: &str, key: &str, value: serde_json::Value) {
        self.options
            .entry(model.to_string())
            .or_default()
            .insert(key.to_string(), value);
    }

    /// Clear a single option for a model. Returns `true` if the key existed.
    /// Removes the model entry entirely if no options remain.
    pub fn clear_option(&mut self, model: &str, key: &str) -> bool {
        if let Some(model_opts) = self.options.get_mut(model) {
            let removed = model_opts.remove(key).is_some();
            if model_opts.is_empty() {
                self.options.remove(model);
            }
            removed
        } else {
            false
        }
    }

    /// Clear all options for a model. Returns `true` if the model had options.
    pub fn clear_model_options(&mut self, model: &str) -> bool {
        self.options.remove(model).is_some()
    }

    /// Set an alias mapping `alias` to `model`.
    pub fn set_alias(&mut self, alias: &str, model: &str) {
        self.aliases.insert(alias.to_string(), model.to_string());
    }

    /// Remove an alias. Returns `true` if the alias existed.
    pub fn remove_alias(&mut self, alias: &str) -> bool {
        self.aliases.remove(alias).is_some()
    }
}

// ---------------------------------------------------------------------------
// parse_option_value
// ---------------------------------------------------------------------------

/// Smart-coerce a string into a JSON value.
///
/// Tries, in order: integer, float, bool (`true`/`false`), `null`, fallback to string.
pub fn parse_option_value(s: &str) -> serde_json::Value {
    // Integer (no decimal point)
    if let Ok(n) = s.parse::<i64>() {
        return serde_json::Value::Number(n.into());
    }
    // Float
    if let Ok(f) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return serde_json::Value::Number(n);
        }
    }
    // Bool
    match s {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        "null" => return serde_json::Value::Null,
        _ => {}
    }
    // Fallback: string
    serde_json::Value::String(s.to_string())
}

// ---------------------------------------------------------------------------
// KeyStore
// ---------------------------------------------------------------------------

/// API key storage backed by a TOML file.
#[derive(Debug)]
pub struct KeyStore {
    keys: HashMap<String, String>,
    path: PathBuf,
}

impl KeyStore {
    /// Load keys from a TOML file. Returns an empty store if file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        let keys = match std::fs::read_to_string(path) {
            Ok(contents) => {
                toml::from_str::<HashMap<String, String>>(&contents)
                    .map_err(|e| LlmError::Config(format!("invalid keys.toml: {e}")))?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
            Err(e) => return Err(LlmError::Io(e)),
        };
        Ok(Self {
            keys,
            path: path.to_path_buf(),
        })
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.keys.get(name).map(|s| s.as_str())
    }

    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.keys.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Set a key, writing the updated store to disk.
    /// Creates parent directories and sets 0o600 permissions on Unix.
    pub fn set(&mut self, name: &str, value: &str) -> Result<()> {
        self.keys.insert(name.to_string(), value.to_string());

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string(&self.keys)
            .map_err(|e| LlmError::Config(format!("failed to serialize keys: {e}")))?;
        std::fs::write(&self.path, &contents)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&self.path, perms)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// resolve_key
// ---------------------------------------------------------------------------

/// Resolve an API key through the 4-level fallback chain.
///
/// 1. `explicit_key` (from `--key` CLI flag)
/// 2. `key_store.get(key_alias)` (from `keys.toml`)
/// 3. Environment variable (e.g. `OPENAI_API_KEY`)
/// 4. Error with actionable message
pub fn resolve_key(
    explicit_key: Option<&str>,
    key_store: &KeyStore,
    key_alias: &str,
    env_var: Option<&str>,
) -> Result<String> {
    if let Some(key) = explicit_key {
        return Ok(key.to_string());
    }

    if let Some(key) = key_store.get(key_alias) {
        return Ok(key.to_string());
    }

    if let Some(var_name) = env_var
        && let Ok(val) = std::env::var(var_name)
        && !val.is_empty()
    {
        return Ok(val);
    }

    let mut msg = format!("No key found - set one with 'llm keys set {key_alias}'");
    if let Some(var_name) = env_var {
        msg.push_str(&format!(" or set the {var_name} environment variable"));
    }
    Err(LlmError::NeedsKey(msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Cycle 1: Paths ---

    #[test]
    fn paths_from_dir() {
        let paths = Paths::from_dir(Path::new("/tmp/llm-test"));
        assert_eq!(paths.config_dir(), Path::new("/tmp/llm-test"));
        assert_eq!(paths.data_dir(), Path::new("/tmp/llm-test"));
    }

    #[test]
    fn paths_derived_methods() {
        let paths = Paths::from_dir(Path::new("/base"));
        assert_eq!(paths.config_file(), PathBuf::from("/base/config.toml"));
        assert_eq!(paths.keys_file(), PathBuf::from("/base/keys.toml"));
        assert_eq!(paths.logs_dir(), PathBuf::from("/base/logs"));
        assert_eq!(paths.agents_dir(), PathBuf::from("/base/agents"));
    }

    #[test]
    fn paths_agents_dir() {
        let paths = Paths {
            config_dir: PathBuf::from("/etc/llm"),
            data_dir: PathBuf::from("/var/llm"),
        };
        assert_eq!(paths.agents_dir(), PathBuf::from("/etc/llm/agents"));
    }

    #[test]
    fn paths_agents_dir_from_dir() {
        let paths = Paths::from_dir(Path::new("/tmp/llm-test"));
        assert_eq!(paths.agents_dir(), PathBuf::from("/tmp/llm-test/agents"));
    }

    #[test]
    fn paths_separate_dirs() {
        let paths = Paths {
            config_dir: PathBuf::from("/etc/llm"),
            data_dir: PathBuf::from("/var/llm"),
        };
        assert_eq!(paths.config_file(), PathBuf::from("/etc/llm/config.toml"));
        assert_eq!(paths.keys_file(), PathBuf::from("/etc/llm/keys.toml"));
        assert_eq!(paths.logs_dir(), PathBuf::from("/var/llm/logs"));
    }

    #[test]
    fn paths_resolve_xdg_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_str().unwrap();

        temp_env::with_vars(
            [
                ("HOME", Some(home)),
                ("LLM_USER_PATH", None::<&str>),
                ("XDG_CONFIG_HOME", None::<&str>),
                ("XDG_DATA_HOME", None::<&str>),
            ],
            || {
                let paths = Paths::resolve().unwrap();
                assert_eq!(paths.config_dir(), tmp.path().join(".config/llm"));
                assert_eq!(paths.data_dir(), tmp.path().join(".local/share/llm"));
            },
        );
    }

    #[test]
    fn paths_resolve_xdg_custom() {
        let tmp = tempfile::tempdir().unwrap();
        let xdg_config = tmp.path().join("myconfig");
        let xdg_data = tmp.path().join("mydata");

        temp_env::with_vars(
            [
                ("HOME", Some(tmp.path().to_str().unwrap())),
                ("LLM_USER_PATH", None::<&str>),
                ("XDG_CONFIG_HOME", Some(xdg_config.to_str().unwrap())),
                ("XDG_DATA_HOME", Some(xdg_data.to_str().unwrap())),
            ],
            || {
                let paths = Paths::resolve().unwrap();
                assert_eq!(paths.config_dir(), xdg_config.join("llm"));
                assert_eq!(paths.data_dir(), xdg_data.join("llm"));
            },
        );
    }

    // --- Cycle 2: Config ---

    #[test]
    fn config_default() {
        let config = Config::default();
        assert_eq!(config.default_model, "gpt-4o-mini");
        assert!(config.logging);
        assert!(config.aliases.is_empty());
        assert!(config.options.is_empty());
        assert!(config.providers.is_empty());
    }

    #[test]
    fn config_load_missing_file() {
        let config = Config::load(Path::new("/nonexistent/config.toml")).unwrap();
        assert_eq!(config.default_model, "gpt-4o-mini");
        assert!(config.logging);
    }

    #[test]
    fn config_load_valid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
default_model = "claude-sonnet-4-20250514"
logging = false

[aliases]
claude = "claude-sonnet-4-20250514"
fast = "gpt-4o-mini"

[options.gpt-4o]
temperature = 0.7
"#,
        )
        .unwrap();

        let config = Config::load(&path).unwrap();
        assert_eq!(config.default_model, "claude-sonnet-4-20250514");
        assert!(!config.logging);
        assert_eq!(config.aliases.len(), 2);
        assert_eq!(config.aliases["claude"], "claude-sonnet-4-20250514");
        assert_eq!(config.options["gpt-4o"]["temperature"], 0.7);
    }

    #[test]
    fn config_load_partial_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "logging = false\n").unwrap();

        let config = Config::load(&path).unwrap();
        assert_eq!(config.default_model, "gpt-4o-mini"); // default
        assert!(!config.logging); // overridden
        assert!(config.aliases.is_empty()); // default
    }

    #[test]
    fn config_load_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "").unwrap();

        let config = Config::load(&path).unwrap();
        assert_eq!(config.default_model, "gpt-4o-mini");
        assert!(config.logging);
    }

    #[test]
    fn config_load_invalid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "not valid {{{{ toml").unwrap();

        let result = Config::load(&path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::Config(_)));
    }

    #[test]
    fn config_resolve_model_alias() {
        let mut config = Config::default();
        config
            .aliases
            .insert("claude".into(), "claude-sonnet-4-20250514".into());

        assert_eq!(config.resolve_model("claude"), "claude-sonnet-4-20250514");
    }

    #[test]
    fn config_resolve_model_passthrough() {
        let config = Config::default();
        assert_eq!(config.resolve_model("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn config_effective_default_model_env_override() {
        let config = Config::default();
        temp_env::with_vars(
            [("LLM_DEFAULT_MODEL", Some("o3"))],
            || {
                assert_eq!(config.effective_default_model(), "o3");
            },
        );
    }

    #[test]
    fn config_effective_default_model_fallback() {
        let config = Config::default();
        temp_env::with_vars(
            [("LLM_DEFAULT_MODEL", None::<&str>)],
            || {
                assert_eq!(config.effective_default_model(), "gpt-4o-mini");
            },
        );
    }

    #[test]
    fn paths_resolve_llm_user_path() {
        temp_env::with_vars(
            [
                ("LLM_USER_PATH", Some("/custom/llm")),
                ("HOME", Some("/should-not-matter")),
            ],
            || {
                let paths = Paths::resolve().unwrap();
                assert_eq!(paths.config_dir(), Path::new("/custom/llm"));
                assert_eq!(paths.data_dir(), Path::new("/custom/llm"));
            },
        );
    }

    // --- Cycle 3: KeyStore read ---

    #[test]
    fn keystore_load_missing_file() {
        let store = KeyStore::load(Path::new("/nonexistent/keys.toml")).unwrap();
        assert!(store.list().is_empty());
    }

    #[test]
    fn keystore_load_valid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-abc\"\nanthropic = \"sk-ant-xyz\"\n").unwrap();

        let store = KeyStore::load(&path).unwrap();
        assert_eq!(store.get("openai"), Some("sk-abc"));
        assert_eq!(store.get("anthropic"), Some("sk-ant-xyz"));
    }

    #[test]
    fn keystore_load_invalid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "not {{ valid").unwrap();

        let result = KeyStore::load(&path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::Config(_)));
    }

    #[test]
    fn keystore_get_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-test123\"\n").unwrap();

        let store = KeyStore::load(&path).unwrap();
        assert_eq!(store.get("openai"), Some("sk-test123"));
    }

    #[test]
    fn keystore_get_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-test\"\n").unwrap();

        let store = KeyStore::load(&path).unwrap();
        assert_eq!(store.get("anthropic"), None);
    }

    #[test]
    fn keystore_list() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-1\"\nanthropic = \"sk-2\"\nollama = \"\"\n").unwrap();

        let store = KeyStore::load(&path).unwrap();
        assert_eq!(store.list(), vec!["anthropic", "ollama", "openai"]); // sorted
    }

    #[test]
    fn keystore_path() {
        let store = KeyStore::load(Path::new("/some/keys.toml")).unwrap();
        assert_eq!(store.path(), Path::new("/some/keys.toml"));
    }

    // --- Cycle 4: KeyStore write ---

    #[test]
    fn keystore_set_new_key() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");

        let mut store = KeyStore::load(&path).unwrap();
        store.set("openai", "sk-new").unwrap();

        // Verify by re-loading
        let store2 = KeyStore::load(&path).unwrap();
        assert_eq!(store2.get("openai"), Some("sk-new"));
    }

    #[test]
    fn keystore_set_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-old\"\nanthropic = \"sk-ant\"\n").unwrap();

        let mut store = KeyStore::load(&path).unwrap();
        store.set("openai", "sk-new").unwrap();

        let store2 = KeyStore::load(&path).unwrap();
        assert_eq!(store2.get("openai"), Some("sk-new"));
        assert_eq!(store2.get("anthropic"), Some("sk-ant")); // preserved
    }

    #[test]
    fn keystore_set_creates_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sub").join("dir").join("keys.toml");

        let mut store = KeyStore::load(&path).unwrap();
        store.set("openai", "sk-test").unwrap();
        assert!(path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn keystore_set_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");

        let mut store = KeyStore::load(&path).unwrap();
        store.set("openai", "sk-secret").unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    // --- Cycle 5: resolve_key ---

    #[test]
    fn resolve_key_explicit() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-stored\"\n").unwrap();
        let store = KeyStore::load(&path).unwrap();

        let key = resolve_key(Some("sk-explicit"), &store, "openai", Some("OPENAI_API_KEY")).unwrap();
        assert_eq!(key, "sk-explicit");
    }

    #[test]
    fn resolve_key_from_store() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        std::fs::write(&path, "openai = \"sk-stored\"\n").unwrap();
        let store = KeyStore::load(&path).unwrap();

        temp_env::with_vars(
            [("OPENAI_API_KEY", None::<&str>)],
            || {
                let key = resolve_key(None, &store, "openai", Some("OPENAI_API_KEY")).unwrap();
                assert_eq!(key, "sk-stored");
            },
        );
    }

    #[test]
    fn resolve_key_from_env() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        // empty store
        let store = KeyStore::load(&path).unwrap();

        temp_env::with_vars(
            [("OPENAI_API_KEY", Some("sk-from-env"))],
            || {
                let key = resolve_key(None, &store, "openai", Some("OPENAI_API_KEY")).unwrap();
                assert_eq!(key, "sk-from-env");
            },
        );
    }

    #[test]
    fn resolve_key_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        let store = KeyStore::load(&path).unwrap();

        temp_env::with_vars(
            [("OPENAI_API_KEY", None::<&str>)],
            || {
                let err = resolve_key(None, &store, "openai", Some("OPENAI_API_KEY")).unwrap_err();
                let msg = err.to_string();
                assert!(msg.contains("llm keys set openai"), "msg: {msg}");
                assert!(msg.contains("OPENAI_API_KEY"), "msg: {msg}");
            },
        );
    }

    #[test]
    fn resolve_key_env_empty_string_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        let store = KeyStore::load(&path).unwrap();

        temp_env::with_vars(
            [("OPENAI_API_KEY", Some(""))],
            || {
                let result = resolve_key(None, &store, "openai", Some("OPENAI_API_KEY"));
                assert!(result.is_err());
            },
        );
    }

    #[test]
    fn resolve_key_no_env_var() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("keys.toml");
        let store = KeyStore::load(&path).unwrap();

        let err = resolve_key(None, &store, "openai", None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("llm keys set openai"), "msg: {msg}");
        assert!(!msg.contains("environment variable"), "msg: {msg}");
    }

    // --- parse_option_value ---

    #[test]
    fn parse_option_value_int() {
        assert_eq!(parse_option_value("42"), serde_json::json!(42));
        assert_eq!(parse_option_value("-1"), serde_json::json!(-1));
        assert_eq!(parse_option_value("0"), serde_json::json!(0));
    }

    #[test]
    fn parse_option_value_float() {
        assert_eq!(parse_option_value("0.7"), serde_json::json!(0.7));
        assert_eq!(parse_option_value("1.5"), serde_json::json!(1.5));
    }

    #[test]
    fn parse_option_value_bool() {
        assert_eq!(parse_option_value("true"), serde_json::json!(true));
        assert_eq!(parse_option_value("false"), serde_json::json!(false));
    }

    #[test]
    fn parse_option_value_null() {
        assert_eq!(parse_option_value("null"), serde_json::Value::Null);
    }

    #[test]
    fn parse_option_value_string_fallback() {
        assert_eq!(parse_option_value("hello"), serde_json::json!("hello"));
        assert_eq!(parse_option_value("gpt-4o"), serde_json::json!("gpt-4o"));
        // "True" (capitalized) is not bool
        assert_eq!(parse_option_value("True"), serde_json::json!("True"));
    }

    #[test]
    fn parse_option_value_edge_cases() {
        // Large integer
        assert_eq!(parse_option_value("4096"), serde_json::json!(4096));
        // Negative float
        assert_eq!(parse_option_value("-0.5"), serde_json::json!(-0.5));
        // Empty string
        assert_eq!(parse_option_value(""), serde_json::json!(""));
    }

    // --- Config model_options / set_option / clear ---

    #[test]
    fn config_model_options_empty() {
        let config = Config::default();
        assert!(config.model_options("gpt-4o").is_empty());
    }

    #[test]
    fn config_set_and_get_option() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.7));
        config.set_option("gpt-4o", "max_tokens", serde_json::json!(200));

        let opts = config.model_options("gpt-4o");
        assert_eq!(opts.len(), 2);
        assert_eq!(opts["temperature"], serde_json::json!(0.7));
        assert_eq!(opts["max_tokens"], serde_json::json!(200));
    }

    #[test]
    fn config_set_option_overwrite() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.5));
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.9));
        assert_eq!(config.model_options("gpt-4o")["temperature"], serde_json::json!(0.9));
    }

    #[test]
    fn config_clear_option_single() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.7));
        config.set_option("gpt-4o", "max_tokens", serde_json::json!(200));

        assert!(config.clear_option("gpt-4o", "temperature"));
        let opts = config.model_options("gpt-4o");
        assert_eq!(opts.len(), 1);
        assert!(!opts.contains_key("temperature"));
    }

    #[test]
    fn config_clear_option_removes_empty_model() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.7));

        assert!(config.clear_option("gpt-4o", "temperature"));
        assert!(!config.options.contains_key("gpt-4o"));
    }

    #[test]
    fn config_clear_option_missing() {
        let mut config = Config::default();
        assert!(!config.clear_option("gpt-4o", "temperature"));
    }

    #[test]
    fn config_clear_model_options() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.7));
        config.set_option("gpt-4o", "max_tokens", serde_json::json!(200));

        assert!(config.clear_model_options("gpt-4o"));
        assert!(config.model_options("gpt-4o").is_empty());
        assert!(!config.options.contains_key("gpt-4o"));
    }

    #[test]
    fn config_clear_model_options_missing() {
        let mut config = Config::default();
        assert!(!config.clear_model_options("gpt-4o"));
    }

    #[test]
    fn config_options_save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");

        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.7));
        config.set_option("gpt-4o", "max_tokens", serde_json::json!(200));
        config.save(&path).unwrap();

        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.model_options("gpt-4o")["temperature"], serde_json::json!(0.7));
        assert_eq!(loaded.model_options("gpt-4o")["max_tokens"], serde_json::json!(200));
    }

    // --- Aliases: set_alias / remove_alias ---

    #[test]
    fn config_set_alias() {
        let mut config = Config::default();
        config.set_alias("claude", "claude-sonnet-4-20250514");
        assert_eq!(config.aliases["claude"], "claude-sonnet-4-20250514");
    }

    #[test]
    fn config_set_alias_overwrite() {
        let mut config = Config::default();
        config.set_alias("claude", "claude-sonnet-4-20250514");
        config.set_alias("claude", "claude-opus-4-20250514");
        assert_eq!(config.aliases["claude"], "claude-opus-4-20250514");
    }

    #[test]
    fn config_remove_alias() {
        let mut config = Config::default();
        config.set_alias("claude", "claude-sonnet-4-20250514");
        assert!(config.remove_alias("claude"));
        assert!(!config.aliases.contains_key("claude"));
    }

    #[test]
    fn config_remove_alias_missing() {
        let mut config = Config::default();
        assert!(!config.remove_alias("nonexistent"));
    }

    #[test]
    fn config_alias_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");

        let mut config = Config::default();
        config.set_alias("claude", "claude-sonnet-4-20250514");
        config.set_alias("fast", "gpt-4o-mini");
        config.save(&path).unwrap();

        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.aliases["claude"], "claude-sonnet-4-20250514");
        assert_eq!(loaded.aliases["fast"], "gpt-4o-mini");
    }
}
