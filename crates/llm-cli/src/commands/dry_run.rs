use std::collections::BTreeMap;
use std::fmt::Write as _;

use llm_core::{ParallelConfig, RetryConfig};
use serde::Serialize;

/// Source of the resolved model identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    Cli,
    Agent,
    Default,
}

impl ModelSource {
    fn as_str(self) -> &'static str {
        match self {
            ModelSource::Cli => "cli",
            ModelSource::Agent => "agent",
            ModelSource::Default => "default",
        }
    }
}

/// Origin of a resolved tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolSource {
    Builtin,
    External,
}

impl ToolSource {
    fn as_str(self) -> &'static str {
        match self {
            ToolSource::Builtin => "builtin",
            ToolSource::External => "external",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolEntry {
    pub name: String,
    pub source: ToolSource,
}

/// Snapshot of everything the agent run would send to the provider.
#[derive(Debug, Clone, Serialize)]
pub struct DryRunReport {
    pub agent_name: String,
    pub agent_path: String,
    pub model: String,
    pub model_source: ModelSource,
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub prompt_text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolEntry>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, serde_json::Value>,
    pub chain_limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,
    pub parallel: ParallelConfig,
    pub logging_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<serde_json::Value>,
}

impl DryRunReport {
    /// Render the report as a human-readable labeled block.
    pub fn render_plain(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Agent:       {}", self.agent_name);
        let _ = writeln!(out, "Path:        {}", self.agent_path);
        let _ = writeln!(
            out,
            "Model:       {} (source: {})",
            self.model,
            self.model_source.as_str()
        );
        let _ = writeln!(out, "Provider:    {}", self.provider);
        if let Some(system) = &self.system_prompt {
            let _ = writeln!(out, "System:      {system}");
        }
        let _ = writeln!(out, "Prompt:      {}", self.prompt_text);
        if !self.tools.is_empty() {
            let _ = writeln!(out, "Tools:");
            for t in &self.tools {
                let _ = writeln!(out, "  - {} ({})", t.name, t.source.as_str());
            }
        }
        if !self.options.is_empty() {
            let _ = writeln!(out, "Options:");
            for (k, v) in &self.options {
                let _ = writeln!(out, "  {k}: {v}");
            }
        }
        let _ = writeln!(out, "Chain limit: {}", self.chain_limit);
        if let Some(budget) = self.budget {
            let _ = writeln!(out, "Budget:      max_tokens={budget}");
        }
        if let Some(retry) = &self.retry {
            let _ = writeln!(
                out,
                "Retry:       max_retries={}, base_delay_ms={}, max_delay_ms={}, jitter={}",
                retry.max_retries, retry.base_delay_ms, retry.max_delay_ms, retry.jitter
            );
        }
        let cap_str = match self.parallel.max_concurrent {
            Some(n) => n.to_string(),
            None => "unlimited".into(),
        };
        let _ = writeln!(
            out,
            "Parallel:    enabled={}, max_concurrent={cap_str}",
            self.parallel.enabled,
        );
        let _ = writeln!(
            out,
            "Logging:     {}",
            if self.logging_enabled { "enabled" } else { "disabled" }
        );
        if let Some(prompt_json) = &self.prompt {
            let _ = writeln!(out, "Prompt (full JSON):");
            let pretty = serde_json::to_string_pretty(prompt_json)
                .unwrap_or_else(|_| prompt_json.to_string());
            for line in pretty.lines() {
                let _ = writeln!(out, "  {line}");
            }
        }
        out
    }

    /// Render the report as pretty-printed JSON.
    pub fn render_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_report() -> DryRunReport {
        let mut options = BTreeMap::new();
        options.insert("temperature".into(), serde_json::json!(0.7));
        options.insert("max_tokens".into(), serde_json::json!(200));

        DryRunReport {
            agent_name: "researcher".into(),
            agent_path: "/tmp/agents/researcher.toml".into(),
            model: "gpt-4o-mini".into(),
            model_source: ModelSource::Agent,
            provider: "openai".into(),
            system_prompt: Some("You are helpful.".into()),
            prompt_text: "hello world".into(),
            tools: vec![
                ToolEntry {
                    name: "llm_version".into(),
                    source: ToolSource::Builtin,
                },
                ToolEntry {
                    name: "web_search".into(),
                    source: ToolSource::External,
                },
            ],
            options,
            chain_limit: 10,
            budget: Some(5000),
            retry: Some(RetryConfig::default()),
            parallel: ParallelConfig::default(),
            logging_enabled: true,
            prompt: None,
        }
    }

    fn minimal_report() -> DryRunReport {
        DryRunReport {
            agent_name: "minimal".into(),
            agent_path: "/tmp/agents/minimal.toml".into(),
            model: "gpt-4o-mini".into(),
            model_source: ModelSource::Default,
            provider: "openai".into(),
            system_prompt: None,
            prompt_text: "hi".into(),
            tools: vec![],
            options: BTreeMap::new(),
            chain_limit: 5,
            budget: None,
            retry: None,
            parallel: ParallelConfig::default(),
            logging_enabled: false,
            prompt: None,
        }
    }

    #[test]
    fn plain_render_includes_all_fields() {
        let out = full_report().render_plain();
        assert!(out.contains("Agent:       researcher"));
        assert!(out.contains("/tmp/agents/researcher.toml"));
        assert!(out.contains("Model:       gpt-4o-mini"));
        assert!(out.contains("source: agent"));
        assert!(out.contains("Provider:    openai"));
        assert!(out.contains("System:      You are helpful."));
        assert!(out.contains("Prompt:      hello world"));
        assert!(out.contains("llm_version (builtin)"));
        assert!(out.contains("web_search (external)"));
        assert!(out.contains("temperature: 0.7"));
        assert!(out.contains("max_tokens: 200"));
        assert!(out.contains("Chain limit: 10"));
        assert!(out.contains("Budget:      max_tokens=5000"));
        assert!(out.contains("Retry:       max_retries=3"));
        assert!(out.contains("Logging:     enabled"));
    }

    #[test]
    fn plain_render_options_sorted() {
        let out = full_report().render_plain();
        let max_idx = out.find("max_tokens").unwrap();
        let temp_idx = out.find("temperature").unwrap();
        assert!(max_idx < temp_idx, "options should be alphabetically sorted");
    }

    #[test]
    fn plain_render_omits_absent_optional_fields() {
        let out = minimal_report().render_plain();
        assert!(!out.contains("System:"));
        assert!(!out.contains("Tools:"));
        assert!(!out.contains("Options:"));
        assert!(!out.contains("Budget:"));
        assert!(!out.contains("Retry:"));
        assert!(!out.contains("Prompt (full JSON)"));
        assert!(out.contains("Logging:     disabled"));
    }

    #[test]
    fn plain_render_includes_prompt_json_when_present() {
        let mut r = minimal_report();
        r.prompt = Some(serde_json::json!({
            "prompt": "hi",
            "system": "you are helpful"
        }));
        let out = r.render_plain();
        assert!(out.contains("Prompt (full JSON):"));
        assert!(out.contains("you are helpful"));
    }

    #[test]
    fn json_render_round_trips() {
        let r = full_report();
        let json = r.render_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["agent_name"], "researcher");
        assert_eq!(parsed["model"], "gpt-4o-mini");
        assert_eq!(parsed["model_source"], "agent");
        assert_eq!(parsed["provider"], "openai");
        assert_eq!(parsed["system_prompt"], "You are helpful.");
        assert_eq!(parsed["prompt_text"], "hello world");
        assert_eq!(parsed["tools"][0]["name"], "llm_version");
        assert_eq!(parsed["tools"][0]["source"], "builtin");
        assert_eq!(parsed["tools"][1]["source"], "external");
        assert_eq!(parsed["options"]["temperature"], 0.7);
        assert_eq!(parsed["chain_limit"], 10);
        assert_eq!(parsed["budget"], 5000);
        assert_eq!(parsed["logging_enabled"], true);
    }

    #[test]
    fn json_render_omits_none_fields() {
        let json = minimal_report().render_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("system_prompt").is_none());
        assert!(parsed.get("budget").is_none());
        assert!(parsed.get("retry").is_none());
        assert!(parsed.get("prompt").is_none());
        let obj = parsed.as_object().unwrap();
        assert!(!obj.contains_key("tools") || parsed["tools"].as_array().unwrap().is_empty());
    }

    #[test]
    fn json_render_includes_prompt_when_verbose() {
        let mut r = minimal_report();
        r.prompt = Some(serde_json::json!({"prompt": "hi"}));
        let json = r.render_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["prompt"]["prompt"], "hi");
    }
}
