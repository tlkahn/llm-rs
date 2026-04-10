pub mod aliases;
pub mod chat;
pub mod keys;
pub mod logs;
pub mod models;
pub mod options;
pub mod plugins;
pub mod prompt;
pub mod schemas;
pub mod tools;

use llm_core::{Config, Options, Provider, parse_option_value};

use crate::subprocess;

/// Returns all compiled-in providers.
pub fn compiled_providers() -> Vec<Box<dyn Provider>> {
    let mut providers: Vec<Box<dyn Provider>> = Vec::new();

    #[cfg(feature = "openai")]
    {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        providers.push(Box::new(llm_openai::provider::OpenAiProvider::new(&base_url)));
    }

    #[cfg(feature = "anthropic")]
    {
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        providers.push(Box::new(llm_anthropic::provider::AnthropicProvider::new(&base_url)));
    }

    providers
}

/// Returns all providers: compiled-in + discovered subprocess providers.
pub async fn providers() -> Vec<Box<dyn Provider>> {
    let mut all = compiled_providers();

    for path in subprocess::discovery::discover_providers() {
        match subprocess::provider::SubprocessProvider::from_binary(path.clone()).await {
            Ok(p) => all.push(Box::new(p)),
            Err(e) => eprintln!(
                "warning: skipping provider {}: {e}",
                path.display()
            ),
        }
    }

    all
}

/// Merge config-level options for a model with CLI `-o` overrides.
/// CLI options win on conflict.
pub(crate) fn build_options(config: &Config, model_id: &str, cli_options: &[String]) -> Options {
    let mut opts = config.model_options(model_id);
    for pair in cli_options.chunks(2) {
        if pair.len() == 2 {
            opts.insert(pair[0].clone(), parse_option_value(&pair[1]));
        }
    }
    opts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_options_empty() {
        let config = Config::default();
        let opts = build_options(&config, "gpt-4o", &[]);
        assert!(opts.is_empty());
    }

    #[test]
    fn build_options_config_only() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.7));
        let opts = build_options(&config, "gpt-4o", &[]);
        assert_eq!(opts["temperature"], serde_json::json!(0.7));
    }

    #[test]
    fn build_options_cli_only() {
        let config = Config::default();
        let cli = vec!["temperature".into(), "0.9".into()];
        let opts = build_options(&config, "gpt-4o", &cli);
        assert_eq!(opts["temperature"], serde_json::json!(0.9));
    }

    #[test]
    fn build_options_cli_overrides_config() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.5));
        let cli = vec!["temperature".into(), "1.0".into()];
        let opts = build_options(&config, "gpt-4o", &cli);
        assert_eq!(opts["temperature"], serde_json::json!(1.0));
    }

    #[test]
    fn build_options_merge() {
        let mut config = Config::default();
        config.set_option("gpt-4o", "temperature", serde_json::json!(0.5));
        let cli = vec!["max_tokens".into(), "200".into()];
        let opts = build_options(&config, "gpt-4o", &cli);
        assert_eq!(opts["temperature"], serde_json::json!(0.5));
        assert_eq!(opts["max_tokens"], serde_json::json!(200));
    }
}
