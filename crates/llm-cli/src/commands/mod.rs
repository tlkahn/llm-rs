pub mod chat;
pub mod keys;
pub mod logs;
pub mod models;
pub mod plugins;
pub mod prompt;
pub mod schemas;
pub mod tools;

use llm_core::Provider;

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
