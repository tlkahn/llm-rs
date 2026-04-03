pub mod keys;
pub mod logs;
pub mod models;
pub mod prompt;

use llm_core::Provider;

/// Returns all compiled-in providers.
pub fn providers() -> Vec<Box<dyn Provider>> {
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
