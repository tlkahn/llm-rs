use async_trait::async_trait;
use llm_core::retry::RetryConfig;
use llm_core::stream::ResponseStream;
use llm_core::types::{ModelInfo, Prompt};
use llm_core::{Provider, Result};

pub struct RetryProvider<'a> {
    inner: &'a dyn Provider,
    config: RetryConfig,
}

impl<'a> RetryProvider<'a> {
    pub fn new(inner: &'a dyn Provider, config: RetryConfig) -> Self {
        Self { inner, config }
    }
}

#[async_trait]
impl Provider for RetryProvider<'_> {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.inner.models()
    }

    fn needs_key(&self) -> Option<&str> {
        self.inner.needs_key()
    }

    fn key_env_var(&self) -> Option<&str> {
        self.inner.key_env_var()
    }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> Result<ResponseStream> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner.execute(model, prompt, key, stream).await {
                Ok(s) => return Ok(s),
                Err(e) if e.is_retryable() && attempt < self.config.max_retries => {
                    let delay = self.config.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap())
    }
}
