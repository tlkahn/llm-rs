use async_trait::async_trait;
use llm_core::retry::RetryConfig;
use llm_core::stream::ResponseStream;
use llm_core::types::{ModelInfo, Prompt};
use llm_core::{Provider, Result};

/// A provider wrapper that retries transient errors with exponential backoff.
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
                    eprintln!(
                        "[retry] Attempt {}/{} failed ({}), retrying in {:.1}s...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        e,
                        delay.as_secs_f64()
                    );
                    tokio::time::sleep(delay).await;
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::stream::Chunk;
    use llm_core::LlmError;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock provider that fails N times then succeeds.
    struct FailThenSucceed {
        fail_count: u32,
        calls: AtomicU32,
        error: LlmError,
    }

    impl FailThenSucceed {
        fn new(fail_count: u32, error: LlmError) -> Self {
            Self {
                fail_count,
                calls: AtomicU32::new(0),
                error,
            }
        }

        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Provider for FailThenSucceed {
        fn id(&self) -> &str {
            "mock"
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo::new("mock-model")]
        }

        async fn execute(
            &self,
            _model: &str,
            _prompt: &Prompt,
            _key: Option<&str>,
            _stream: bool,
        ) -> Result<ResponseStream> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                // Recreate the error each time (LlmError is not Clone)
                match &self.error {
                    LlmError::HttpError { status, message } => {
                        Err(LlmError::HttpError {
                            status: *status,
                            message: message.clone(),
                        })
                    }
                    _ => Err(LlmError::Provider("test error".into())),
                }
            } else {
                let chunks = vec![
                    Ok(Chunk::Text("success".into())),
                    Ok(Chunk::Done),
                ];
                Ok(Box::pin(futures::stream::iter(chunks)))
            }
        }
    }

    /// A provider that always fails.
    struct AlwaysFail {
        calls: AtomicU32,
        error: LlmError,
    }

    impl AlwaysFail {
        fn new(error: LlmError) -> Self {
            Self {
                calls: AtomicU32::new(0),
                error,
            }
        }

        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Provider for AlwaysFail {
        fn id(&self) -> &str {
            "mock"
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo::new("mock-model")]
        }

        async fn execute(
            &self,
            _model: &str,
            _prompt: &Prompt,
            _key: Option<&str>,
            _stream: bool,
        ) -> Result<ResponseStream> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.error {
                LlmError::HttpError { status, message } => {
                    Err(LlmError::HttpError {
                        status: *status,
                        message: message.clone(),
                    })
                }
                _ => Err(LlmError::Provider("permanent error".into())),
            }
        }
    }

    fn no_jitter_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            base_delay_ms: 1, // minimal delay for tests
            max_delay_ms: 10,
            jitter: false,
        }
    }

    #[tokio::test]
    async fn delegates_on_success() {
        let inner = FailThenSucceed::new(
            0,
            LlmError::HttpError { status: 429, message: "unused".into() },
        );
        let retry = RetryProvider::new(&inner, no_jitter_config(3));

        let prompt = Prompt::new("test");
        let stream = retry.execute("mock-model", &prompt, None, false).await;
        assert!(stream.is_ok());
        assert_eq!(inner.call_count(), 1);
    }

    #[tokio::test]
    async fn retries_on_retryable_error() {
        let inner = FailThenSucceed::new(
            2,
            LlmError::HttpError { status: 429, message: "rate limited".into() },
        );
        let retry = RetryProvider::new(&inner, no_jitter_config(3));

        let prompt = Prompt::new("test");
        let stream = retry.execute("mock-model", &prompt, None, false).await;
        assert!(stream.is_ok());
        assert_eq!(inner.call_count(), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let inner = AlwaysFail::new(
            LlmError::HttpError { status: 429, message: "rate limited".into() },
        );
        let retry = RetryProvider::new(&inner, no_jitter_config(2));

        let prompt = Prompt::new("test");
        let result = retry.execute("mock-model", &prompt, None, false).await;
        assert!(result.is_err());
        // 1 original + 2 retries = 3 total calls
        assert_eq!(inner.call_count(), 3);
        if let Err(LlmError::HttpError { status, .. }) = result {
            assert_eq!(status, 429);
        } else {
            panic!("expected HttpError");
        }
    }

    #[tokio::test]
    async fn no_retry_on_permanent_error() {
        let inner = AlwaysFail::new(
            LlmError::HttpError { status: 401, message: "unauthorized".into() },
        );
        let retry = RetryProvider::new(&inner, no_jitter_config(3));

        let prompt = Prompt::new("test");
        let result = retry.execute("mock-model", &prompt, None, false).await;
        assert!(result.is_err());
        assert_eq!(inner.call_count(), 1); // immediate failure, no retries
    }

    #[test]
    fn preserves_provider_metadata() {
        let inner = FailThenSucceed::new(
            0,
            LlmError::HttpError { status: 429, message: "unused".into() },
        );
        let retry = RetryProvider::new(&inner, RetryConfig::default());

        assert_eq!(retry.id(), "mock");
        assert_eq!(retry.models().len(), 1);
        assert_eq!(retry.needs_key(), None);
        assert_eq!(retry.key_env_var(), None);
    }
}
