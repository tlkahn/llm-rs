use async_trait::async_trait;

use crate::error::Result;
use crate::stream::ResponseStream;
use crate::types::{ModelInfo, Prompt};

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn models(&self) -> Vec<ModelInfo>;

    fn needs_key(&self) -> Option<&str> {
        None
    }

    fn key_env_var(&self) -> Option<&str> {
        None
    }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> Result<ResponseStream>;
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Provider {
    fn id(&self) -> &str;
    fn models(&self) -> Vec<ModelInfo>;

    fn needs_key(&self) -> Option<&str> {
        None
    }

    fn key_env_var(&self) -> Option<&str> {
        None
    }

    async fn execute(
        &self,
        model: &str,
        prompt: &Prompt,
        key: Option<&str>,
        stream: bool,
    ) -> Result<ResponseStream>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;
    use crate::stream::Chunk;
    use futures::StreamExt;

    // A mock provider for testing the trait contract
    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        fn id(&self) -> &str {
            "mock"
        }

        fn models(&self) -> Vec<ModelInfo> {
            vec![
                ModelInfo::new("mock-fast"),
                ModelInfo {
                    id: "mock-smart".into(),
                    can_stream: true,
                    supports_tools: true,
                    supports_schema: true,
                    attachment_types: vec!["image/png".into()],
                },
            ]
        }

        async fn execute(
            &self,
            _model: &str,
            _prompt: &Prompt,
            _key: Option<&str>,
            _stream: bool,
        ) -> Result<ResponseStream> {
            let chunks = vec![
                Ok(Chunk::Text("Hello from mock".into())),
                Ok(Chunk::Done),
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }

    // A provider that requires a key
    struct KeyProvider;

    #[async_trait]
    impl Provider for KeyProvider {
        fn id(&self) -> &str {
            "key-provider"
        }

        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo::new("key-model")]
        }

        fn needs_key(&self) -> Option<&str> {
            Some("test_key")
        }

        fn key_env_var(&self) -> Option<&str> {
            Some("TEST_API_KEY")
        }

        async fn execute(
            &self,
            _model: &str,
            _prompt: &Prompt,
            key: Option<&str>,
            _stream: bool,
        ) -> Result<ResponseStream> {
            let key = key.ok_or_else(|| LlmError::NeedsKey("test_key required".into()))?;
            let chunks = vec![
                Ok(Chunk::Text(format!("key={key}"))),
                Ok(Chunk::Done),
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }
    }

    #[test]
    fn provider_id() {
        let p = MockProvider;
        assert_eq!(p.id(), "mock");
    }

    #[test]
    fn provider_lists_models() {
        let p = MockProvider;
        let models = p.models();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "mock-fast");
        assert!(models[1].supports_tools);
    }

    #[test]
    fn provider_needs_key_defaults_to_none() {
        let p = MockProvider;
        assert_eq!(p.needs_key(), None);
        assert_eq!(p.key_env_var(), None);
    }

    #[test]
    fn provider_needs_key_returns_alias() {
        let p = KeyProvider;
        assert_eq!(p.needs_key(), Some("test_key"));
        assert_eq!(p.key_env_var(), Some("TEST_API_KEY"));
    }

    #[tokio::test]
    async fn provider_execute_returns_stream() {
        let p = MockProvider;
        let prompt = Prompt::new("Hello");
        let stream = p.execute("mock-fast", &prompt, None, true).await.unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 2);
        if let Ok(Chunk::Text(t)) = &chunks[0] {
            assert_eq!(t, "Hello from mock");
        } else {
            panic!("expected Text chunk");
        }
    }

    #[tokio::test]
    async fn provider_execute_with_key() {
        let p = KeyProvider;
        let prompt = Prompt::new("Hello");
        let stream = p
            .execute("key-model", &prompt, Some("sk-test"), true)
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        if let Ok(Chunk::Text(t)) = &chunks[0] {
            assert_eq!(t, "key=sk-test");
        } else {
            panic!("expected Text chunk");
        }
    }

    #[tokio::test]
    async fn provider_execute_without_key_errors() {
        let p = KeyProvider;
        let prompt = Prompt::new("Hello");
        let result = p.execute("key-model", &prompt, None, true).await;
        assert!(result.is_err());
        if let Err(LlmError::NeedsKey(msg)) = result {
            assert!(msg.contains("test_key"));
        } else {
            panic!("expected NeedsKey error");
        }
    }
}
