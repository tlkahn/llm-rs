use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("model error: {0}")]
    Model(String),

    #[error("no key found: {0}")]
    NeedsKey(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("store error: {0}")]
    Store(String),
}

pub type Result<T> = std::result::Result<T, LlmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_model() {
        let err = LlmError::Model("rate limited".into());
        assert_eq!(err.to_string(), "model error: rate limited");
    }

    #[test]
    fn error_display_needs_key() {
        let err = LlmError::NeedsKey(
            "No key found - set one with 'llm keys set openai'".into(),
        );
        assert!(err.to_string().contains("llm keys set openai"));
    }

    #[test]
    fn error_display_provider() {
        let err = LlmError::Provider("connection timeout".into());
        assert_eq!(err.to_string(), "provider error: connection timeout");
    }

    #[test]
    fn error_display_config() {
        let err = LlmError::Config("invalid TOML".into());
        assert_eq!(err.to_string(), "config error: invalid TOML");
    }

    #[test]
    fn error_display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: LlmError = io_err.into();
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn error_display_store() {
        let err = LlmError::Store("conversation not found".into());
        assert_eq!(err.to_string(), "store error: conversation not found");
    }

    #[test]
    fn error_io_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let llm_err: LlmError = io_err.into();
        assert!(matches!(llm_err, LlmError::Io(_)));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmError>();
    }

    #[test]
    fn result_type_works() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: Result<i32> = Err(LlmError::Config("bad".into()));
        assert!(err.is_err());
    }
}
