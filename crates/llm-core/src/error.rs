use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("model error: {0}")]
    Model(String),

    #[error("no key found: {0}")]
    NeedsKey(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("HTTP error {status}: {message}")]
    HttpError { status: u16, message: String },

    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("store error: {0}")]
    Store(String),
}

impl LlmError {
    /// Returns `true` if this error is transient and worth retrying.
    /// Only HTTP 429 (rate limit) and 5xx (server errors) are retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(self, LlmError::HttpError { status, .. }
            if *status == 429 || (500..=599).contains(status))
    }
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

    #[test]
    fn error_display_http() {
        let err = LlmError::HttpError { status: 429, message: "rate limited".into() };
        assert_eq!(err.to_string(), "HTTP error 429: rate limited");
    }

    #[test]
    fn http_error_retryable_429() {
        let err = LlmError::HttpError { status: 429, message: "rate limited".into() };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_error_retryable_5xx() {
        for status in [500, 502, 503, 504] {
            let err = LlmError::HttpError { status, message: "server error".into() };
            assert!(err.is_retryable(), "status {status} should be retryable");
        }
    }

    #[test]
    fn http_error_not_retryable_4xx() {
        for status in [400, 401, 403, 404, 422] {
            let err = LlmError::HttpError { status, message: "client error".into() };
            assert!(!err.is_retryable(), "status {status} should not be retryable");
        }
    }

    #[test]
    fn non_http_errors_not_retryable() {
        assert!(!LlmError::Provider("fail".into()).is_retryable());
        assert!(!LlmError::Model("bad".into()).is_retryable());
        assert!(!LlmError::NeedsKey("key".into()).is_retryable());
        assert!(!LlmError::Config("cfg".into()).is_retryable());
        assert!(!LlmError::Store("store".into()).is_retryable());
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "io");
        assert!(!LlmError::Io(io_err).is_retryable());
    }
}
