use std::time::Duration;

use serde::{Deserialize, Serialize};

fn default_max_retries() -> u32 {
    3
}
fn default_base_delay_ms() -> u64 {
    1000
}
fn default_max_delay_ms() -> u64 {
    30_000
}
fn default_jitter() -> bool {
    true
}

/// Configuration for retry with exponential backoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries (default: 3).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Base delay in milliseconds (default: 1000).
    #[serde(default = "default_base_delay_ms")]
    pub base_delay_ms: u64,

    /// Maximum delay in milliseconds (default: 30000).
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,

    /// Whether to add jitter to delays (default: true).
    #[serde(default = "default_jitter")]
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            base_delay_ms: default_base_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            jitter: default_jitter(),
        }
    }
}

impl RetryConfig {
    /// Compute the delay for a given attempt (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let exp = self
            .base_delay_ms
            .saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
        let capped = exp.min(self.max_delay_ms);
        if self.jitter {
            Duration::from_millis(fastrand::u64(0..=capped))
        } else {
            Duration::from_millis(capped)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 30_000);
        assert!(config.jitter);
    }

    #[test]
    fn delay_exponential_no_jitter() {
        let config = RetryConfig {
            jitter: false,
            ..Default::default()
        };
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(1000));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(2000));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(4000));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(8000));
    }

    #[test]
    fn delay_capped_at_max() {
        let config = RetryConfig {
            jitter: false,
            max_delay_ms: 30_000,
            ..Default::default()
        };
        // 2^10 * 1000 = 1_024_000, but capped at 30_000
        assert_eq!(config.delay_for_attempt(10), Duration::from_millis(30_000));
    }

    #[test]
    fn delay_with_jitter_in_bounds() {
        let config = RetryConfig::default();
        for _ in 0..100 {
            let delay = config.delay_for_attempt(0);
            assert!(delay <= Duration::from_millis(1000));
        }
    }

    #[test]
    fn delay_attempt_zero() {
        let config = RetryConfig {
            jitter: false,
            ..Default::default()
        };
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(1000));
    }

    #[test]
    fn serde_roundtrip() {
        let config = RetryConfig {
            max_retries: 5,
            base_delay_ms: 500,
            max_delay_ms: 10_000,
            jitter: false,
        };
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: RetryConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.max_retries, 5);
        assert_eq!(parsed.base_delay_ms, 500);
        assert_eq!(parsed.max_delay_ms, 10_000);
        assert!(!parsed.jitter);
    }

    #[test]
    fn serde_defaults_from_empty() {
        let parsed: RetryConfig = toml::from_str("").unwrap();
        assert_eq!(parsed.max_retries, 3);
        assert_eq!(parsed.base_delay_ms, 1000);
        assert_eq!(parsed.max_delay_ms, 30_000);
        assert!(parsed.jitter);
    }
}
