pub mod config;
pub mod error;
pub mod provider;
pub mod stream;
pub mod types;

// Re-export key types at crate root for convenience
pub use config::{Config, KeyStore, Paths, resolve_key};
pub use error::{LlmError, Result};
pub use provider::Provider;
pub use stream::{Chunk, ResponseStream, collect_text, collect_tool_calls, collect_usage};
pub use types::{
    Attachment, AttachmentSource, ModelInfo, Options, Prompt, Response, Tool, ToolCall, ToolResult,
    Usage,
};
