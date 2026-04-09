pub mod chain;
pub mod config;
pub mod error;
pub mod provider;
pub mod schema;
pub mod stream;
pub mod types;

// Re-export key types at crate root for convenience
pub use chain::{ChainEvent, ChainResult, ToolExecutor, chain};
pub use config::{Config, KeyStore, Paths, resolve_key};
pub use error::{LlmError, Result};
pub use provider::Provider;
pub use schema::{multi_schema, parse_schema_dsl};
pub use stream::{Chunk, ResponseStream, collect_text, collect_tool_calls, collect_usage};
pub use types::{
    Attachment, AttachmentSource, Message, ModelInfo, Options, Prompt, Response, Role, Tool,
    ToolCall, ToolResult, Usage,
};
