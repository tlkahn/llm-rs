pub mod records;
pub mod store;

#[cfg(not(target_arch = "wasm32"))]
pub mod logs;
#[cfg(not(target_arch = "wasm32"))]
pub mod query;

#[cfg(not(target_arch = "wasm32"))]
pub use logs::{LogStore, conversation_name, reconstruct_messages};
#[cfg(not(target_arch = "wasm32"))]
pub use query::{
    ListOptions, latest_conversation_id, list_conversations, list_conversations_filtered,
};
pub use records::{ConversationRecord, LineRecord, ResponseRecord};
pub use store::{ConversationStore, ConversationSummary};

#[cfg(not(target_arch = "wasm32"))]
pub use store::build_response;
