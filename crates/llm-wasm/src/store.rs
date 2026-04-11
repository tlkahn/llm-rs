//! WASM `ConversationStore` backend driven by JS-provided callbacks.
//!
//! A `JsConversationStore` holds four JS `Function`s:
//!   - `logResponse(conversationIdOrNull, model, response) -> Promise<string>`
//!   - `readConversation(id) -> Promise<{meta, responses}>`
//!   - `listConversations(limit) -> Promise<Array<summary>>`
//!   - `latestConversationId() -> Promise<string | null>`
//!
//! Each promise must eventually resolve to the serialized form of the
//! corresponding Rust type (as produced by `serde_wasm_bindgen`).

use async_trait::async_trait;
use js_sys::{Function, Promise};
use llm_core::types::{Options, Response, ToolCall, ToolResult, Usage};
use llm_core::{LlmError, Result};
use llm_store::{ConversationRecord, ConversationStore, ConversationSummary};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

/// wasm32-safe `Response` builder. Mirrors `llm_store::build_response`
/// (native-only, pulls `ulid` + `chrono`) using JS APIs: `crypto.randomUUID`
/// for a unique id and `Date` for the datetime.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_response_wasm(
    model: &str,
    prompt: &str,
    system: Option<&str>,
    options: Options,
    response_text: String,
    usage: Option<Usage>,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    schema: Option<serde_json::Value>,
    schema_id: Option<String>,
    duration_ms: u64,
) -> Response {
    let id = js_sys::Reflect::get(
        &js_sys::global(),
        &JsValue::from_str("crypto"),
    )
    .ok()
    .and_then(|crypto| {
        js_sys::Reflect::get(&crypto, &JsValue::from_str("randomUUID"))
            .ok()
            .and_then(|f| f.dyn_into::<Function>().ok())
            .and_then(|f| f.call0(&crypto).ok())
            .and_then(|v| v.as_string())
    })
    .unwrap_or_else(|| format!("r-{}", js_sys::Date::now() as u64));

    let datetime = {
        let iso = js_sys::Date::new_0().to_iso_string();
        iso.as_string().unwrap_or_default()
    };

    Response {
        id,
        model: model.to_string(),
        prompt: prompt.to_string(),
        system: system.map(|s| s.to_string()),
        response: response_text,
        options,
        usage,
        tool_calls,
        tool_results,
        attachments: Vec::new(),
        schema,
        schema_id,
        duration_ms,
        datetime,
    }
}

/// JS-callback-backed conversation store.
///
/// Cloneable and cheap to share; the four underlying `Function` handles are
/// ref-counted by `js_sys`.
#[derive(Clone)]
pub(crate) struct JsConversationStore {
    log_response_fn: Function,
    read_conversation_fn: Function,
    list_conversations_fn: Function,
    latest_conversation_id_fn: Function,
}

impl JsConversationStore {
    /// Parse a JS spec `{ logResponse, readConversation, listConversations, latestConversationId }`.
    pub fn from_spec(spec: JsValue) -> std::result::Result<Self, JsError> {
        let get = |key: &str| -> std::result::Result<Function, JsError> {
            let v = js_sys::Reflect::get(&spec, &JsValue::from_str(key))
                .map_err(|_| JsError::new(&format!("spec.{key} is missing")))?;
            if !v.is_function() {
                return Err(JsError::new(&format!("spec.{key} must be a function")));
            }
            Ok(v.unchecked_into::<Function>())
        };
        Ok(Self {
            log_response_fn: get("logResponse")?,
            read_conversation_fn: get("readConversation")?,
            list_conversations_fn: get("listConversations")?,
            latest_conversation_id_fn: get("latestConversationId")?,
        })
    }
}

async fn invoke_awaiting(f: &Function, args: &js_sys::Array) -> Result<JsValue> {
    let this = JsValue::NULL;
    let ret = f
        .apply(&this, args)
        .map_err(|e| LlmError::Store(format!("store callback threw: {:?}", e)))?;
    // If the return value is a Promise, await it. Otherwise treat it as the
    // already-resolved value (the JS side might expose a synchronous store).
    if ret.is_instance_of::<Promise>() {
        let promise: Promise = ret.unchecked_into();
        JsFuture::from(promise)
            .await
            .map_err(|e| LlmError::Store(format!("store promise rejected: {:?}", e)))
    } else {
        Ok(ret)
    }
}

#[async_trait(?Send)]
impl ConversationStore for JsConversationStore {
    async fn log_response(
        &self,
        conversation_id: Option<&str>,
        model: &str,
        response: &Response,
    ) -> Result<String> {
        let cid_js = match conversation_id {
            Some(c) => JsValue::from_str(c),
            None => JsValue::NULL,
        };
        let response_js = serde_wasm_bindgen::to_value(response)
            .map_err(|e| LlmError::Store(format!("serialize Response: {e}")))?;
        let args = js_sys::Array::new();
        args.push(&cid_js);
        args.push(&JsValue::from_str(model));
        args.push(&response_js);
        let ret = invoke_awaiting(&self.log_response_fn, &args).await?;
        ret.as_string()
            .ok_or_else(|| LlmError::Store("logResponse must resolve to a string".into()))
    }

    async fn read_conversation(
        &self,
        id: &str,
    ) -> Result<(ConversationRecord, Vec<Response>)> {
        let args = js_sys::Array::new();
        args.push(&JsValue::from_str(id));
        let ret = invoke_awaiting(&self.read_conversation_fn, &args).await?;
        // Expect { meta: ConversationRecord, responses: [Response, ...] }.
        let meta_js = js_sys::Reflect::get(&ret, &JsValue::from_str("meta"))
            .map_err(|_| LlmError::Store("readConversation result missing `meta`".into()))?;
        let responses_js = js_sys::Reflect::get(&ret, &JsValue::from_str("responses"))
            .map_err(|_| LlmError::Store("readConversation result missing `responses`".into()))?;
        let meta: ConversationRecord = serde_wasm_bindgen::from_value(meta_js)
            .map_err(|e| LlmError::Store(format!("deserialize meta: {e}")))?;
        let responses: Vec<Response> = serde_wasm_bindgen::from_value(responses_js)
            .map_err(|e| LlmError::Store(format!("deserialize responses: {e}")))?;
        Ok((meta, responses))
    }

    async fn list_conversations(&self, limit: usize) -> Result<Vec<ConversationSummary>> {
        let args = js_sys::Array::new();
        args.push(&JsValue::from_f64(limit as f64));
        let ret = invoke_awaiting(&self.list_conversations_fn, &args).await?;
        serde_wasm_bindgen::from_value(ret)
            .map_err(|e| LlmError::Store(format!("deserialize summaries: {e}")))
    }

    async fn latest_conversation_id(&self) -> Result<Option<String>> {
        let args = js_sys::Array::new();
        let ret = invoke_awaiting(&self.latest_conversation_id_fn, &args).await?;
        if ret.is_null() || ret.is_undefined() {
            Ok(None)
        } else {
            ret.as_string()
                .map(Some)
                .ok_or_else(|| LlmError::Store("latestConversationId returned non-string".into()))
        }
    }
}
