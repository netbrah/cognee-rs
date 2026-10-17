use std::collections::HashMap;

use chrono::{DateTime, Utc};
use cognee_llm::Message;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Graph element IDs that were used to produce the answer for a Q&A entry.
///
/// Matches the Python `used_graph_element_ids` dict: `{"node_ids": [...], "edge_ids": [...]}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsedGraphElementIds {
    #[serde(default)]
    pub node_ids: Vec<String>,
    #[serde(default)]
    pub edge_ids: Vec<String>,
}

/// A single question-answer entry stored in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionQAEntry {
    pub id: Uuid,
    pub session_id: String,
    pub user_id: Option<String>,
    pub question: String,
    pub answer: String,
    pub context: Option<String>,
    pub created_at: DateTime<Utc>,
    /// User-provided feedback text for this Q&A entry.
    #[serde(default)]
    pub feedback_text: Option<String>,
    /// User-provided feedback score (1-5 rating, validated on write).
    #[serde(default)]
    pub feedback_score: Option<i32>,
    /// Graph node/edge IDs that were used to produce the answer.
    #[serde(default)]
    pub used_graph_element_ids: Option<UsedGraphElementIds>,
    /// Metadata tracking for the memify pipeline (e.g. `{"feedback_weights_applied": true}`).
    #[serde(default)]
    pub memify_metadata: Option<HashMap<String, bool>>,
}

/// Session context passed to retrievers: the session ID and any loaded
/// conversation history (as LLM messages).
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    pub session_id: Option<String>,
    pub history: Vec<Message>,
    pub formatted_history: String,
    /// Stored knowledge-graph snapshot to prepend to history.
    ///
    /// Set by `improve()` stage 4 (`sync_graph_to_session`) and loaded by the
    /// search orchestrator before retrieval, so follow-up questions benefit from
    /// prior graph knowledge. Matches Python's `get_graph_context` / prepend
    /// logic in `session_manager.py:435-450`.
    pub graph_context: Option<String>,
}

/// One agent-trace step persisted in a session — mirrors Python's
/// `SessionAgentTraceEntry` (no `created_at`; ordering is positional).
///
/// Library-internal type; kept `snake_case` to match Python's persisted JSON
/// shape so cross-SDK reads stay byte-equal. This is **not** an HTTP DTO —
/// the wire-side camelCase rule (Decision 10) does not apply here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTraceStep {
    /// Server-generated UUID4 — returned by `SessionManager::add_agent_trace_step`.
    pub trace_id: String,
    pub origin_function: String,
    /// Free-form per Python validator; typically `"success"` / `"error"`.
    pub status: String,
    #[serde(default)]
    pub memory_query: String,
    #[serde(default)]
    pub memory_context: String,
    /// Default `{}` — matches Python's `default_factory=dict`.
    #[serde(default = "empty_object")]
    pub method_params: serde_json::Value,
    #[serde(default)]
    pub method_return_value: Option<serde_json::Value>,
    #[serde(default)]
    pub error_message: String,
    /// LLM-resolved feedback string — generation owned by `SessionManager`
    /// (LIB-01); callers in this task pass `""` until LIB-01 lands.
    #[serde(default)]
    pub session_feedback: String,
}

fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

/// Storage-boundary format for `user_id`, shared by every session backend.
///
/// Callers disagree on UUID formatting for the same identity: the remember
/// write path passes `Uuid::to_string()` (hyphenated) while the session
/// routers pass the hex id produced by `uuid_hex::to_hex`. Every `user_id`
/// crossing into storage — in writes, read filters, and Redis keys — is
/// reduced to lower-case hex here, so both formats address the same rows
/// regardless of which caller wrote them. UUID hyphens are position-only
/// separators, so stripping them is lossless; non-UUID ids (e.g. "default")
/// pass through unchanged apart from case.
pub(crate) fn norm_user_id(user_id: &str) -> String {
    user_id.replace('-', "").to_lowercase()
}

/// Convenience over [`norm_user_id`] for the `Option<&str>` caller shape.
pub(crate) fn norm_user_id_opt(user_id: Option<&str>) -> Option<String> {
    user_id.map(norm_user_id)
}
