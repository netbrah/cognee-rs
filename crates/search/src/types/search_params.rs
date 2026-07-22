use serde::{Deserialize, Serialize};

use crate::types::SearchRequest;

/// Per-request retriever behavior overrides.
///
/// All fields are optional. When `None`, the retriever falls back to its
/// constructor-time defaults. This lets callers override only the params
/// they care about on a per-request basis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchParams {
    /// Max number of results to return from vector search.
    pub top_k: Option<usize>,

    /// Override the LLM system prompt text directly.
    pub system_prompt: Option<String>,

    /// Override the LLM system prompt by file path.
    pub system_prompt_path: Option<String>,

    /// Number of candidates for wide graph search (before re-ranking).
    pub wide_search_top_k: Option<usize>,

    /// Distance penalty applied during triplet scoring.
    pub triplet_distance_penalty: Option<f32>,

    /// Filter graph to nodes of this type.
    pub node_type: Option<String>,

    /// Filter graph to nodes with these names.
    pub node_name: Option<Vec<String>>,

    /// "OR" (default) or "AND" for multi-name filtering.
    pub node_name_filter_operator: Option<String>,

    /// Influence weight for feedback-based re-ranking.
    pub feedback_influence: Option<f32>,

    /// Maximum CoT iterations (GraphCompletionCot).
    pub max_iter: Option<usize>,

    /// Number of context extension rounds (GraphCompletionContextExtension).
    pub context_extension_rounds: Option<usize>,

    /// Optional JSON schema for structured LLM output.
    /// When `Some`, completion-generating retrievers return `SearchOutput::Structured`
    /// instead of `SearchOutput::Text`.
    pub response_schema: Option<serde_json::Value>,

    /// Number of hops from query result nodes to include in the graph context.
    pub neighborhood_depth: Option<usize>,

    /// Number of initial seed nodes for neighborhood expansion.
    pub neighborhood_seed_top_k: Option<usize>,

    /// (HybridCompletion) Max chunks for the BM25 lexical lane.
    /// Target default when `None`: `.or(top_k).unwrap_or(10)`.
    pub chunks_top_k: Option<usize>,

    /// (HybridCompletion) Max entities for the graph lane.
    /// Target default when `None`: `.or(top_k).unwrap_or(10)`.
    pub entities_top_k: Option<usize>,

    /// (HybridCompletion) Max facts/triplets for the graph lane.
    /// Target default when `None`: `.or(top_k).unwrap_or(10)`.
    pub facts_top_k: Option<usize>,

    /// (HybridCompletion) Max edges expanded per entity.
    /// Target default when `None`: `10`.
    pub max_edges_per_entity: Option<usize>,

    /// (HybridCompletion) Max text summaries to include.
    /// Target default when `None`: `None` (no fallback — matches Python).
    pub text_summaries_top_k: Option<usize>,

    /// (HybridCompletion) Whether to weight results by `DataPoint` importance.
    /// Target default when `None`: `true`.
    pub use_importance_weight: Option<bool>,

    /// (HybridCompletion, reserved/inert — Phase 2) Whether to weight by truth score.
    /// Target default when `None`: `false`.
    pub use_truth_weight: Option<bool>,

    /// (HybridCompletion, reserved/inert — Phase 2) Whether to include the global context index.
    /// Target default when `None`: `false`.
    pub include_global_context_index: Option<bool>,

    /// (HybridCompletion, reserved/inert — Phase 2) Max entries from the global context index.
    /// Target default when `None`: `3`.
    pub global_context_index_top_k: Option<usize>,
}

impl SearchParams {
    pub fn top_k_or(&self, default: usize) -> usize {
        self.top_k.unwrap_or(default)
    }

    pub fn wide_search_top_k_or(&self, default: usize) -> usize {
        self.wide_search_top_k.unwrap_or(default)
    }

    pub fn triplet_distance_penalty_or(&self, default: f32) -> f32 {
        self.triplet_distance_penalty.unwrap_or(default)
    }

    pub fn feedback_influence_or(&self, default: f32) -> f32 {
        self.feedback_influence.unwrap_or(default)
    }
}

impl From<&SearchRequest> for SearchParams {
    fn from(req: &SearchRequest) -> Self {
        Self {
            top_k: req.top_k,
            system_prompt: req.system_prompt.clone(),
            system_prompt_path: req.system_prompt_path.clone(),
            wide_search_top_k: req.wide_search_top_k,
            triplet_distance_penalty: req.triplet_distance_penalty,
            node_type: req.node_type.clone(),
            node_name: req.node_name.clone(),
            node_name_filter_operator: req.node_name_filter_operator.clone(),
            feedback_influence: req.feedback_influence,
            max_iter: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("max_iter"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            context_extension_rounds: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("context_extension_rounds"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            response_schema: req.response_schema.clone(),
            neighborhood_depth: req.neighborhood_depth,
            neighborhood_seed_top_k: req.neighborhood_seed_top_k,
            chunks_top_k: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("chunks_top_k"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            entities_top_k: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("entities_top_k"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            facts_top_k: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("facts_top_k"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            max_edges_per_entity: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("max_edges_per_entity"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            text_summaries_top_k: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("text_summaries_top_k"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            use_importance_weight: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("use_importance_weight"))
                .and_then(|v| v.as_bool()),
            use_truth_weight: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("use_truth_weight"))
                .and_then(|v| v.as_bool()),
            include_global_context_index: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("include_global_context_index"))
                .and_then(|v| v.as_bool()),
            global_context_index_top_k: req
                .retriever_specific_config
                .as_ref()
                .and_then(|c| c.get("global_context_index_top_k"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::SearchParams;
    use crate::types::SearchRequest;

    #[test]
    fn hybrid_knobs_extracted_from_retriever_specific_config() {
        let json = r#"{
            "query_text": "test",
            "retriever_specific_config": {
                "chunks_top_k": 5,
                "entities_top_k": 6,
                "facts_top_k": 7,
                "max_edges_per_entity": 8,
                "text_summaries_top_k": 9,
                "use_importance_weight": false,
                "use_truth_weight": true,
                "include_global_context_index": true,
                "global_context_index_top_k": 4
            }
        }"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        let params = SearchParams::from(&req);
        assert_eq!(params.chunks_top_k, Some(5));
        assert_eq!(params.entities_top_k, Some(6));
        assert_eq!(params.facts_top_k, Some(7));
        assert_eq!(params.max_edges_per_entity, Some(8));
        assert_eq!(params.text_summaries_top_k, Some(9));
        assert_eq!(params.use_importance_weight, Some(false));
        assert_eq!(params.use_truth_weight, Some(true));
        assert_eq!(params.include_global_context_index, Some(true));
        assert_eq!(params.global_context_index_top_k, Some(4));
    }

    #[test]
    fn hybrid_knobs_default_none_when_absent() {
        let json = r#"{
            "query_text": "test",
            "retriever_specific_config": {}
        }"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        let params = SearchParams::from(&req);
        assert_eq!(params.chunks_top_k, None);
        assert_eq!(params.entities_top_k, None);
        assert_eq!(params.facts_top_k, None);
        assert_eq!(params.max_edges_per_entity, None);
        assert_eq!(params.text_summaries_top_k, None);
        assert_eq!(params.use_importance_weight, None);
        assert_eq!(params.use_truth_weight, None);
        assert_eq!(params.include_global_context_index, None);
        assert_eq!(params.global_context_index_top_k, None);
    }
}
