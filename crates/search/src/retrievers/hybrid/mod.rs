//! Hybrid-retrieval retriever ([`HybridRetriever`], `SearchType::HybridCompletion`).
//!
//! Port of Python cognee's `cognee/modules/retrieval/hybrid_retriever.py` and
//! `cognee/modules/retrieval/hybrid/{results,pairs,ranking,chunks}.py`. The
//! retriever fuses three lanes into a single completion context: a chunk lane
//! that merges a per-query Okapi BM25 lexical channel (landed separately as
//! [`crate::retrievers::bm25_scored_chunks`]) with the vector
//! `DocumentChunk_text` / `TextSummary_text` channels via Reciprocal Rank
//! Fusion (RRF, with optional importance-weight boosting), an entity lane, and a
//! standalone-facts lane.
//!
//! [`HybridRetriever`] implements the crate's `SearchRetriever` trait and is
//! wired to `SearchType::HybridCompletion`; the chunk-ranking spine lives in the
//! private submodules below and is orchestrated via [`retrieve_hybrid_chunks`].

mod chunks;
mod context;
mod entities;
mod facts;
mod pairs;
mod ranking;
mod results;

pub(crate) use chunks::{HybridChunksResult, retrieve_hybrid_chunks, search_collection};
pub(crate) use context::extract_used_ids;
pub(crate) use entities::{build_entities, format_entities};
pub(crate) use facts::{edge_rank_by_id, format_facts, select_facts};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tracing::debug;
use uuid::Uuid;

use cognee_embedding::EmbeddingEngine;
use cognee_graph::GraphDBTrait;
use cognee_llm::{GenerationOptions, Llm};
use cognee_session::SessionContext;
use cognee_vector::VectorDB;

use self::context::{format_hybrid_context, format_passages};
use self::entities::{EdgeBullet, EntityResult};
use self::facts::FactResult;
use self::results::result_id;
use crate::retrievers::SearchRetriever;
use crate::types::{
    SearchContext, SearchError, SearchItem, SearchOutput, SearchParams, SearchType,
};
use crate::utils::{
    DEFAULT_HYBRID_USER_PROMPT_TEMPLATE, build_messages_with_history, render_user_prompt,
    resolve_system_prompt,
};

/// Effective top-k default for the chunks/entities/facts lanes when neither a
/// request-level knob nor `top_k` is supplied.
///
/// Python's class default is 5, but the search factory overrides all three with
/// `retriever_specific_config.get(key, top_k)` where `top_k` defaults to 15 in
/// `cognee.search()`. The class default is therefore unreachable through the
/// search API; Rust uses 15 so no-args behavior matches Python end-to-end.
const DEFAULT_TOP_K: usize = 15;
/// Default max edges expanded per entity (`HybridRetriever.__init__`).
const DEFAULT_MAX_EDGES_PER_ENTITY: usize = 10;
/// Default global-context-index top-k (inert in Phase 1).
const DEFAULT_GLOBAL_CONTEXT_INDEX_TOP_K: usize = 3;
/// Default node-name filter operator.
const DEFAULT_NODE_NAME_FILTER_OPERATOR: &str = "OR";

const ENTITY_DATA_TYPE: &str = "Entity";
const ENTITY_FIELD: &str = "name";
const EDGE_TYPE_DATA_TYPE: &str = "EdgeType";
const EDGE_TYPE_FIELD: &str = "relationship_name";

/// Completion retriever fusing the chunk lane (P1-07) and the entity/facts lane
/// (P1-08) into a single `SearchType::HybridCompletion` retriever.
///
/// Port of Python's `HybridRetriever` (`hybrid_retriever.py`). Embeds the query
/// once, runs the chunk lane and the entity/facts lane concurrently, tags each
/// lane's output with a `"kind"` payload discriminator, and renders the
/// sectioned prompt context Python feeds the LLM.
///
/// **Phase-1 scope:** the truth-subspace (`use_truth_weight`) and global-context
/// (`include_global_context_index` / `global_context_index_top_k`) knobs are
/// accepted on the wire but inert — no truth-subspace alignment or global-context
/// section is produced (locked deferral).
pub struct HybridRetriever {
    vector_db: Arc<dyn VectorDB>,
    embedding_engine: Arc<dyn EmbeddingEngine>,
    graph_db: Arc<dyn GraphDBTrait>,
    llm: Arc<dyn Llm>,
    chunks_top_k: usize,
    entities_top_k: usize,
    facts_top_k: usize,
    max_edges_per_entity: usize,
    text_summaries_top_k: Option<usize>,
    use_importance_weight: bool,
    // Accepted on the wire but inert in Phase 1 (locked deferral).
    use_truth_weight: bool,
    include_global_context_index: bool,
    #[allow(
        dead_code,
        reason = "global-context-index knob is accepted on the wire but inert in Phase 1"
    )]
    global_context_index_top_k: usize,
    node_name: Option<Vec<String>>,
    node_name_filter_operator: String,
    system_prompt: Option<String>,
    system_prompt_path: Option<String>,
    user_prompt_template: Option<String>,
    generation_options: Option<GenerationOptions>,
}

impl HybridRetriever {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vector_db: Arc<dyn VectorDB>,
        embedding_engine: Arc<dyn EmbeddingEngine>,
        graph_db: Arc<dyn GraphDBTrait>,
        llm: Arc<dyn Llm>,
        chunks_top_k: Option<usize>,
        entities_top_k: Option<usize>,
        facts_top_k: Option<usize>,
        max_edges_per_entity: Option<usize>,
        text_summaries_top_k: Option<usize>,
        use_importance_weight: Option<bool>,
        use_truth_weight: Option<bool>,
        include_global_context_index: Option<bool>,
        global_context_index_top_k: Option<usize>,
        node_name: Option<Vec<String>>,
        node_name_filter_operator: Option<String>,
        system_prompt: Option<String>,
        system_prompt_path: Option<String>,
        user_prompt_template: Option<String>,
        generation_options: Option<GenerationOptions>,
    ) -> Self {
        Self {
            vector_db,
            embedding_engine,
            graph_db,
            llm,
            chunks_top_k: chunks_top_k.unwrap_or(DEFAULT_TOP_K),
            entities_top_k: entities_top_k.unwrap_or(DEFAULT_TOP_K),
            facts_top_k: facts_top_k.unwrap_or(DEFAULT_TOP_K),
            max_edges_per_entity: max_edges_per_entity.unwrap_or(DEFAULT_MAX_EDGES_PER_ENTITY),
            text_summaries_top_k,
            use_importance_weight: use_importance_weight.unwrap_or(true),
            use_truth_weight: use_truth_weight.unwrap_or(false),
            include_global_context_index: include_global_context_index.unwrap_or(false),
            global_context_index_top_k: global_context_index_top_k
                .unwrap_or(DEFAULT_GLOBAL_CONTEXT_INDEX_TOP_K),
            node_name,
            node_name_filter_operator: node_name_filter_operator
                .unwrap_or_else(|| DEFAULT_NODE_NAME_FILTER_OPERATOR.to_string()),
            system_prompt,
            system_prompt_path,
            user_prompt_template,
            generation_options,
        }
    }

    /// Entity/facts lane: the Rust equivalent of Python's
    /// `_retrieve_entities_and_facts` (`hybrid_retriever.py:167-198`) plus
    /// `_select_facts` (`:200-211`). Runs the `Entity_name` search (node filter
    /// applied) and the `EdgeType_relationship_name` search (node filter NOT
    /// applied) concurrently, builds ranked entity edge bullets, then gates facts
    /// off for scoped searches.
    #[allow(clippy::too_many_arguments)]
    async fn retrieve_entities_and_facts(
        &self,
        query_vector: &[f32],
        entities_top_k: usize,
        facts_top_k: usize,
        max_edges_per_entity: usize,
        node_name: Option<&[String]>,
        node_name_filter_operator: &str,
    ) -> Result<(Vec<EntityResult>, Vec<FactResult>), SearchError> {
        let edge_limit = entities_top_k
            .saturating_mul(max_edges_per_entity)
            .saturating_add(facts_top_k);

        let entity_future = search_collection(
            &self.vector_db,
            ENTITY_DATA_TYPE,
            ENTITY_FIELD,
            query_vector,
            entities_top_k,
            node_name,
            node_name_filter_operator,
            false,
        );
        // Python passes `apply_node_filter=False` for the edge lane
        // (`hybrid_retriever.py:188`); the Rust equivalent is `node_name=None`.
        let edge_future = search_collection(
            &self.vector_db,
            EDGE_TYPE_DATA_TYPE,
            EDGE_TYPE_FIELD,
            query_vector,
            edge_limit,
            None,
            node_name_filter_operator,
            false,
        );

        let (entity_hits, edge_hits) = tokio::try_join!(entity_future, edge_future)?;

        let edge_ranks = edge_rank_by_id(&edge_hits);
        let entities = build_entities(
            self.graph_db.as_ref(),
            &entity_hits,
            max_edges_per_entity,
            &edge_ranks,
        )
        .await;

        // Facts are gated off for scoped searches: EdgeType rows carry no
        // node-set fields (Python `_select_facts`, `hybrid_retriever.py:200-211`).
        let facts = if facts_top_k == 0 || node_name.is_some_and(|names| !names.is_empty()) {
            Vec::new()
        } else {
            let exclude_ids: HashSet<String> = entities
                .iter()
                .flat_map(|entity| entity.edges.iter())
                .filter_map(|edge| edge.edge_type_id.clone())
                .collect();
            select_facts(&edge_hits, &exclude_ids, facts_top_k)
        };

        Ok((entities, facts))
    }
}

/// Tag a chunk lane [`SearchItem`] with `"kind": "chunk"` and carry its paired
/// summary onto the item's own payload so `get_completion` can reconstruct
/// passages without the side-channel `chunk_summaries` map.
fn tag_chunk_item(mut item: SearchItem, chunk_summaries: &HashMap<String, String>) -> SearchItem {
    let summary = result_id(&item).and_then(|id| chunk_summaries.get(&id).cloned());
    if let Value::Object(map) = &mut item.payload {
        map.insert("kind".to_string(), Value::String("chunk".to_string()));
        if let Some(summary) = summary {
            map.insert("chunk_summary".to_string(), Value::String(summary));
        }
    }
    item
}

/// Convert an [`EntityResult`] into a `"kind": "entity"` tagged [`SearchItem`],
/// nesting the edge bullets as a JSON array so downstream consumers can walk
/// them without a second schema.
fn entity_to_item(entity: &EntityResult) -> SearchItem {
    let edges: Vec<Value> = entity
        .edges
        .iter()
        .map(|edge| {
            json!({
                "text": edge.text,
                "source": edge.source,
                "target": edge.target,
                "source_id": edge.source_id,
                "relationship": edge.relationship,
                "target_id": edge.target_id,
                "edge_type_id": edge.edge_type_id,
            })
        })
        .collect();

    SearchItem {
        id: Uuid::parse_str(&entity.id).ok(),
        score: None,
        payload: json!({
            "kind": "entity",
            "id": entity.id,
            "name": entity.name,
            "description": entity.description,
            "type": entity.entity_type,
            "edges": edges,
        }),
    }
}

/// Convert a [`FactResult`] into a `"kind": "fact"` tagged [`SearchItem`]. No
/// `source_id`/`target_id`/`edges` — the field-level contract that lets a
/// kind-aware id extractor skip facts.
fn fact_to_item(fact: &FactResult) -> SearchItem {
    SearchItem {
        id: Uuid::parse_str(&fact.id).ok(),
        score: None,
        payload: json!({
            "kind": "fact",
            "id": fact.id,
            "text": fact.text,
        }),
    }
}

fn payload_str(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn payload_str_opt(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Reconstruct an [`EntityResult`] from a `"kind": "entity"` tagged item.
fn item_to_entity(item: &SearchItem) -> EntityResult {
    let payload = &item.payload;
    let edges = payload
        .get("edges")
        .and_then(Value::as_array)
        .map(|edges| {
            edges
                .iter()
                .map(|edge| EdgeBullet {
                    text: payload_str(edge, "text"),
                    source: payload_str_opt(edge, "source"),
                    target: payload_str_opt(edge, "target"),
                    source_id: payload_str_opt(edge, "source_id"),
                    relationship: payload_str_opt(edge, "relationship"),
                    target_id: payload_str_opt(edge, "target_id"),
                    edge_type_id: payload_str_opt(edge, "edge_type_id"),
                })
                .collect()
        })
        .unwrap_or_default();

    EntityResult {
        id: payload_str(payload, "id"),
        name: payload_str(payload, "name"),
        description: payload_str_opt(payload, "description"),
        entity_type: payload_str_opt(payload, "type"),
        edges,
    }
}

/// Reconstruct a [`FactResult`] from a `"kind": "fact"` tagged item.
fn item_to_fact(item: &SearchItem) -> FactResult {
    FactResult {
        id: payload_str(&item.payload, "id"),
        text: payload_str(&item.payload, "text"),
    }
}

#[async_trait]
impl SearchRetriever for HybridRetriever {
    fn search_type(&self) -> SearchType {
        SearchType::HybridCompletion
    }

    async fn get_context(
        &self,
        query: &str,
        params: &SearchParams,
    ) -> Result<SearchContext, SearchError> {
        // Three-layer resolution for the top-k family: request knob -> top_k ->
        // constructor default.
        let chunks_top_k = params
            .chunks_top_k
            .or(params.top_k)
            .unwrap_or(self.chunks_top_k);
        let entities_top_k = params
            .entities_top_k
            .or(params.top_k)
            .unwrap_or(self.entities_top_k);
        let facts_top_k = params
            .facts_top_k
            .or(params.top_k)
            .unwrap_or(self.facts_top_k);
        let max_edges_per_entity = params
            .max_edges_per_entity
            .unwrap_or(self.max_edges_per_entity);
        let text_summaries_top_k = params.text_summaries_top_k.or(self.text_summaries_top_k);
        let use_importance_weight = params
            .use_importance_weight
            .unwrap_or(self.use_importance_weight);
        let node_name = params.node_name.as_deref().or(self.node_name.as_deref());
        let node_name_filter_operator = params
            .node_name_filter_operator
            .as_deref()
            .unwrap_or(self.node_name_filter_operator.as_str());

        if params
            .include_global_context_index
            .unwrap_or(self.include_global_context_index)
        {
            debug!(
                "HYBRID_COMPLETION: include_global_context_index is set but unsupported in \
                 Phase 1; no global-context section will be produced"
            );
        }
        if params.use_truth_weight.unwrap_or(self.use_truth_weight) {
            debug!(
                "HYBRID_COMPLETION: use_truth_weight is set but inert in Phase 1; baseline \
                 ranking is used"
            );
        }

        // Embed the query once and share the vector across both lanes.
        let embeddings = self.embedding_engine.embed(&[query]).await?;
        let query_vector = embeddings.into_iter().next().ok_or_else(|| {
            SearchError::InvalidInput("embedding engine returned no vectors".to_string())
        })?;

        let (chunk_result, (entities, facts)) = tokio::try_join!(
            retrieve_hybrid_chunks(
                &self.vector_db,
                &self.graph_db,
                query,
                chunks_top_k,
                text_summaries_top_k,
                node_name,
                node_name_filter_operator,
                use_importance_weight,
                &query_vector,
            ),
            self.retrieve_entities_and_facts(
                &query_vector,
                entities_top_k,
                facts_top_k,
                max_edges_per_entity,
                node_name,
                node_name_filter_operator,
            ),
        )?;

        let HybridChunksResult {
            chunks,
            chunk_summaries,
        } = chunk_result;

        let mut context: SearchContext =
            Vec::with_capacity(chunks.len() + entities.len() + facts.len());
        for chunk in chunks {
            context.push(tag_chunk_item(chunk, &chunk_summaries));
        }
        for entity in &entities {
            context.push(entity_to_item(entity));
        }
        for fact in &facts {
            context.push(fact_to_item(fact));
        }
        Ok(context)
    }

    async fn get_completion(
        &self,
        query: &str,
        context: Option<SearchContext>,
        session: &SessionContext,
        params: &SearchParams,
    ) -> Result<SearchOutput, SearchError> {
        let completion_context = match context {
            Some(existing_context) => existing_context,
            None => self.get_context(query, params).await?,
        };

        // Partition the flat context back into the three lanes by "kind".
        let mut chunks: Vec<SearchItem> = Vec::new();
        let mut entities: Vec<EntityResult> = Vec::new();
        let mut facts: Vec<FactResult> = Vec::new();
        for item in &completion_context {
            match item.payload.get("kind").and_then(Value::as_str) {
                Some("entity") => entities.push(item_to_entity(item)),
                Some("fact") => facts.push(item_to_fact(item)),
                Some("chunk") => chunks.push(item.clone()),
                _ => {}
            }
        }

        // Reconstruct the chunk-summary map from the chunk items' own payloads.
        let chunk_summaries: HashMap<String, String> = chunks
            .iter()
            .filter_map(|item| {
                let id = result_id(item)?;
                let summary = item.payload.get("chunk_summary").and_then(Value::as_str)?;
                Some((id, summary.to_string()))
            })
            .collect();

        let passages = format_passages(&chunks, &chunk_summaries);
        let entities_section = format_entities(&entities);
        let facts_section = format_facts(&facts);
        let context_text =
            format_hybrid_context(None, &passages, &entities_section, &facts_section);

        let system_prompt = resolve_system_prompt(
            params
                .system_prompt
                .as_deref()
                .or(self.system_prompt.as_deref()),
            params
                .system_prompt_path
                .as_deref()
                .or(self.system_prompt_path.as_deref()),
        )?;

        let user_prompt = render_user_prompt(
            Some(
                self.user_prompt_template
                    .as_deref()
                    .unwrap_or(DEFAULT_HYBRID_USER_PROMPT_TEMPLATE),
            ),
            query,
            &context_text,
        );

        debug!(
            context_items = completion_context.len(),
            "Hybrid context assembled:\n{context_text}"
        );
        debug!("LLM user prompt:\n{user_prompt}");

        let messages = build_messages_with_history(system_prompt, user_prompt, session);

        if let Some(schema) = &params.response_schema {
            let structured_value = self
                .llm
                .create_structured_output_with_messages_raw(
                    messages,
                    schema,
                    self.generation_options.clone(),
                )
                .await
                .map_err(|e| SearchError::LlmError(e.to_string()))?;
            Ok(SearchOutput::Structured(structured_value))
        } else {
            let completion = self
                .llm
                .generate(messages, self.generation_options.clone())
                .await?;
            Ok(SearchOutput::Text(completion.content))
        }
    }

    /// HYBRID_COMPLETION rejects query batches at every entry point (Python
    /// `_reject_query_batch`, `hybrid_retriever.py:300-302`). No default
    /// per-query-loop fallback.
    async fn get_context_batch(
        &self,
        _queries: &[String],
        _params: &SearchParams,
    ) -> Result<Vec<SearchContext>, SearchError> {
        Err(SearchError::InvalidInput(
            "HYBRID_COMPLETION does not support query_batch.".to_string(),
        ))
    }

    async fn get_completion_batch(
        &self,
        _queries: &[String],
        _contexts: Option<Vec<SearchContext>>,
        _session: &SessionContext,
        _params: &SearchParams,
    ) -> Result<Vec<SearchOutput>, SearchError> {
        Err(SearchError::InvalidInput(
            "HYBRID_COMPLETION does not support query_batch.".to_string(),
        ))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod retriever_tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use cognee_embedding::EmbeddingResult;
    use cognee_embedding::engine::EmbeddingEngine;
    use cognee_graph::{GraphDBTrait, MockGraphDB};
    use cognee_llm::{GenerationOptions, GenerationResponse, Llm, LlmResult, Message, TokenUsage};
    use cognee_models::EdgeType;
    use cognee_vector::{MockVectorDB, VectorDB, VectorPoint};
    use serde_json::{Value, json};
    use uuid::Uuid;

    use cognee_session::SessionContext;

    use super::HybridRetriever;
    use crate::retrievers::SearchRetriever;
    use crate::types::{SearchContext, SearchError, SearchItem, SearchOutput, SearchParams};
    use crate::utils::DEFAULT_RAG_SYSTEM_PROMPT;

    const CHUNK_TEXT: &str = "the rust ownership model chunk";
    const ENTITY_NAME: &str = "Alice";
    const BULLET_TEXT: &str = "Alice works at Acme.";
    const FACT_TEXT: &str = "Acme acquired Initech.";

    struct AlignedEmbedding;

    #[async_trait]
    impl EmbeddingEngine for AlignedEmbedding {
        async fn embed(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
        fn dimension(&self) -> usize {
            2
        }
        fn batch_size(&self) -> usize {
            8
        }
        fn max_sequence_length(&self) -> usize {
            128
        }
    }

    #[derive(Default)]
    struct CapturingLlm {
        last_messages: Mutex<Vec<Message>>,
        response_text: String,
        structured_response: Option<Value>,
    }

    #[async_trait]
    impl Llm for CapturingLlm {
        async fn generate(
            &self,
            messages: Vec<Message>,
            _options: Option<GenerationOptions>,
        ) -> LlmResult<GenerationResponse> {
            self.last_messages.lock().unwrap().clone_from(&messages);
            Ok(GenerationResponse {
                content: self.response_text.clone(),
                model: "test-model".to_string(),
                usage: Some(TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                }),
                finish_reason: Some("stop".to_string()),
            })
        }

        async fn create_structured_output_with_messages_raw(
            &self,
            messages: Vec<Message>,
            _json_schema: &Value,
            _options: Option<GenerationOptions>,
        ) -> LlmResult<Value> {
            self.last_messages.lock().unwrap().clone_from(&messages);
            Ok(self
                .structured_response
                .clone()
                .unwrap_or_else(|| json!({ "ok": true })))
        }

        fn model(&self) -> &str {
            "test-model"
        }
    }

    async fn edge_props(edge_text: &str) -> HashMap<Cow<'static, str>, Value> {
        let mut props: HashMap<Cow<'static, str>, Value> = HashMap::new();
        props.insert(Cow::from("edge_text"), json!(edge_text));
        props
    }

    /// Build vector + graph DBs holding one chunk, one entity (Alice with a
    /// works_at edge to Acme), and two EdgeType rows (one that becomes an entity
    /// bullet, one that survives as a standalone fact).
    async fn populated_dbs() -> (Arc<dyn VectorDB>, Arc<dyn GraphDBTrait>, Uuid) {
        let db = MockVectorDB::new();

        // DocumentChunk_text lane.
        db.create_collection("DocumentChunk", "text", 2)
            .await
            .unwrap();
        let chunk_id = Uuid::new_v4();
        db.index_points(
            "DocumentChunk",
            "text",
            &[VectorPoint::new(chunk_id, vec![1.0, 0.0])
                .with_metadata("id", json!(chunk_id.to_string()))
                .with_metadata("text", json!(CHUNK_TEXT))],
        )
        .await
        .unwrap();

        // Entity_name lane.
        db.create_collection("Entity", "name", 2).await.unwrap();
        let alice_id = Uuid::new_v4();
        db.index_points(
            "Entity",
            "name",
            &[VectorPoint::new(alice_id, vec![1.0, 0.0])
                .with_metadata("id", json!(alice_id.to_string()))
                .with_metadata("name", json!(ENTITY_NAME))],
        )
        .await
        .unwrap();

        // EdgeType_relationship_name lane: two rows.
        db.create_collection("EdgeType", "relationship_name", 2)
            .await
            .unwrap();
        let bullet_edge_id = EdgeType::deterministic_id(BULLET_TEXT);
        let fact_edge_id = EdgeType::deterministic_id(FACT_TEXT);
        db.index_points(
            "EdgeType",
            "relationship_name",
            &[
                VectorPoint::new(bullet_edge_id, vec![1.0, 0.0])
                    .with_metadata("id", json!(bullet_edge_id.to_string()))
                    .with_metadata("text", json!(BULLET_TEXT)),
                VectorPoint::new(fact_edge_id, vec![1.0, 0.0])
                    .with_metadata("id", json!(fact_edge_id.to_string()))
                    .with_metadata("text", json!(FACT_TEXT)),
            ],
        )
        .await
        .unwrap();

        // Graph: Alice --works_at--> Acme, with the bullet edge_text.
        let graph = MockGraphDB::new();
        graph
            .add_node_raw(json!({ "id": alice_id.to_string(), "name": ENTITY_NAME }))
            .await
            .unwrap();
        graph
            .add_node_raw(json!({ "id": "acme-id", "name": "Acme" }))
            .await
            .unwrap();
        graph
            .add_edge(
                &alice_id.to_string(),
                "acme-id",
                "works_at",
                Some(edge_props(BULLET_TEXT).await),
            )
            .await
            .unwrap();

        (Arc::new(db), Arc::new(graph), alice_id)
    }

    #[allow(clippy::too_many_arguments)]
    fn retriever(
        vector_db: Arc<dyn VectorDB>,
        graph_db: Arc<dyn GraphDBTrait>,
        llm: Arc<dyn Llm>,
    ) -> HybridRetriever {
        HybridRetriever::new(
            vector_db,
            Arc::new(AlignedEmbedding),
            graph_db,
            llm,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn kinds(context: &SearchContext) -> Vec<String> {
        context
            .iter()
            .filter_map(|item| {
                item.payload
                    .get("kind")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    #[tokio::test]
    async fn get_context_returns_tagged_chunk_entity_and_fact_items() {
        let (vector_db, graph_db, _alice) = populated_dbs().await;
        let llm = Arc::new(CapturingLlm::default());
        let retriever = retriever(vector_db, graph_db, llm);

        let context = retriever
            .get_context("query", &SearchParams::default())
            .await
            .unwrap();

        assert_eq!(kinds(&context), vec!["chunk", "entity", "fact"]);

        let chunk = &context[0];
        assert_eq!(chunk.payload["text"], json!(CHUNK_TEXT));

        let entity = &context[1];
        assert_eq!(entity.payload["name"], json!(ENTITY_NAME));
        assert!(entity.payload["edges"].is_array());

        let fact = &context[2];
        assert_eq!(fact.payload["text"], json!(FACT_TEXT));
    }

    #[tokio::test]
    async fn facts_are_gated_off_when_node_name_is_set() {
        let (vector_db, graph_db, _alice) = populated_dbs().await;
        let llm = Arc::new(CapturingLlm::default());
        let retriever = retriever(vector_db, graph_db, llm);

        let params = SearchParams {
            node_name: Some(vec!["some-set".to_string()]),
            ..SearchParams::default()
        };
        let context = retriever.get_context("query", &params).await.unwrap();

        // No item is tagged as a fact when the search is node-scoped.
        assert!(!kinds(&context).iter().any(|kind| kind == "fact"));
    }

    #[tokio::test]
    async fn get_completion_renders_all_section_headers() {
        let (vector_db, graph_db, _alice) = populated_dbs().await;
        let llm = Arc::new(CapturingLlm {
            response_text: "hybrid answer".to_string(),
            ..Default::default()
        });
        let retriever = retriever(vector_db, graph_db, Arc::clone(&llm) as Arc<dyn Llm>);

        let output = retriever
            .get_completion(
                "what happened?",
                None,
                &SessionContext::default(),
                &SearchParams::default(),
            )
            .await
            .unwrap();

        match output {
            SearchOutput::Text(text) => assert_eq!(text, "hybrid answer"),
            _ => panic!("expected text output"),
        }

        let messages = llm.last_messages.lock().unwrap().clone();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, DEFAULT_RAG_SYSTEM_PROMPT);
        let user = &messages[1].content;
        assert!(user.contains("what happened?"));
        assert!(user.contains("## Relevant passages"));
        assert!(user.contains("## Relevant entities"));
        assert!(user.contains("## Related facts"));
        assert!(user.contains(CHUNK_TEXT));
        assert!(user.contains(FACT_TEXT));
    }

    #[tokio::test]
    async fn get_completion_reuses_provided_context_without_running_lanes() {
        // Empty vector DB: if the lanes ran, the chunk lane would raise NotFound
        // on the missing DocumentChunk_text collection.
        let vector_db: Arc<dyn VectorDB> = Arc::new(MockVectorDB::new());
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());
        let llm = Arc::new(CapturingLlm {
            response_text: "reused".to_string(),
            ..Default::default()
        });
        let retriever = retriever(vector_db, graph_db, Arc::clone(&llm) as Arc<dyn Llm>);

        let provided: SearchContext = vec![
            SearchItem {
                id: None,
                score: None,
                payload: json!({ "kind": "chunk", "id": "c1", "text": "provided chunk" }),
            },
            SearchItem {
                id: None,
                score: None,
                payload: json!({
                    "kind": "entity",
                    "id": "e1",
                    "name": "Bob",
                    "description": Value::Null,
                    "type": Value::Null,
                    "edges": []
                }),
            },
        ];

        let output = retriever
            .get_completion(
                "who?",
                Some(provided),
                &SessionContext::default(),
                &SearchParams::default(),
            )
            .await
            .unwrap();

        match output {
            SearchOutput::Text(text) => assert_eq!(text, "reused"),
            _ => panic!("expected text output"),
        }

        let messages = llm.last_messages.lock().unwrap().clone();
        let user = &messages[1].content;
        assert!(user.contains("provided chunk"));
        assert!(user.contains("### Bob"));
    }

    #[tokio::test]
    async fn response_schema_selects_structured_output() {
        let (vector_db, graph_db, _alice) = populated_dbs().await;
        let llm = Arc::new(CapturingLlm {
            structured_response: Some(json!({ "answer": "structured" })),
            ..Default::default()
        });
        let retriever = retriever(vector_db, graph_db, llm);

        let params = SearchParams {
            response_schema: Some(json!({ "type": "object" })),
            ..SearchParams::default()
        };
        let output = retriever
            .get_completion("q", None, &SessionContext::default(), &params)
            .await
            .unwrap();

        match output {
            SearchOutput::Structured(value) => {
                assert_eq!(value, json!({ "answer": "structured" }));
            }
            _ => panic!("expected structured output"),
        }
    }

    #[tokio::test]
    async fn respects_request_level_top_k_override() {
        let (vector_db, graph_db, _alice) = populated_dbs().await;

        // Index extra DocumentChunk_text rows (aligned with the query vector) so
        // several chunks match. With the default limit (15) every one survives
        // ranking; a request-level chunks_top_k of 1 must truncate the effective
        // limit down to a single chunk — that difference is what proves the
        // override is honored rather than silently ignored.
        for i in 0..3 {
            let extra = Uuid::new_v4();
            vector_db
                .index_points(
                    "DocumentChunk",
                    "text",
                    &[VectorPoint::new(extra, vec![1.0, 0.0])
                        .with_metadata("id", json!(extra.to_string()))
                        .with_metadata("text", json!(format!("{CHUNK_TEXT} {i}")))],
                )
                .await
                .unwrap();
        }

        let llm = Arc::new(CapturingLlm::default());
        let retriever = retriever(vector_db, graph_db, llm);

        // Baseline: without an override the default limit keeps all matching
        // chunks, so more than one chunk comes back.
        let default_context = retriever
            .get_context("query", &SearchParams::default())
            .await
            .unwrap();
        let default_chunks = kinds(&default_context)
            .iter()
            .filter(|k| *k == "chunk")
            .count();
        assert!(
            default_chunks > 1,
            "default limit should return every matching chunk, got {default_chunks}"
        );

        // A per-request chunks_top_k of 1 truncates the effective limit to a
        // single chunk.
        let params = SearchParams {
            chunks_top_k: Some(1),
            ..SearchParams::default()
        };
        let context = retriever.get_context("query", &params).await.unwrap();
        let chunk_count = kinds(&context).iter().filter(|k| *k == "chunk").count();
        assert_eq!(chunk_count, 1);
    }

    #[tokio::test]
    async fn batch_methods_reject_with_invalid_input() {
        let vector_db: Arc<dyn VectorDB> = Arc::new(MockVectorDB::new());
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());
        let llm = Arc::new(CapturingLlm::default());
        let retriever = retriever(vector_db, graph_db, llm);

        let queries = vec!["a".to_string(), "b".to_string()];
        let params = SearchParams::default();

        match retriever.get_context_batch(&queries, &params).await {
            Err(SearchError::InvalidInput(message)) => {
                assert_eq!(message, "HYBRID_COMPLETION does not support query_batch.");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }

        match retriever
            .get_completion_batch(&queries, None, &SessionContext::default(), &params)
            .await
        {
            Err(SearchError::InvalidInput(message)) => {
                assert_eq!(message, "HYBRID_COMPLETION does not support query_batch.");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}
