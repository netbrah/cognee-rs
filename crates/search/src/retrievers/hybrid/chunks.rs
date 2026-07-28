//! Three-lane orchestration for the hybrid chunk retriever.
//!
//! Port of `cognee/modules/retrieval/hybrid/chunks.py` (Phase-1 subset). Fans a
//! query out to the BM25 lexical lane
//! ([`crate::retrievers::bm25_scored_chunks`]) and the vector
//! `DocumentChunk_text` / `TextSummary_text` lanes, merges them into
//! chunk↔summary pairs, ranks with RRF, backfills source chunks and summary
//! text, and returns the top chunks plus a `chunk_id -> summary_text` map.
//!
//! This task **receives** an already-computed `query_vector` (the future
//! retriever task, P1-09, embeds the query once and passes it down), so there
//! is no embedding call here; the Phase-2 truth-subspace parameters are
//! intentionally absent (locked deferral).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cognee_graph::{GraphDBTrait, NodeTruthState};
use cognee_vector::VectorDB;
use serde_json::Value;
use uuid::Uuid;

use crate::retrievers::context_items::search_results_to_context;
use crate::retrievers::hybrid::pairs::{
    ChunkSummaryPair, attach_source_chunks, chunk_summary_pairs, source_chunk_ids_to_load,
    summary_id_for_chunk, summary_text_by_chunk_id,
};
use crate::retrievers::hybrid::ranking::rank_chunk_summary_pairs;
use crate::retrievers::hybrid::results::{
    display_value, payload_matches_node_filter, result_id, scored_payload,
};
use crate::types::{SearchError, SearchItem};

const DOCUMENT_CHUNK_TYPE: &str = "DocumentChunk";
const TEXT_SUMMARY_TYPE: &str = "TextSummary";
const TEXT_FIELD: &str = "text";

/// Upper bound on the client-side node-filter over-fetch window in
/// [`search_collection`].
///
/// When a `node_name` filter is active, `search_collection` fetches up to the
/// full collection so that limit-then-filter matches Python's server-side
/// filter-then-limit exactly. This cap keeps that widening bounded: collections
/// larger than the cap fall back to the heuristic window and retain a residual
/// recall gap (documented on [`search_collection`]). Sized to comfortably cover
/// this project's edge-device / modest-dataset target while never fetching an
/// unbounded number of rows.
const NODE_FILTER_RECALL_FETCH_CAP: usize = 4096;

/// Result of a hybrid chunk retrieval: the ranked chunks and their paired
/// summaries keyed by chunk id.
pub(crate) struct HybridChunksResult {
    pub chunks: Vec<SearchItem>,
    pub chunk_summaries: HashMap<String, String>,
}

/// Candidate limit for the `TextSummary_text` lane.
///
/// Port of `summary_candidate_limit` (`chunks.py:106-109`). Python's `None`
/// branch clamps to `max(0, chunks_top_k)`; Rust's `usize` makes that a no-op,
/// so this is just `text_summaries_top_k.unwrap_or(chunks_top_k)`.
pub(crate) fn summary_candidate_limit(
    chunks_top_k: usize,
    text_summaries_top_k: Option<usize>,
) -> usize {
    text_summaries_top_k.unwrap_or(chunks_top_k)
}

/// Similarity-search a vector collection, guarding a missing collection and
/// applying the client-side node filter (Python filters server-side).
///
/// Port of `search_collection` (`chunks.py:143-175`) adapted to Rust's
/// `VectorDB` trait, which has no server-side `node_name` parameter. A missing
/// collection is a hard [`SearchError::NotFound`] when `required` (Python raises
/// `NoDataError`; Rust reuses `NotFound`, matching `ChunksRetriever` /
/// `SummariesRetriever`), otherwise an empty channel.
///
/// # Node-filter recall bound (divergence from Python)
///
/// Python threads `node_name` into `vector_engine.search(...)`, which filters
/// **server-side then limits** (`filter-then-limit`): the engine only counts
/// in-set rows toward `limit`, so every returned row is in-set and no valid
/// in-set row is ever crowded out by out-of-set rows. `VectorDB::search_similar`
/// has no filter parameter, so this port must **limit-then-filter**: fetch a
/// window ordered by pure similarity, then drop out-of-set rows and truncate to
/// `limit`.
///
/// That inverts the ordering, so it has a **recall bound**: an in-set row is
/// silently dropped when strictly more than `fetch_limit - limit` out-of-set
/// rows outrank it by similarity (they exhaust the window before it is reached).
///
/// To keep that gap out of the realistic case, `fetch_limit` widens to the
/// **entire collection** whenever the collection fits under
/// [`NODE_FILTER_RECALL_FETCH_CAP`]. Fetching every row makes limit-then-filter
/// *exactly* equal to filter-then-limit (the window can no longer be exhausted),
/// so parity is exact for any collection at or below the cap — which covers this
/// project's edge-device / modest-dataset target. Only collections larger than
/// the cap fall back to the bounded heuristic window and retain the residual
/// recall gap above. The bound is intentional: the true fix (a server-side
/// `node_name` filter on `VectorDB::search_similar`) is a larger, out-of-scope
/// trait change, tracked as an accepted divergence in the Phase-2 plan
/// (`docs/plans/hybrid-retriever/phase-2/P2-05.md`).
///
/// The fetch never grows without bound: it is capped at
/// [`NODE_FILTER_RECALL_FETCH_CAP`].
#[allow(clippy::too_many_arguments)]
pub(crate) async fn search_collection(
    vector_db: &Arc<dyn VectorDB>,
    data_type: &str,
    field: &str,
    query_vector: &[f32],
    limit: usize,
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
    required: bool,
) -> Result<Vec<SearchItem>, SearchError> {
    if limit == 0 {
        return Ok(vec![]);
    }

    if !vector_db.has_collection(data_type, field).await? {
        if required {
            return Err(SearchError::NotFound(format!(
                "missing vector collection: {data_type}_{field}"
            )));
        }
        return Ok(vec![]);
    }

    match node_name {
        Some(names) if !names.is_empty() => {
            // limit-then-filter (Python filters server-side). Widen the window
            // to the whole collection when it fits under the cap so that
            // dropping out-of-set rows and truncating to `limit` reproduces
            // Python's filter-then-limit exactly; only collections above the
            // cap keep the bounded heuristic window and its residual recall
            // gap. See the fn doc comment for the full recall-bound rationale.
            let heuristic_window = limit.saturating_mul(4).max(limit + 20);
            let collection_size = vector_db.collection_size(data_type, field).await?;
            let fetch_limit =
                heuristic_window.max(collection_size.min(NODE_FILTER_RECALL_FETCH_CAP));
            let results = vector_db
                .search_similar(data_type, field, query_vector, fetch_limit)
                .await?;
            let items = search_results_to_context(results)?;
            Ok(items
                .into_iter()
                .filter(|item| {
                    payload_matches_node_filter(
                        &item.payload,
                        Some(names),
                        node_name_filter_operator,
                    )
                })
                .take(limit)
                .collect())
        }
        _ => {
            let results = vector_db
                .search_similar(data_type, field, query_vector, limit)
                .await?;
            search_results_to_context(results)
        }
    }
}

/// Score the BM25 lexical lane and shape it into node-filtered [`SearchItem`]s.
///
/// Port of `search_bm25_chunks` (`chunks.py:112-140`). The BM25 lane already
/// landed as the free function [`crate::retrievers::bm25_scored_chunks`], which
/// handles `limit == 0`, the fresh-per-call corpus rebuild, truncate-then-drop
/// ordering, and fail-open behavior (never `Err`). This wrapper only shapes its
/// `(payload, score)` output into [`SearchItem`]s and applies the node filter.
async fn search_bm25_chunks(
    graph_db: &Arc<dyn GraphDBTrait>,
    query: &str,
    limit: usize,
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
) -> Vec<SearchItem> {
    let scored = crate::retrievers::bm25_scored_chunks(graph_db, query, limit).await;
    scored
        .into_iter()
        .filter_map(|item| {
            let (payload, score) = scored_payload(item);
            if score <= 0.0 {
                return None;
            }
            let id = payload
                .get("id")
                .and_then(Value::as_str)
                .and_then(|raw| Uuid::parse_str(raw).ok());
            Some(SearchItem {
                id,
                score: Some(score),
                payload,
            })
        })
        .filter(|item| {
            payload_matches_node_filter(&item.payload, node_name, node_name_filter_operator)
        })
        .collect()
}

/// Fetch `DocumentChunk_text` rows by id for summary-only pairs, dropping any
/// that fail the node filter.
///
/// Port of `load_source_chunks_for_summaries` (`chunks.py:178-209`) using the
/// batched retrieve-by-ids call.
async fn load_source_chunks_for_summaries(
    vector_db: &Arc<dyn VectorDB>,
    chunk_ids: &[String],
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
) -> Result<Vec<SearchItem>, SearchError> {
    let uuids: Vec<Uuid> = chunk_ids
        .iter()
        .filter_map(|id| Uuid::parse_str(id).ok())
        .collect();
    let results = vector_db
        .retrieve(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, &uuids)
        .await?;
    let items = search_results_to_context(results)?;

    let found: HashSet<String> = items.iter().filter_map(result_id).collect();
    let missing: Vec<&String> = chunk_ids.iter().filter(|id| !found.contains(*id)).collect();
    if !missing.is_empty() {
        tracing::warn!(
            ?missing,
            "TextSummary_text hit referenced missing DocumentChunk_text row(s)"
        );
    }

    let mut source_chunks = Vec::new();
    let mut filtered_ids = Vec::new();
    for item in items {
        if payload_matches_node_filter(&item.payload, node_name, node_name_filter_operator) {
            source_chunks.push(item);
        } else if let Some(id) = result_id(&item) {
            filtered_ids.push(id);
        }
    }
    if !filtered_ids.is_empty() {
        tracing::warn!(
            ?filtered_ids,
            "TextSummary_text source chunk failed node filter"
        );
    }
    Ok(source_chunks)
}

/// Backfill `summary_text` onto the ranked pairs by batch-fetching
/// `TextSummary_text` rows.
///
/// Port of `load_summary_text_for_ranked_pairs` (`chunks.py:212-280`). A missing
/// `TextSummary_text` collection is tolerated (`has_collection` guard, mirroring
/// Python's `CollectionNotFoundError` early-return).
async fn load_summary_text_for_ranked_pairs(
    vector_db: &Arc<dyn VectorDB>,
    ranked_pairs: &mut [ChunkSummaryPair],
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
) -> Result<(), SearchError> {
    let mut summary_uuids: Vec<Uuid> = Vec::new();
    for pair in ranked_pairs.iter_mut() {
        if pair.summary_text.is_some() {
            continue;
        }
        let Some(chunk_id) = pair.chunk_id.clone().filter(|id| !id.is_empty()) else {
            continue;
        };
        let Some(summary_id) = pair
            .summary_id
            .clone()
            .or_else(|| summary_id_for_chunk(&chunk_id))
        else {
            tracing::debug!(%chunk_id, "Cannot fetch paired TextSummary for non-UUID chunk id");
            continue;
        };
        if let Ok(uuid) = Uuid::parse_str(&summary_id) {
            summary_uuids.push(uuid);
        }
        pair.summary_id = Some(summary_id);
    }

    if summary_uuids.is_empty() {
        return Ok(());
    }

    if !vector_db
        .has_collection(TEXT_SUMMARY_TYPE, TEXT_FIELD)
        .await?
    {
        tracing::warn!("TextSummary_text collection missing while loading chunk summaries");
        return Ok(());
    }

    let results = vector_db
        .retrieve(TEXT_SUMMARY_TYPE, TEXT_FIELD, &summary_uuids)
        .await?;
    let items = search_results_to_context(results)?;
    let mut summaries_by_id: HashMap<String, SearchItem> = HashMap::new();
    for item in items {
        if let Some(id) = result_id(&item) {
            summaries_by_id.insert(id, item);
        }
    }

    for pair in ranked_pairs.iter_mut() {
        let (Some(chunk_id), Some(summary_id)) = (pair.chunk_id.clone(), pair.summary_id.clone())
        else {
            continue;
        };
        if chunk_id.is_empty() || summary_id.is_empty() {
            continue;
        }

        let Some(summary) = summaries_by_id.get(&summary_id) else {
            tracing::warn!(%chunk_id, "DocumentChunk_text row has no paired TextSummary_text row");
            continue;
        };

        let Some(summary_text) = summary.payload.get("text").and_then(display_value) else {
            tracing::warn!(%chunk_id, %summary_id, "Paired TextSummary_text row has no text");
            continue;
        };

        if !payload_matches_node_filter(&summary.payload, node_name, node_name_filter_operator) {
            tracing::warn!(%chunk_id, %summary_id, "Paired TextSummary_text row failed node filter");
            continue;
        }

        pair.summary_text = Some(summary_text);
    }

    Ok(())
}

/// Retrieve, merge, and rank hybrid chunks for `query`.
///
/// Port of `retrieve_hybrid_chunks` (`chunks.py:27-103`, Phase-1 subset). Runs
/// the three lanes concurrently (BM25 is infallible; the required
/// `DocumentChunk_text` vector lane propagates a `NotFound`; the summary lane is
/// optional), builds pairs, backfills source chunks for summary-only pairs,
/// ranks, backfills summary text on the **ranked** pairs, and returns the top
/// chunks plus their summaries.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn retrieve_hybrid_chunks(
    vector_db: &Arc<dyn VectorDB>,
    graph_db: &Arc<dyn GraphDBTrait>,
    query: &str,
    chunks_top_k: usize,
    text_summaries_top_k: Option<usize>,
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
    use_importance_weight: bool,
    query_vector: &[f32],
    use_truth_weight: bool,
    q_coords: Option<&[f64]>,
    truth_state_by_id: Option<&HashMap<String, NodeTruthState>>,
    current_truth_epoch: Option<i64>,
) -> Result<HybridChunksResult, SearchError> {
    let candidate_limit = chunks_top_k.saturating_mul(2);
    let summary_limit = summary_candidate_limit(chunks_top_k, text_summaries_top_k);

    let bm25_future = search_bm25_chunks(
        graph_db,
        query,
        candidate_limit,
        node_name,
        node_name_filter_operator,
    );
    let vector_future = search_collection(
        vector_db,
        DOCUMENT_CHUNK_TYPE,
        TEXT_FIELD,
        query_vector,
        candidate_limit,
        node_name,
        node_name_filter_operator,
        true,
    );
    let summary_future = search_collection(
        vector_db,
        TEXT_SUMMARY_TYPE,
        TEXT_FIELD,
        query_vector,
        summary_limit,
        node_name,
        node_name_filter_operator,
        false,
    );

    let (bm25_chunks, vector_result, summary_result) =
        tokio::join!(bm25_future, vector_future, summary_future);
    let vector_chunks = vector_result?;
    let summary_hits = summary_result?;

    let mut pairs = chunk_summary_pairs(
        &bm25_chunks,
        &vector_chunks,
        &summary_hits,
        node_name,
        node_name_filter_operator,
    );

    let missing_source_chunk_ids = source_chunk_ids_to_load(&pairs);
    if !missing_source_chunk_ids.is_empty() {
        let source_chunks = load_source_chunks_for_summaries(
            vector_db,
            &missing_source_chunk_ids,
            node_name,
            node_name_filter_operator,
        )
        .await?;
        attach_source_chunks(&mut pairs, &source_chunks);
    }

    let mut ranked_pairs = rank_chunk_summary_pairs(
        pairs,
        chunks_top_k,
        use_importance_weight,
        use_truth_weight,
        q_coords,
        truth_state_by_id,
        current_truth_epoch,
    );
    if summary_limit > 0 {
        load_summary_text_for_ranked_pairs(
            vector_db,
            &mut ranked_pairs,
            node_name,
            node_name_filter_operator,
        )
        .await?;
    }

    let chunk_summaries = summary_text_by_chunk_id(&ranked_pairs);
    let chunks = ranked_pairs
        .into_iter()
        .filter_map(|pair| pair.chunk)
        .collect();

    Ok(HybridChunksResult {
        chunks,
        chunk_summaries,
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use std::sync::Arc;

    use cognee_graph::{GraphDBTrait, GraphDBTraitExt, MockGraphDB};
    use cognee_vector::{MockVectorDB, VectorDB, VectorPoint};
    use serde::Serialize;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::types::SearchError;

    fn dyn_vector(db: MockVectorDB) -> Arc<dyn VectorDB> {
        Arc::new(db)
    }

    /// Index a DocumentChunk_text vector point.
    async fn index_chunk(
        db: &MockVectorDB,
        id: Uuid,
        text: &str,
        belongs_to_set: Option<Vec<&str>>,
        vector: Vec<f32>,
    ) {
        let mut point = VectorPoint::new(id, vector)
            .with_metadata("id", json!(id.to_string()))
            .with_metadata("text", json!(text));
        if let Some(sets) = belongs_to_set {
            point = point.with_metadata("belongs_to_set", json!(sets));
        }
        db.index_points(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, &[point])
            .await
            .unwrap();
    }

    /// Index a TextSummary_text vector point.
    async fn index_summary(
        db: &MockVectorDB,
        id: Uuid,
        text: &str,
        source_chunk_id: Uuid,
        vector: Vec<f32>,
    ) {
        let point = VectorPoint::new(id, vector)
            .with_metadata("id", json!(id.to_string()))
            .with_metadata("text", json!(text))
            .with_metadata("source_chunk_id", json!(source_chunk_id.to_string()));
        db.index_points(TEXT_SUMMARY_TYPE, TEXT_FIELD, &[point])
            .await
            .unwrap();
    }

    #[derive(Serialize)]
    struct ChunkNode {
        id: String,
        #[serde(rename = "type")]
        kind: String,
        text: String,
    }

    async fn add_graph_chunk(graph: &MockGraphDB, id: Uuid, text: &str) {
        let node = ChunkNode {
            id: id.to_string(),
            kind: DOCUMENT_CHUNK_TYPE.to_string(),
            text: text.to_string(),
        };
        graph.add_node(&node).await.unwrap();
    }

    #[tokio::test]
    async fn missing_document_chunk_collection_is_not_found() {
        let vector_db = dyn_vector(MockVectorDB::new());
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());

        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "query",
            3,
            None,
            None,
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await;

        assert!(matches!(result, Err(SearchError::NotFound(_))));
    }

    #[tokio::test]
    async fn missing_text_summary_collection_is_tolerated() {
        let db = MockVectorDB::new();
        db.create_collection(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();
        let chunk_id = Uuid::new_v4();
        index_chunk(&db, chunk_id, "rust ownership model", None, vec![1.0, 0.0]).await;
        let vector_db = dyn_vector(db);
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());

        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "query",
            3,
            None,
            None,
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.chunks.len(), 1);
        assert!(result.chunk_summaries.is_empty());
    }

    #[tokio::test]
    async fn node_name_filter_drops_non_matching_vector_hits() {
        let db = MockVectorDB::new();
        db.create_collection(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();
        let keep_a = Uuid::new_v4();
        let keep_b = Uuid::new_v4();
        let drop_c = Uuid::new_v4();
        index_chunk(&db, keep_a, "text a", Some(vec!["keep"]), vec![1.0, 0.0]).await;
        index_chunk(&db, keep_b, "text b", Some(vec!["keep"]), vec![1.0, 0.0]).await;
        index_chunk(&db, drop_c, "text c", Some(vec!["drop"]), vec![1.0, 0.0]).await;
        let vector_db = dyn_vector(db);
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());

        let node_name = vec!["keep".to_string()];
        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "query",
            3,
            None,
            Some(&node_name),
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.chunks.len(), 2);
        for chunk in &result.chunks {
            let sets = chunk.payload.get("belongs_to_set").unwrap();
            assert_eq!(sets, &json!(["keep"]));
        }
    }

    #[tokio::test]
    async fn node_filter_keeps_in_set_chunks_outranked_by_many_out_of_set() {
        // Regression for the client-side node-filter recall bound: 30 out-of-set
        // chunks outrank the 2 in-set chunks by pure similarity. With
        // chunks_top_k = 2 the candidate limit is 4 and the heuristic over-fetch
        // window is 24 — smaller than the 30 out-of-set rows — so a plain
        // limit-then-filter would exhaust the window on out-of-set rows and drop
        // both in-set chunks. Widening the fetch to the full collection (below
        // NODE_FILTER_RECALL_FETCH_CAP) restores filter-then-limit parity and
        // recovers both in-set chunks.
        let db = MockVectorDB::new();
        db.create_collection(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();

        // 30 out-of-set chunks, maximally aligned with the query [1, 0].
        for i in 0..30 {
            index_chunk(
                &db,
                Uuid::new_v4(),
                &format!("out of set {i}"),
                Some(vec!["drop"]),
                vec![1.0, 0.0],
            )
            .await;
        }
        // 2 in-set chunks, slightly less aligned so they rank strictly below
        // every out-of-set chunk.
        let keep_a = Uuid::new_v4();
        let keep_b = Uuid::new_v4();
        index_chunk(&db, keep_a, "in set a", Some(vec!["keep"]), vec![0.8, 0.6]).await;
        index_chunk(&db, keep_b, "in set b", Some(vec!["keep"]), vec![0.8, 0.6]).await;

        let vector_db = dyn_vector(db);
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());

        let node_name = vec!["keep".to_string()];
        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "query",
            2,
            None,
            Some(&node_name),
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.chunks.len(), 2, "both in-set chunks must survive");
        for chunk in &result.chunks {
            let sets = chunk.payload.get("belongs_to_set").unwrap();
            assert_eq!(sets, &json!(["keep"]));
        }
    }

    #[tokio::test]
    async fn summary_only_hit_triggers_source_chunk_backfill() {
        // chunks_top_k = 2 -> candidate_limit = 4. Five filler chunks aligned
        // with the query push the orthogonal source chunk C out of the vector
        // lane's top-4, so C only reaches the pipeline via the summary hit's
        // source_chunk_id and must be backfilled by retrieve-by-id.
        let db = MockVectorDB::new();
        db.create_collection(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();
        db.create_collection(TEXT_SUMMARY_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();

        for i in 0..5 {
            index_chunk(
                &db,
                Uuid::new_v4(),
                &format!("filler {i}"),
                None,
                vec![1.0, 0.0],
            )
            .await;
        }
        // Source chunk C: orthogonal to the query so it is not a direct hit.
        let chunk_c = Uuid::new_v4();
        index_chunk(
            &db,
            chunk_c,
            "backfilled source chunk",
            None,
            vec![0.0, 1.0],
        )
        .await;
        // Summary references C and is aligned with the query.
        let summary_id = Uuid::new_v5(&chunk_c, b"TextSummary");
        index_summary(
            &db,
            summary_id,
            "the paired summary",
            chunk_c,
            vec![1.0, 0.0],
        )
        .await;

        let vector_db = dyn_vector(db);
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());

        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "query",
            2,
            None,
            None,
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        // C was backfilled and survived ranking; its summary is attached.
        let chunk_c_str = chunk_c.to_string();
        assert!(
            result
                .chunks
                .iter()
                .any(|c| c.payload.get("id") == Some(&json!(chunk_c_str))),
            "backfilled source chunk should appear in ranked chunks"
        );
        assert_eq!(
            result.chunk_summaries.get(&chunk_c_str).map(String::as_str),
            Some("the paired summary")
        );
    }

    #[tokio::test]
    async fn chunk_summaries_reflect_ranked_pairs_only() {
        // chunks_top_k = 1 -> limit = 1: only the single top-ranked pair's
        // summary is reported, not every candidate summary.
        let db = MockVectorDB::new();
        db.create_collection(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();
        db.create_collection(TEXT_SUMMARY_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();

        // A strong direct chunk hit that will win the single slot.
        let winner = Uuid::new_v4();
        index_chunk(&db, winner, "winning chunk", None, vec![1.0, 0.0]).await;
        let winner_summary = Uuid::new_v5(&winner, b"TextSummary");
        index_summary(
            &db,
            winner_summary,
            "winner summary",
            winner,
            vec![1.0, 0.0],
        )
        .await;

        // A weaker summary-only candidate that should be truncated away.
        let loser = Uuid::new_v4();
        index_chunk(&db, loser, "loser chunk", None, vec![0.0, 1.0]).await;
        let loser_summary = Uuid::new_v5(&loser, b"TextSummary");
        index_summary(&db, loser_summary, "loser summary", loser, vec![0.2, 1.0]).await;

        let vector_db = dyn_vector(db);
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(MockGraphDB::new());

        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "query",
            1,
            None,
            None,
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunk_summaries.len(), 1);
        assert_eq!(
            result
                .chunk_summaries
                .get(&winner.to_string())
                .map(String::as_str),
            Some("winner summary")
        );
        assert!(!result.chunk_summaries.contains_key(&loser.to_string()));
    }

    #[tokio::test]
    async fn bm25_lane_contributes_graph_chunks() {
        // No vector DocumentChunk hits aligned with the query, but a graph
        // DocumentChunk node feeds the BM25 lane.
        let db = MockVectorDB::new();
        db.create_collection(DOCUMENT_CHUNK_TYPE, TEXT_FIELD, 2)
            .await
            .unwrap();
        let chunk_id = Uuid::new_v4();
        // Present in the vector collection so retrieve/search can see it, and in
        // the graph so BM25 scores it.
        index_chunk(
            &db,
            chunk_id,
            "ownership borrow checker",
            None,
            vec![1.0, 0.0],
        )
        .await;
        let vector_db = dyn_vector(db);

        let mock_graph = MockGraphDB::new();
        add_graph_chunk(&mock_graph, chunk_id, "ownership borrow checker").await;
        let graph_db: Arc<dyn GraphDBTrait> = Arc::new(mock_graph);

        let result = retrieve_hybrid_chunks(
            &vector_db,
            &graph_db,
            "ownership borrow",
            3,
            None,
            None,
            "OR",
            false,
            &[1.0, 0.0],
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert!(!result.chunks.is_empty());
        assert!(
            result
                .chunks
                .iter()
                .any(|c| c.payload.get("id") == Some(&json!(chunk_id.to_string())))
        );
    }
}
