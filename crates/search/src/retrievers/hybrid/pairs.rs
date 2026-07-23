//! Chunk↔summary pairing and merge.
//!
//! Port of `cognee/modules/retrieval/hybrid/pairs.py`. Python's ad-hoc dict is
//! replaced by the [`ChunkSummaryPair`] struct; the per-query candidate list is
//! tiny (`2 * chunks_top_k` at most), so the linear-scan matching stays O(n²)
//! with a negligible `n`.

use std::collections::HashMap;

use uuid::Uuid;

use crate::retrievers::hybrid::results::{display_value, payload_matches_node_filter, result_id};
use crate::types::SearchItem;

/// One candidate chunk together with the summary paired to it (if any) and the
/// per-lane ranks that fed the fusion.
#[derive(Debug, Clone, Default)]
pub(crate) struct ChunkSummaryPair {
    pub chunk_id: Option<String>,
    pub chunk_text: Option<String>,
    pub summary_id: Option<String>,
    pub summary_text: Option<String>,
    pub chunk: Option<SearchItem>,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub summary_rank: Option<usize>,
}

/// Build chunk↔summary pairs from the three lanes.
///
/// Port of `chunk_summary_pairs` (`pairs.py:15-65`): a two-phase loop over the
/// BM25 then vector chunks (first-rank-wins per lane, with id-adoption when a
/// text-merged id-less BM25 payload later matches a vector chunk), then a
/// summary loop keyed strictly by `source_chunk_id` (node-filtered; hits with
/// no `source_chunk_id` are dropped with a warning).
pub(crate) fn chunk_summary_pairs(
    bm25_chunks: &[SearchItem],
    vector_chunks: &[SearchItem],
    summary_hits: &[SearchItem],
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
) -> Vec<ChunkSummaryPair> {
    let mut pairs: Vec<ChunkSummaryPair> = Vec::new();

    for (is_bm25, chunks) in [(true, bm25_chunks), (false, vector_chunks)] {
        for (rank, chunk) in chunks.iter().enumerate() {
            let chunk_id = result_id(chunk);
            let chunk_text = chunk.payload.get("text").and_then(display_value);
            if chunk_id.is_none() && chunk_text.is_none() {
                continue;
            }

            let index =
                match find_chunk_summary_pair(&pairs, chunk_id.as_deref(), chunk_text.as_deref()) {
                    Some(index) => index,
                    None => {
                        pairs.push(new_chunk_summary_pair(chunk_id.clone(), chunk_text.clone()));
                        pairs.len() - 1
                    }
                };

            let pair = &mut pairs[index];
            if pair.chunk.is_none() {
                set_pair_chunk(pair, chunk);
            } else if pair.chunk_id.is_none() {
                // Text-merged onto an id-less chunk (e.g. BM25 payload without
                // id): adopt the id so summary hits can pair by source_chunk_id.
                if let Some(id) = chunk_id.clone() {
                    pair.chunk_id = Some(id);
                }
            }

            let rank_slot = if is_bm25 {
                &mut pair.bm25_rank
            } else {
                &mut pair.vector_rank
            };
            if rank_slot.is_none() {
                *rank_slot = Some(rank);
            }
        }
    }

    for (rank, summary) in summary_hits.iter().enumerate() {
        if !payload_matches_node_filter(&summary.payload, node_name, node_name_filter_operator) {
            continue;
        }

        let Some(chunk_id) = summary
            .payload
            .get("source_chunk_id")
            .and_then(display_value)
        else {
            tracing::warn!(
                summary_id = ?result_id(summary),
                "TextSummary_text hit has no source_chunk_id"
            );
            continue;
        };

        let index = match find_chunk_summary_pair(&pairs, Some(chunk_id.as_str()), None) {
            Some(index) => index,
            None => {
                pairs.push(new_chunk_summary_pair(Some(chunk_id.clone()), None));
                pairs.len() - 1
            }
        };

        let pair = &mut pairs[index];
        if pair.summary_rank.is_none() {
            pair.summary_rank = Some(rank);
            pair.summary_id = result_id(summary);
            pair.summary_text = summary.payload.get("text").and_then(display_value);
        }
    }

    pairs
}

/// Chunk ids that need a source-chunk backfill: pairs with a `summary_rank` but
/// no `chunk` yet and a non-empty `chunk_id`.
///
/// Port of `source_chunk_ids_to_load` (`pairs.py:68-73`).
pub(crate) fn source_chunk_ids_to_load(pairs: &[ChunkSummaryPair]) -> Vec<String> {
    pairs
        .iter()
        .filter(|pair| {
            pair.summary_rank.is_some()
                && pair.chunk.is_none()
                && pair.chunk_id.as_deref().is_some_and(|id| !id.is_empty())
        })
        .filter_map(|pair| pair.chunk_id.clone())
        .collect()
}

/// Attach backfilled source chunks onto their matching pairs by `result_id`.
///
/// Port of `attach_source_chunks` (`pairs.py:76-80`).
pub(crate) fn attach_source_chunks(pairs: &mut [ChunkSummaryPair], chunks: &[SearchItem]) {
    for chunk in chunks {
        let chunk_id = result_id(chunk);
        if let Some(index) = find_chunk_summary_pair(pairs, chunk_id.as_deref(), None) {
            set_pair_chunk(&mut pairs[index], chunk);
        }
    }
}

/// Map of `chunk_id -> summary_text` over the pairs that carry both.
///
/// Port of `summary_text_by_chunk_id` (`pairs.py:83-88`).
pub(crate) fn summary_text_by_chunk_id(pairs: &[ChunkSummaryPair]) -> HashMap<String, String> {
    let mut summaries = HashMap::new();
    for pair in pairs {
        if let (Some(chunk_id), Some(text)) = (pair.chunk_id.as_ref(), pair.summary_text.as_ref())
            && !chunk_id.is_empty()
            && !text.is_empty()
        {
            summaries.insert(chunk_id.clone(), text.clone());
        }
    }
    summaries
}

/// Deterministic `TextSummary` id for a `DocumentChunk` id.
///
/// Port of `summary_id_for_chunk` (`pairs.py:91-96`): parse `chunk_id` as a
/// UUID (`None` on failure), then `uuid5(chunk_uuid, "TextSummary")`. This
/// mirrors the scheme in `cognee-cognify`'s `TextSummary::new`
/// (`summarization/models.rs:73`) inline — no cross-crate dependency — so the
/// two never drift.
pub(crate) fn summary_id_for_chunk(chunk_id: &str) -> Option<String> {
    let chunk_uuid = Uuid::parse_str(chunk_id).ok()?;
    Some(Uuid::new_v5(&chunk_uuid, b"TextSummary").to_string())
}

/// Set a pair's chunk, adopting its id/text when present (keeping existing
/// values otherwise). Port of `set_pair_chunk` (`pairs.py:99-102`).
fn set_pair_chunk(pair: &mut ChunkSummaryPair, chunk: &SearchItem) {
    pair.chunk = Some(chunk.clone());
    if let Some(id) = result_id(chunk) {
        pair.chunk_id = Some(id);
    }
    if let Some(text) = chunk.payload.get("text").and_then(display_value) {
        pair.chunk_text = Some(text);
    }
}

/// Find the index of the pair matching `chunk_id` (exact) or, failing that, an
/// id-less pair with the same `chunk_text`. Port of `_find_chunk_summary_pair`
/// (`pairs.py:105-115`).
fn find_chunk_summary_pair(
    pairs: &[ChunkSummaryPair],
    chunk_id: Option<&str>,
    chunk_text: Option<&str>,
) -> Option<usize> {
    for (index, pair) in pairs.iter().enumerate() {
        if let Some(id) = chunk_id
            && pair.chunk_id.as_deref() == Some(id)
        {
            return Some(index);
        }
        if let Some(text) = chunk_text
            && pair.chunk_id.is_none()
            && pair.chunk_text.as_deref() == Some(text)
        {
            return Some(index);
        }
    }
    None
}

/// Fresh pair seeded with a chunk id/text. Port of `_new_chunk_summary_pair`
/// (`pairs.py:118-131`).
fn new_chunk_summary_pair(
    chunk_id: Option<String>,
    chunk_text: Option<String>,
) -> ChunkSummaryPair {
    ChunkSummaryPair {
        chunk_id,
        chunk_text,
        ..Default::default()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    fn item(payload: serde_json::Value) -> SearchItem {
        SearchItem {
            id: None,
            score: None,
            payload,
        }
    }

    #[test]
    fn bm25_and_vector_hit_for_same_id_merge() {
        let id = Uuid::new_v4().to_string();
        let bm25 = vec![item(json!({"id": id, "text": "hello world"}))];
        let vector = vec![item(json!({"id": id, "text": "hello world"}))];
        let pairs = chunk_summary_pairs(&bm25, &vector, &[], None, "OR");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].bm25_rank, Some(0));
        assert_eq!(pairs[0].vector_rank, Some(0));
        assert_eq!(pairs[0].chunk_id.as_deref(), Some(id.as_str()));
    }

    #[test]
    fn idless_bm25_hit_adopts_id_from_vector() {
        let id = Uuid::new_v4().to_string();
        // BM25 payload with no id, matched by text.
        let bm25 = vec![item(json!({"text": "shared text"}))];
        let vector = vec![item(json!({"id": id, "text": "shared text"}))];
        let pairs = chunk_summary_pairs(&bm25, &vector, &[], None, "OR");
        assert_eq!(pairs.len(), 1, "text-merge collapses into one pair");
        assert_eq!(pairs[0].bm25_rank, Some(0));
        assert_eq!(pairs[0].vector_rank, Some(0));
        assert_eq!(
            pairs[0].chunk_id.as_deref(),
            Some(id.as_str()),
            "id adopted from the vector chunk"
        );
    }

    #[test]
    fn summary_without_source_chunk_id_is_dropped() {
        let summaries = vec![item(json!({"id": "s1", "text": "a summary"}))];
        let pairs = chunk_summary_pairs(&[], &[], &summaries, None, "OR");
        assert!(pairs.is_empty());
    }

    #[test]
    fn summary_keyed_by_source_chunk_id() {
        let chunk_id = Uuid::new_v4().to_string();
        let summaries = vec![item(json!({
            "id": "summary-id",
            "text": "the summary",
            "source_chunk_id": chunk_id,
        }))];
        let pairs = chunk_summary_pairs(&[], &[], &summaries, None, "OR");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].chunk_id.as_deref(), Some(chunk_id.as_str()));
        assert_eq!(pairs[0].summary_rank, Some(0));
        assert_eq!(pairs[0].summary_text.as_deref(), Some("the summary"));
        assert!(pairs[0].chunk.is_none());
    }

    #[test]
    fn summary_id_for_chunk_matches_uuid5_scheme() {
        let chunk_id = Uuid::new_v4();
        let expected = Uuid::new_v5(&chunk_id, b"TextSummary").to_string();
        assert_eq!(summary_id_for_chunk(&chunk_id.to_string()), Some(expected));
        // Non-UUID input -> None.
        assert_eq!(summary_id_for_chunk("not-a-uuid"), None);
    }

    #[test]
    fn summary_id_for_chunk_known_vector() {
        // Fixed chunk id -> stable uuid5(chunk, "TextSummary").
        let chunk_id = "12345678-1234-5678-1234-567812345678";
        let parsed = Uuid::parse_str(chunk_id).unwrap();
        let expected = Uuid::new_v5(&parsed, b"TextSummary").to_string();
        assert_eq!(summary_id_for_chunk(chunk_id), Some(expected));
    }

    #[test]
    fn source_chunk_ids_only_for_summary_only_pairs() {
        let chunk_id = Uuid::new_v4().to_string();
        let summaries = vec![item(json!({
            "id": "s",
            "text": "sum",
            "source_chunk_id": chunk_id,
        }))];
        let pairs = chunk_summary_pairs(&[], &[], &summaries, None, "OR");
        let to_load = source_chunk_ids_to_load(&pairs);
        assert_eq!(to_load, vec![chunk_id]);
    }
}
