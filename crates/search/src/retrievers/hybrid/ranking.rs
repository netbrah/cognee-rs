//! Reciprocal Rank Fusion (RRF) + importance-weight scoring.
//!
//! Port of `cognee/modules/retrieval/hybrid/ranking.py`, **Phase-1 subset
//! only**. The Python function also threads `use_truth_weight` / `q_coords` /
//! `truth_state_by_id` / `current_truth_epoch` for the Phase-2 truth-subspace
//! boost; per the locked Phase-2 deferral those parameters are intentionally
//! absent here and will be added by the Phase-2 task without touching this
//! code path.

use serde_json::Value;

use crate::retrievers::hybrid::pairs::ChunkSummaryPair;
use crate::retrievers::hybrid::results::{payload, result_id};

/// RRF constant `k` derived from the requested chunk count.
///
/// Port of `_rrf_k` (`ranking.py:51-52`): `clamp(20 + 2*chunks_top_k, 30, 60)`.
/// Python's `max(30, min(60, ...))` and Rust's `.clamp(30, 60)` are equivalent
/// for non-negative inputs.
pub(crate) fn rrf_k(chunks_top_k: usize) -> usize {
    (20 + 2 * chunks_top_k).clamp(30, 60)
}

/// Multiplicative importance boost read from a chunk payload.
///
/// Port of `_importance_factor` (`ranking.py:55-59`): reads
/// `payload["importance_weight"]` as a JSON number (default `0.5` if missing or
/// non-numeric — this rejects strings/bools/arrays, an intentional, documented
/// divergence from CPython's `isinstance(True, int)` quirk), clamps to
/// `[0.0, 1.0]`, and returns `0.75 + 0.5 * importance`.
pub(crate) fn importance_factor(chunk_payload: &Value) -> f64 {
    let importance = chunk_payload
        .get("importance_weight")
        .and_then(Value::as_f64)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    0.75 + 0.5 * importance
}

/// Rank chunk↔summary pairs by RRF (optionally importance-weighted) and
/// truncate to `limit`.
///
/// Port of `rank_chunk_summary_pairs` (`ranking.py:7-48`, Phase-1 subset). For
/// each pair carrying a `chunk`, collect the present ranks from
/// `(bm25_rank, vector_rank, summary_rank)` (skip if none), compute
/// `rrf_score = Σ 1/(k + rank + 1)`, multiply by `importance_factor` when
/// `use_importance_weight`, and sort by `(-final, -rrf, min_rank, chunk_id)`
/// (float legs via `f64::total_cmp`, per the locked total-ordering decision).
pub(crate) fn rank_chunk_summary_pairs(
    pairs: Vec<ChunkSummaryPair>,
    limit: usize,
    use_importance_weight: bool,
) -> Vec<ChunkSummaryPair> {
    if limit == 0 {
        return vec![];
    }

    let k = rrf_k(limit);
    let mut ranked: Vec<(f64, f64, usize, String, ChunkSummaryPair)> = Vec::new();

    for pair in pairs {
        let Some(chunk) = pair.chunk.as_ref() else {
            continue;
        };

        let ranks: Vec<usize> = [pair.bm25_rank, pair.vector_rank, pair.summary_rank]
            .into_iter()
            .flatten()
            .collect();
        if ranks.is_empty() {
            continue;
        }

        let rrf_score: f64 = ranks.iter().map(|rank| 1.0 / (k + rank + 1) as f64).sum();
        let final_score = if use_importance_weight {
            rrf_score * importance_factor(payload(chunk))
        } else {
            rrf_score
        };
        let min_rank = ranks.iter().copied().min().unwrap_or(0);
        let chunk_id = pair
            .chunk_id
            .clone()
            .or_else(|| result_id(chunk))
            .unwrap_or_default();

        ranked.push((final_score, rrf_score, min_rank, chunk_id, pair));
    }

    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.1.total_cmp(&left.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });

    ranked
        .into_iter()
        .take(limit)
        .map(|(_, _, _, _, pair)| pair)
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::retrievers::hybrid::pairs::ChunkSummaryPair;
    use crate::types::SearchItem;

    fn chunk_item(id: &str, importance: Option<f64>) -> SearchItem {
        let mut payload = json!({"id": id, "text": format!("text-{id}")});
        if let Some(weight) = importance {
            payload["importance_weight"] = json!(weight);
        }
        SearchItem {
            id: None,
            score: None,
            payload,
        }
    }

    fn pair(
        id: &str,
        bm25: Option<usize>,
        vector: Option<usize>,
        summary: Option<usize>,
        importance: Option<f64>,
    ) -> ChunkSummaryPair {
        ChunkSummaryPair {
            chunk_id: Some(id.to_string()),
            chunk_text: Some(format!("text-{id}")),
            summary_id: None,
            summary_text: None,
            chunk: Some(chunk_item(id, importance)),
            bm25_rank: bm25,
            vector_rank: vector,
            summary_rank: summary,
        }
    }

    #[test]
    fn rrf_k_boundaries() {
        assert_eq!(rrf_k(0), 30);
        assert_eq!(rrf_k(5), 30); // 20 + 10 = 30
        assert_eq!(rrf_k(10), 40); // 20 + 20 = 40 (mid-range)
        assert_eq!(rrf_k(20), 60); // 20 + 40 = 60
        assert_eq!(rrf_k(100), 60); // clamped
    }

    #[test]
    fn importance_factor_bounds() {
        assert_eq!(importance_factor(&json!({})), 1.0); // default 0.5 -> 0.75 + 0.25
        assert_eq!(importance_factor(&json!({"importance_weight": "x"})), 1.0);
        assert_eq!(importance_factor(&json!({"importance_weight": 0.0})), 0.75);
        assert_eq!(importance_factor(&json!({"importance_weight": 1.0})), 1.25);
        assert_eq!(importance_factor(&json!({"importance_weight": 0.5})), 1.0);
        // Out of range clamps.
        assert_eq!(importance_factor(&json!({"importance_weight": 1.5})), 1.25);
        assert_eq!(importance_factor(&json!({"importance_weight": -1.0})), 0.75);
    }

    #[test]
    fn limit_zero_returns_empty() {
        let pairs = vec![pair("a", Some(0), Some(0), Some(0), None)];
        assert!(rank_chunk_summary_pairs(pairs, 0, false).is_empty());
    }

    #[test]
    fn pair_without_ranks_is_skipped() {
        let pairs = vec![pair("a", None, None, None, None)];
        assert!(rank_chunk_summary_pairs(pairs, 5, false).is_empty());
    }

    #[test]
    fn all_three_lanes_outrank_single_lane() {
        // Same rank slot value but different lane counts.
        let three = pair("three", Some(1), Some(1), Some(1), None);
        let one = pair("one", Some(0), None, None, None);
        let ranked = rank_chunk_summary_pairs(vec![one, three], 5, false);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].chunk_id.as_deref(), Some("three"));
    }

    #[test]
    fn importance_weight_can_reorder() {
        // Identical ranks; the higher importance weight wins when enabled.
        let low = pair("low", Some(0), None, None, Some(0.0));
        let high = pair("high", Some(0), None, None, Some(1.0));
        let ranked = rank_chunk_summary_pairs(vec![low, high], 5, true);
        assert_eq!(ranked[0].chunk_id.as_deref(), Some("high"));

        // With importance off, tie-break falls to chunk_id string order.
        let low = pair("low", Some(0), None, None, Some(0.0));
        let high = pair("high", Some(0), None, None, Some(1.0));
        let ranked = rank_chunk_summary_pairs(vec![low, high], 5, false);
        assert_eq!(ranked[0].chunk_id.as_deref(), Some("high")); // "high" < "low"
    }

    #[test]
    fn tie_break_by_chunk_id() {
        // Identical scores/ranks -> ascending chunk_id string.
        let b = pair("bbb", Some(0), None, None, None);
        let a = pair("aaa", Some(0), None, None, None);
        let ranked = rank_chunk_summary_pairs(vec![b, a], 5, false);
        assert_eq!(ranked[0].chunk_id.as_deref(), Some("aaa"));
        assert_eq!(ranked[1].chunk_id.as_deref(), Some("bbb"));
    }

    #[test]
    fn hand_computed_rrf_score_orders_by_min_rank_on_tie() {
        // limit=5 -> k=30. Pair X: ranks {bm25:0} -> 1/(30+0+1) = 1/31.
        // Pair Y: ranks {vector:0, summary:2} -> 1/31 + 1/33.
        // Y has a higher rrf sum, so ranks first.
        let x = pair("x", Some(0), None, None, None);
        let y = pair("y", None, Some(0), Some(2), None);
        let ranked = rank_chunk_summary_pairs(vec![x, y], 5, false);
        assert_eq!(ranked[0].chunk_id.as_deref(), Some("y"));
    }

    #[test]
    fn truncates_to_limit() {
        let pairs = vec![
            pair("a", Some(0), None, None, None),
            pair("b", Some(1), None, None, None),
            pair("c", Some(2), None, None, None),
        ];
        let ranked = rank_chunk_summary_pairs(pairs, 2, false);
        assert_eq!(ranked.len(), 2);
    }
}
