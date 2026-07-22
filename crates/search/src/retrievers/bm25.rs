//! Per-query Okapi BM25 lexical scorer for `DocumentChunk` nodes.
//!
//! This is the "BM25 lane" that `HybridRetriever` fans out to alongside the
//! two vector-search lanes. It is a faithful port of Python cognee's
//! `BM25ChunksRetriever` (`cognee/modules/retrieval/bm25_retriever.py`) and the
//! fail-open `search_bm25_chunks` contract
//! (`cognee/modules/retrieval/hybrid/chunks.py:112-140`).
//!
//! Design notes (locked decisions):
//! - Real Okapi BM25 with the exact Python constants `k1 = 1.5`, `b = 0.75` and
//!   the Robertson–Sparck-Jones-floored IDF
//!   (`idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))`). Do not "improve" the
//!   formula — bit-for-bit constant/formula match is the acceptance bar.
//! - The corpus is rebuilt **fresh on every call** (no process-lifetime cache;
//!   we deliberately do NOT reuse `LexicalRetriever`'s `OnceCell`). Python
//!   constructs a new `BM25ChunksRetriever` per search, so `initialize()`
//!   always reloads the chunks from the graph.
//!
//! Intentionally **out of scope** here (the hybrid-assembly caller's job):
//! `node_name` / `node_name_filter_operator` filtering and the
//! `chunks_top_k * 2` candidate doubling (`chunks.py:41,137`). This function
//! takes `limit` as given and does no node-name scoping — the caller must apply
//! node filtering to the returned payloads itself.
//!
//! There is no `NoDataError` distinction in Rust: `get_filtered_graph_data`
//! surfaces a single `GraphDBError`, so any load error collapses into the same
//! warn-and-empty path as Python's catch-all `except Exception`.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use cognee_graph::{GraphDBResult, GraphDBTrait, NodeData};
use regex::Regex;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::utils::DEFAULT_STOP_WORDS;

/// BM25 term-frequency saturation parameter (Python default, locked).
const BM25_K1: f64 = 1.5;
/// BM25 length-normalization parameter (Python default, locked).
const BM25_B: f64 = 0.75;
const DOCUMENT_CHUNK_TYPE: &str = "DocumentChunk";

/// `\w+` tokenizer regex, compiled once. Unicode-aware by default, matching
/// Python 3's `re` default `\w`.
#[allow(
    clippy::expect_used,
    reason = "regex literal is valid at compile time — failure is impossible"
)]
static WORD_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\w+").expect("static \\w+ pattern compiles (build-time guarantee)")
});

/// Lowercase, split on word characters (`\w+`), and drop stop words.
///
/// Port of `tokenize_words` from
/// `cognee/modules/retrieval/lexical_retriever.py:15-23`. This is a standalone
/// tokenizer for the BM25 lane; it deliberately does not reuse
/// `LexicalRetriever::tokenize` (a hand-rolled char scanner for the Jaccard
/// retriever).
fn tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    WORD_REGEX
        .find_iter(&lowered)
        .map(|m| m.as_str())
        .filter(|token| !DEFAULT_STOP_WORDS.contains(*token))
        .map(str::to_string)
        .collect()
}

/// Corpus statistics computed once per query from the tokenized chunks.
struct CorpusStats {
    /// Average chunk length in tokens (`0.0` when the corpus is empty).
    avg_chunk_length: f64,
    /// Per-token inverse document frequency.
    idf: HashMap<String, f64>,
}

/// Build one BM25 corpus entry from a raw graph node, or `None` if the node is
/// not a usable `DocumentChunk`.
///
/// Mirrors `LexicalRetriever::load_document_chunks` plus Python's `id`-backfill
/// (`lexical_retriever.py:76-77`): some graph adapters omit `"id"` from node
/// payloads, so we backfill it from the resolved UUID (or the raw graph node
/// id) to keep downstream hybrid pairing able to key chunks by `payload["id"]`.
fn build_chunk_entry(
    node_id: &str,
    node_data: &NodeData,
) -> Option<(Option<Uuid>, Value, Vec<String>)> {
    let node_type = node_data
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    // The filter widens to any of the given values, so re-check the type here,
    // matching `lexical_retriever.rs:151-157`.
    if node_type != DOCUMENT_CHUNK_TYPE {
        return None;
    }

    let text = node_data.get("text").and_then(Value::as_str)?;
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return None;
    }

    let item_id = node_data
        .get("id")
        .and_then(Value::as_str)
        .and_then(|id| Uuid::parse_str(id).ok())
        .or_else(|| Uuid::parse_str(node_id).ok());

    let mut payload = Value::Object(
        node_data
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect(),
    );

    // Backfill `payload["id"]` when absent or not a string (parity with
    // `lexical_retriever.py:76-77`).
    let has_string_id = payload.get("id").and_then(Value::as_str).is_some();
    if !has_string_id {
        let id_value = item_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| node_id.to_string());
        if let Value::Object(map) = &mut payload {
            map.insert("id".to_string(), Value::String(id_value));
        }
    }

    Some((item_id, payload, tokens))
}

/// Load the full `DocumentChunk` corpus from the graph, fresh (no cache).
///
/// Mirrors `LexicalRetriever::load_document_chunks`
/// (`lexical_retriever.rs:142-175`) but deliberately omits the `OnceCell`
/// caching — decision 5 requires a fresh `get_filtered_graph_data` call on
/// every `bm25_scored_chunks` invocation.
async fn load_chunk_corpus(
    graph_db: &dyn GraphDBTrait,
) -> GraphDBResult<Vec<(Option<Uuid>, Value, Vec<String>)>> {
    let filters = HashMap::from([(Cow::Borrowed("type"), vec![json!(DOCUMENT_CHUNK_TYPE)])]);
    let (nodes, _edges) = graph_db.get_filtered_graph_data(&filters).await?;

    let mut corpus = Vec::new();
    for (node_id, node_data) in nodes {
        if let Some(entry) = build_chunk_entry(&node_id, &node_data) {
            corpus.push(entry);
        }
    }
    Ok(corpus)
}

/// Compute average chunk length and per-token IDF from tokenized chunks.
///
/// Port of `_build_corpus_stats` (`bm25_retriever.py:73-87`). All math is in
/// `f64` to match Python's `float` and avoid precision drift across IDF
/// lookups; the final score is narrowed to `f32` only at the return boundary.
fn build_corpus_stats(corpus: &[(Option<Uuid>, Value, Vec<String>)]) -> CorpusStats {
    let total_chunks = corpus.len();
    let mut total_length = 0usize;
    let mut document_frequency: HashMap<&str, usize> = HashMap::new();

    for (_, _, tokens) in corpus {
        total_length += tokens.len();
        // Count each token once per chunk that contains it (over the chunk's
        // unique tokens), matching `for token in set(tokens)` in Python.
        let unique: HashSet<&str> = tokens.iter().map(String::as_str).collect();
        for token in unique {
            *document_frequency.entry(token).or_insert(0) += 1;
        }
    }

    let avg_chunk_length = if total_chunks == 0 {
        0.0
    } else {
        total_length as f64 / total_chunks as f64
    };

    let n = total_chunks as f64;
    let idf = document_frequency
        .into_iter()
        .map(|(token, df)| {
            let df = df as f64;
            (token.to_string(), (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
        })
        .collect();

    CorpusStats {
        avg_chunk_length,
        idf,
    }
}

/// Okapi BM25 score of a chunk against the query, summed over unique query
/// terms with `tf > 0`.
///
/// Port of `_scorer` (`bm25_retriever.py:89-104`). Returns `0.0` for empty
/// input or an empty corpus (`avg_chunk_length == 0.0`).
fn bm25_score(
    query_tokens: &[String],
    chunk_tokens: &[String],
    idf: &HashMap<String, f64>,
    avg_chunk_length: f64,
) -> f64 {
    if query_tokens.is_empty() || chunk_tokens.is_empty() || avg_chunk_length == 0.0 {
        return 0.0;
    }

    let mut term_frequencies: HashMap<&str, usize> = HashMap::new();
    for token in chunk_tokens {
        *term_frequencies.entry(token.as_str()).or_insert(0) += 1;
    }

    let length_norm =
        BM25_K1 * (1.0 - BM25_B + BM25_B * chunk_tokens.len() as f64 / avg_chunk_length);

    let mut score = 0.0;
    let unique_query: HashSet<&str> = query_tokens.iter().map(String::as_str).collect();
    for token in unique_query {
        let tf = term_frequencies.get(token).copied().unwrap_or(0);
        if tf == 0 {
            continue;
        }
        let tf = tf as f64;
        // `idf.get(...).unwrap_or(&0.0)` on an Option is fine — the forbidden
        // pattern is `.unwrap()`. Unseen query terms contribute IDF 0.0,
        // matching Python's `self.idf.get(token, 0.0)`.
        let token_idf = idf.get(token).copied().unwrap_or(0.0);
        score += token_idf * (tf * (BM25_K1 + 1.0)) / (tf + length_norm);
    }
    score
}

/// Score every `DocumentChunk` in the graph against `query` with Okapi BM25 and
/// return the top `limit` `(payload, score)` pairs, highest score first.
///
/// Fail-open, per the Python `search_bm25_chunks` contract
/// (`cognee/modules/retrieval/hybrid/chunks.py:112-140`): this returns a plain
/// `Vec` (never a `Result`). Empty results are returned for `limit == 0`, an
/// empty query, a corpus-load error (logged at `warn`), or an empty corpus
/// (logged at `debug`).
///
/// The corpus is rebuilt fresh on every call — there is no cross-call cache
/// (decision 5).
///
/// **Caller responsibility:** node-name filtering
/// (`payload_matches_node_filter`) and the `chunks_top_k * 2` candidate
/// doubling are NOT applied here; the hybrid-assembly caller must apply them
/// itself. `limit` is used as given.
#[allow(
    dead_code,
    reason = "internal API consumed by the later hybrid-assembly task (P1-07/P1-09); exercised now by this module's tests"
)]
pub(crate) async fn bm25_scored_chunks(
    graph_db: &Arc<dyn GraphDBTrait>,
    query: &str,
    limit: usize,
) -> Vec<(Value, f32)> {
    if limit == 0 {
        return vec![];
    }

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return vec![];
    }

    let corpus = match load_chunk_corpus(graph_db.as_ref()).await {
        Ok(corpus) => corpus,
        Err(error) => {
            tracing::warn!(%error, "BM25 chunk retrieval failed; using vector chunks only");
            return vec![];
        }
    };
    if corpus.is_empty() {
        tracing::debug!("BM25 chunk corpus is empty; no DocumentChunk nodes to score");
        return vec![];
    }

    let stats = build_corpus_stats(&corpus);

    let mut scored: Vec<(Value, f64)> = corpus
        .iter()
        .map(|(_, payload, tokens)| {
            let score = bm25_score(&query_tokens, tokens, &stats.idf, stats.avg_chunk_length);
            (payload.clone(), score)
        })
        .collect();

    // Sort descending by score (ties arbitrary — Python's `nlargest`/`heapq`
    // is not stable either), truncate to `limit`, THEN drop non-positive
    // scores. Order matters: Python truncates first (`nlargest(top_k, ...)`)
    // and only then filters `score <= 0` (`chunks.py:133-134`), so a weakly
    // matching query may return fewer than `limit` results.
    scored.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);

    scored
        .into_iter()
        .filter(|(_, score)| *score > 0.0)
        .map(|(payload, score)| (payload, score as f32))
        .collect()
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
    use serde::Serialize;
    use serde_json::Value;
    use uuid::Uuid;

    use super::{bm25_score, bm25_scored_chunks, build_chunk_entry, build_corpus_stats, tokenize};

    #[derive(Serialize)]
    struct DocumentChunkNode {
        id: String,
        #[serde(rename = "type")]
        kind: String,
        text: String,
    }

    async fn add_chunk(graph_db: &MockGraphDB, text: &str) -> String {
        let id = Uuid::new_v4().to_string();
        let node = DocumentChunkNode {
            id: id.clone(),
            kind: "DocumentChunk".to_string(),
            text: text.to_string(),
        };
        graph_db.add_node(&node).await.unwrap();
        id
    }

    fn text_of(payload: &Value) -> &str {
        payload.get("text").and_then(Value::as_str).unwrap()
    }

    // --- Tokenizer -------------------------------------------------------

    #[test]
    fn tokenizer_lowercases_splits_and_drops_stop_words() {
        // "the" is a stop word and must be dropped; punctuation is stripped.
        assert_eq!(
            tokenize("The Rust Book, 2nd Edition!"),
            vec!["rust", "book", "2nd", "edition"]
        );
    }

    #[test]
    fn tokenizer_returns_empty_for_all_stop_words() {
        assert!(tokenize("the and or but is").is_empty());
    }

    // --- Corpus stats / IDF ---------------------------------------------

    #[test]
    fn idf_and_avg_length_match_hand_computed_values() {
        // 3-chunk corpus. "shared" appears in all 3; "rare" in exactly 1.
        let corpus = vec![
            (
                None,
                Value::Null,
                vec!["shared".to_string(), "rare".to_string()],
            ),
            (
                None,
                Value::Null,
                vec!["shared".to_string(), "alpha".to_string()],
            ),
            (
                None,
                Value::Null,
                vec!["shared".to_string(), "beta".to_string()],
            ),
        ];

        let stats = build_corpus_stats(&corpus);

        // avg = (2 + 2 + 2) / 3 = 2.0
        assert!((stats.avg_chunk_length - 2.0).abs() < 1e-12);

        // "rare" present in 1 of 3: ln(1 + (3 - 1 + 0.5)/(1 + 0.5)) ≈ 0.9808
        let rare_idf = stats.idf.get("rare").copied().unwrap();
        assert!((rare_idf - 0.980829253011726).abs() < 1e-9);

        // "shared" present in all 3: ln(1 + (3 - 3 + 0.5)/(3 + 0.5)) ≈ 0.1335
        let shared_idf = stats.idf.get("shared").copied().unwrap();
        assert!((shared_idf - 0.13353139262452257).abs() < 1e-9);

        // Rare term must have a higher IDF than a term in every chunk.
        assert!(rare_idf > shared_idf);
    }

    // --- Scorer monotonicity --------------------------------------------

    #[test]
    fn scorer_is_monotonic_in_term_frequency() {
        let query = vec!["rust".to_string()];
        let idf = std::collections::HashMap::from([("rust".to_string(), 1.0)]);
        let avg_len = 4.0;

        // Same length (4 tokens), differing tf for "rust".
        let low_tf = vec![
            "rust".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ];
        let high_tf = vec![
            "rust".to_string(),
            "rust".to_string(),
            "rust".to_string(),
            "z".to_string(),
        ];

        let low = bm25_score(&query, &low_tf, &idf, avg_len);
        let high = bm25_score(&query, &high_tf, &idf, avg_len);
        assert!(high > low, "higher tf must score higher: {high} !> {low}");
    }

    #[test]
    fn scorer_penalizes_longer_chunks_for_same_tf() {
        let query = vec!["rust".to_string()];
        let idf = std::collections::HashMap::from([("rust".to_string(), 1.0)]);
        let avg_len = 4.0;

        // Same tf (1), differing length.
        let short = vec!["rust".to_string(), "x".to_string()];
        let long = vec![
            "rust".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];

        let short_score = bm25_score(&query, &short, &idf, avg_len);
        let long_score = bm25_score(&query, &long, &idf, avg_len);
        assert!(
            short_score > long_score,
            "shorter chunk with same tf must score higher: {short_score} !> {long_score}"
        );
    }

    #[test]
    fn scorer_returns_zero_for_empty_inputs() {
        let idf = std::collections::HashMap::new();
        assert_eq!(bm25_score(&[], &["a".to_string()], &idf, 1.0), 0.0);
        assert_eq!(bm25_score(&["a".to_string()], &[], &idf, 1.0), 0.0);
        assert_eq!(
            bm25_score(&["a".to_string()], &["a".to_string()], &idf, 0.0),
            0.0
        );
    }

    // --- Payload id backfill --------------------------------------------

    #[test]
    fn build_chunk_entry_backfills_missing_id_from_node_id() {
        // A DocumentChunk whose node_data has no "id" field at all.
        let mut node_data = cognee_graph::NodeData::new();
        node_data.insert(
            std::borrow::Cow::Borrowed("type"),
            Value::String("DocumentChunk".to_string()),
        );
        node_data.insert(
            std::borrow::Cow::Borrowed("text"),
            Value::String("rust memory safety".to_string()),
        );

        let (_, payload, tokens) =
            build_chunk_entry("graph-node-123", &node_data).expect("valid chunk");

        assert!(!tokens.is_empty());
        assert_eq!(
            payload.get("id").and_then(Value::as_str),
            Some("graph-node-123"),
            "payload id must be backfilled from the graph node id"
        );
    }

    // --- End-to-end via MockGraphDB -------------------------------------

    #[tokio::test]
    async fn rare_term_outranks_common_term() {
        let mock = Arc::new(MockGraphDB::new());
        // "common" appears in all 3 chunks (small idf); "rare" only in chunk1.
        let rare_id = add_chunk(&mock, "common rare").await;
        add_chunk(&mock, "common alpha").await;
        add_chunk(&mock, "common beta").await;
        let graph_db: Arc<dyn GraphDBTrait> = mock;

        let results = bm25_scored_chunks(&graph_db, "rare common", 10).await;

        assert!(!results.is_empty());
        // The chunk matching the rare term must rank first.
        assert_eq!(
            results[0].0.get("id").and_then(Value::as_str),
            Some(rare_id.as_str())
        );
        assert!(text_of(&results[0].0).contains("rare"));
    }

    #[tokio::test]
    async fn drops_non_matching_chunks_with_zero_score() {
        let mock = Arc::new(MockGraphDB::new());
        add_chunk(&mock, "rust memory safety").await;
        add_chunk(&mock, "python asyncio orchestration").await;
        let graph_db: Arc<dyn GraphDBTrait> = mock;

        let results = bm25_scored_chunks(&graph_db, "rust", 10).await;

        // Only the chunk containing "rust" survives the score <= 0 drop.
        assert_eq!(results.len(), 1);
        assert!(text_of(&results[0].0).contains("rust"));
    }

    #[tokio::test]
    async fn limit_zero_returns_empty() {
        let mock = Arc::new(MockGraphDB::new());
        add_chunk(&mock, "rust memory safety").await;
        let graph_db: Arc<dyn GraphDBTrait> = mock;

        let results = bm25_scored_chunks(&graph_db, "rust", 0).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn empty_corpus_returns_empty() {
        let mock = Arc::new(MockGraphDB::new());
        let graph_db: Arc<dyn GraphDBTrait> = mock;

        let results = bm25_scored_chunks(&graph_db, "rust", 10).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn query_of_only_stop_words_returns_empty() {
        let mock = Arc::new(MockGraphDB::new());
        add_chunk(&mock, "rust memory safety").await;
        let graph_db: Arc<dyn GraphDBTrait> = mock;

        let results = bm25_scored_chunks(&graph_db, "the and or", 10).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn payload_id_present_end_to_end() {
        let mock = Arc::new(MockGraphDB::new());
        let id = add_chunk(&mock, "rust memory safety ownership").await;
        let graph_db: Arc<dyn GraphDBTrait> = mock;

        let results = bm25_scored_chunks(&graph_db, "rust", 10).await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0.get("id").and_then(Value::as_str),
            Some(id.as_str())
        );
    }

    #[tokio::test]
    async fn corpus_is_rebuilt_fresh_each_call() {
        // Regression guard for decision 5: no cross-call cache. A second call
        // after new data lands must see it.
        let mock = Arc::new(MockGraphDB::new());
        add_chunk(&mock, "rust memory safety").await;

        let first = {
            let graph_db: Arc<dyn GraphDBTrait> = Arc::clone(&mock) as Arc<dyn GraphDBTrait>;
            bm25_scored_chunks(&graph_db, "rust", 10).await
        };
        assert_eq!(first.len(), 1);

        // Add a second chunk that also matches the query.
        add_chunk(&mock, "rust ownership model").await;

        let second = {
            let graph_db: Arc<dyn GraphDBTrait> = Arc::clone(&mock) as Arc<dyn GraphDBTrait>;
            bm25_scored_chunks(&graph_db, "rust", 10).await
        };
        assert_eq!(
            second.len(),
            2,
            "second call must reflect the newly-added chunk (no stale cache)"
        );
    }
}
