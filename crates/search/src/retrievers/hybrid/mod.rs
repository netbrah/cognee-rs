//! Hybrid-retrieval ranking core (chunk + BM25 lane).
//!
//! Port of Python cognee's `cognee/modules/retrieval/hybrid/{results,pairs,
//! ranking,chunks}.py`. This module supplies the future `HybridRetriever`
//! (`SearchType::HybridCompletion`, wired in a later task) with its
//! chunk-ranking spine: a per-query Okapi BM25 lexical lane (landed separately
//! as [`crate::retrievers::bm25_scored_chunks`]) merged with the vector
//! `DocumentChunk_text` / `TextSummary_text` channels via Reciprocal Rank
//! Fusion (RRF), with optional importance-weight boosting.
//!
//! This task delivers the pure ranking/merge logic and its vector-orchestration
//! wrapper ([`retrieve_hybrid_chunks`]) only — it does **not** wire a
//! `SearchRetriever` impl, a `SearchType` variant, or wire-level params (those
//! land in a later task). The public surface is therefore `pub(crate)` and inert
//! from outside the crate until that task consumes it.

#[allow(
    dead_code,
    reason = "shared helpers consumed by the later hybrid-assembly task (P1-09) and the entity/facts lane (P1-08); exercised now by this module's tests"
)]
mod chunks;
#[allow(
    dead_code,
    reason = "shared helpers consumed by the later hybrid-assembly task (P1-09) and the entity/facts lane (P1-08); exercised now by this module's tests"
)]
mod pairs;
#[allow(
    dead_code,
    reason = "shared helpers consumed by the later hybrid-assembly task (P1-09) and the entity/facts lane (P1-08); exercised now by this module's tests"
)]
mod ranking;
#[allow(
    dead_code,
    reason = "shared helpers consumed by the later hybrid-assembly task (P1-09) and the entity/facts lane (P1-08); exercised now by this module's tests"
)]
mod results;

#[allow(
    unused_imports,
    reason = "public surface consumed by the later hybrid-assembly task (P1-09); exercised now by this module's tests"
)]
pub(crate) use chunks::{HybridChunksResult, retrieve_hybrid_chunks};
