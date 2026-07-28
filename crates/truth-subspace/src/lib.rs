//! Truth-subspace alignment math plus the vector-engine-backed centroid
//! persistence glue. Ported from Python `cognee/modules/truth_subspace/`.
//!
//! Most of the crate is pure math with no I/O; the exception is the persistence
//! pair [`load_centroids`] / [`upsert_centroids`] (in [`centroids`]), which
//! read and write centroid payloads through a [`cognee_vector::VectorDB`] so
//! truth-subspace reranking survives process restarts.
//!
//! Everything else here is NEUTRAL when inputs are missing/empty/zero:
//! [`align::truth_score`] returns `0.5` and [`align::truth_factor`] returns
//! `1.0`, so callers that pass nothing leave baseline scoring untouched. This
//! keeps the Phase-2 truth-subspace re-ranking knobs (`use_truth_weight` /
//! `build_truth_subspace`, both default-off) safe by construction.
#![forbid(unsafe_code)]

pub mod align;
pub mod centroids;
pub mod constants;
pub mod models;

pub use centroids::{
    build_centroids_from_learning_vectors, centroid_id, centroids_changed,
    extend_centroids_with_learning_vectors, learning_id, load_centroids, upsert_centroids,
};
pub use constants::{DEFAULT_K, TRUTH_CENTROID_COLLECTION, TRUTH_NODE_SET, truth_session_node_set};
pub use models::TruthCentroidPayload;
