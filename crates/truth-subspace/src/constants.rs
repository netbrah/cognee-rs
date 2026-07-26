//! Truth-subspace constants. Ported from Python
//! `cognee/modules/truth_subspace/constants.py`.

/// Slot capacity per dataset — the maximum number of centroid slots a single
/// dataset's truth subspace holds (`constants.py:3`).
pub const DEFAULT_K: usize = 8;

/// Vector collection for truth centroids, as a `(data_type, field_name)` pair.
///
/// Python stores this as one string `"TruthCentroid_vector"`
/// (`constants.py:1`) because its vector API takes a single collection name.
/// Rust's `VectorDBTrait` always takes a `(data_type, field_name)` pair and
/// derives the physical name internally via `format!("{data_type}_{field_name}")`,
/// so the Rust constant is the pair that reconstructs Python's exact string.
/// Not used for I/O here — P2-03 (`load_centroids`/`upsert_centroids`) will
/// destructure it for `vector_engine` calls.
pub const TRUTH_CENTROID_COLLECTION: (&str, &str) = ("TruthCentroid", "vector");

// NOTE: `TRUTH_NODE_SET` and `truth_session_node_set` (`constants.py:2,6-7`)
// belong to the graph nodeset-query path used by `build.py` and are deferred
// to P2-03 — out of scope for this task.
