//! Truth-subspace data models. Ported from Python
//! `cognee/modules/truth_subspace/models.py`.

use serde::{Deserialize, Serialize};

/// Persisted payload for a single truth-centroid slot.
///
/// Ported 1:1 from Python's `TruthCentroidPayload` (`models.py:5-12`), a bare
/// pydantic `BaseModel` (not a `DataPoint` subclass). `slot`/`count` narrow
/// Python's unbounded `int` to `usize` (always non-negative here);
/// `truth_epoch`/`updated_at` stay `i64` (permissive, matches how the deferred
/// build task mints epochs/timestamps). `learning_ids` defaults to an empty
/// vector via `#[serde(default)]`, mirroring `Field(default_factory=list)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthCentroidPayload {
    pub dataset_id: String,
    pub slot: usize,
    pub count: usize,
    pub truth_epoch: i64,
    pub updated_at: i64,
    pub centroid: Vec<f64>,
    #[serde(default)]
    pub learning_ids: Vec<String>,
}
