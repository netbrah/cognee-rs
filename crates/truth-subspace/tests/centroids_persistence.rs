//! Integration tests for the vector-engine-backed centroid persistence pair
//! (`load_centroids` / `upsert_centroids`), ported alongside Python's
//! `centroids.py:165-194`. Backed by `MockVectorDB` (cognee-vector's `testing`
//! feature) so the round-trip runs entirely in-memory with no I/O.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test — panics are acceptable failures"
)]

use cognee_truth_subspace::{
    DEFAULT_K, TRUTH_CENTROID_COLLECTION, TruthCentroidPayload,
    build_centroids_from_learning_vectors, load_centroids, upsert_centroids,
};
use cognee_vector::{MockVectorDB, VectorDB};

/// Build a centroid payload with distinctive, slot-derived values so a
/// round-trip that drops or scrambles any field is caught.
fn payload(dataset_id: &str, slot: usize, centroid: Vec<f64>) -> TruthCentroidPayload {
    TruthCentroidPayload {
        dataset_id: dataset_id.to_string(),
        slot,
        count: (slot + 1) * 3,
        truth_epoch: 7,
        updated_at: 1_700_000_000_000 + slot as i64,
        centroid,
        learning_ids: vec![format!("id-{slot}-a"), format!("id-{slot}-b")],
    }
}

/// Every field survives the upsert → load round trip, including the full-`f64`
/// `centroid` (carried in the metadata blob, not the `f32`-narrowed vector
/// column), so exact `PartialEq` equality holds.
#[tokio::test]
async fn upsert_then_load_round_trips_all_fields() {
    let db = MockVectorDB::new();

    // Realistic payloads from P2-02's builder plus hand-rolled ones exercising
    // awkward f64 values that must round-trip losslessly through JSON metadata.
    let learning_vectors: Vec<(String, Vec<f64>)> = (0..5)
        .map(|i| (format!("s{i}"), vec![1.0, i as f64, 0.5]))
        .collect();
    let mut expected =
        build_centroids_from_learning_vectors("ds", &learning_vectors, 3, Some(123), DEFAULT_K);
    // Overwrite one centroid with values that stress f64 precision.
    if let Some(first) = expected.first_mut() {
        first.centroid = vec![0.123_456_789_012_345_67, -0.987_654_321_098_765_4, 0.5];
    }

    upsert_centroids(&db, &expected).await.unwrap();

    // Physical collection name is exactly Python's "TruthCentroid_vector".
    assert!(
        db.has_collection(TRUTH_CENTROID_COLLECTION.0, TRUTH_CENTROID_COLLECTION.1)
            .await
            .unwrap()
    );

    let loaded = load_centroids(&db, "ds", DEFAULT_K).await.unwrap();
    assert_eq!(
        loaded, expected,
        "all fields (incl. full-f64 centroid) must round-trip exactly"
    );
}

/// `retrieve` gives no ordering guarantee; `load_centroids` must sort by slot.
#[tokio::test]
async fn load_centroids_sorts_by_slot() {
    let db = MockVectorDB::new();
    let shuffled = vec![
        payload("ds", 3, vec![0.1, 0.2]),
        payload("ds", 0, vec![0.3, 0.4]),
        payload("ds", 2, vec![0.5, 0.6]),
        payload("ds", 1, vec![0.7, 0.8]),
    ];
    upsert_centroids(&db, &shuffled).await.unwrap();

    let loaded = load_centroids(&db, "ds", DEFAULT_K).await.unwrap();
    let slots: Vec<usize> = loaded.iter().map(|c| c.slot).collect();
    assert_eq!(slots, vec![0, 1, 2, 3]);
}

/// Centroids for two datasets share the one `TruthCentroid_vector` collection;
/// `load_centroids(dataset_a)` must never return a `dataset_b` row.
#[tokio::test]
async fn load_centroids_filters_other_datasets() {
    let db = MockVectorDB::new();
    upsert_centroids(
        &db,
        &[
            payload("ds-a", 0, vec![1.0, 0.0]),
            payload("ds-a", 1, vec![0.0, 1.0]),
        ],
    )
    .await
    .unwrap();
    upsert_centroids(&db, &[payload("ds-b", 0, vec![0.5, 0.5])])
        .await
        .unwrap();

    let a = load_centroids(&db, "ds-a", DEFAULT_K).await.unwrap();
    assert_eq!(a.len(), 2);
    assert!(a.iter().all(|c| c.dataset_id == "ds-a"));

    let b = load_centroids(&db, "ds-b", DEFAULT_K).await.unwrap();
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].dataset_id, "ds-b");
    assert_eq!(b[0].slot, 0);
}

/// A dataset with fewer than `k` centroids returns exactly the present slots
/// (sorted) and does not error on the missing ids.
#[tokio::test]
async fn load_centroids_partial_slots() {
    let db = MockVectorDB::new();
    let three = vec![
        payload("ds", 0, vec![1.0, 0.0]),
        payload("ds", 2, vec![0.0, 1.0]),
        payload("ds", 5, vec![0.5, 0.5]),
    ];
    upsert_centroids(&db, &three).await.unwrap();

    // k=8 but only 3 slots present → exactly those 3, sorted, 5 misses ignored.
    let loaded = load_centroids(&db, "ds", 8).await.unwrap();
    let slots: Vec<usize> = loaded.iter().map(|c| c.slot).collect();
    assert_eq!(slots, vec![0, 2, 5]);
}

/// An empty `upsert_centroids` short-circuits before any adapter call, so no
/// collection is created and a subsequent load returns empty.
#[tokio::test]
async fn upsert_centroids_empty_is_noop() {
    let db = MockVectorDB::new();
    upsert_centroids(&db, &[]).await.unwrap();

    assert!(
        !db.has_collection(TRUTH_CENTROID_COLLECTION.0, TRUTH_CENTROID_COLLECTION.1)
            .await
            .unwrap(),
        "empty upsert must not create the TruthCentroid_vector collection"
    );
    let loaded = load_centroids(&db, "ds", DEFAULT_K).await.unwrap();
    assert!(loaded.is_empty());
}
