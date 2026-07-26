//! Centroid-slot helpers for truth-subspace reranking. Ported 1:1 from Python
//! `cognee/modules/truth_subspace/centroids.py`.
//!
//! The core invariant is simple: slot `i` always means the centroid stored in
//! slot `i` for the current truth epoch. Everything here is deterministic so a
//! rebuild from the same learning statements produces the same slots. No I/O
//! and no vector-engine dependency — `load_centroids`/`upsert_centroids`
//! (`centroids.py:165-194`) need a live vector engine and are deferred to P2-03.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::align::cosine;
use crate::models::TruthCentroidPayload;

/// Deterministic UUIDv5 id for a `(dataset_id, slot)` centroid.
///
/// Ports `centroids.py:20-21`:
/// `uuid5(NAMESPACE_OID, f"TruthCentroid:{dataset_id}:{slot}")`. This does NOT
/// route through `cognee_utils::data_point_id_for` — that helper normalizes
/// each value (lowercase, spaces → `_`, apostrophes stripped) and `|`-joins,
/// while Python's `centroid_id` does no normalization and `:`-joins, so the
/// two schemes are not equivalent. A raw `Uuid::new_v5` over the exact Python
/// format string keeps the ids bit-identical across SDKs.
pub fn centroid_id(dataset_id: &str, slot: usize) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("TruthCentroid:{dataset_id}:{slot}").as_bytes(),
    )
}

/// Deterministic UUIDv5 id (as a `String`) for a learning statement.
///
/// Ports `centroids.py:24-26,197-198`: uuid5 over `"TruthLearning:{normalized}"`
/// where `normalized` lowercases then whitespace-collapses the statement.
/// Returns a `String` (Python returns `str(uuid5(...))`) because
/// `TruthCentroidPayload::learning_ids` stores ids as text.
///
/// Accepted divergence: this uses `.to_lowercase()` while Python's
/// `_normalize_statement` uses `str.casefold()`, which is stricter for some
/// non-ASCII input (e.g. German ß). Acceptable under the structural-parity bar
/// for non-ASCII learning statements.
pub fn learning_id(statement: &str) -> String {
    let normalized = statement
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("TruthLearning:{normalized}").as_bytes(),
    )
    .to_string()
}

/// L2-normalize a vector. Ports `centroids.py:29-34`.
///
/// A zero-norm input returns a same-length all-zero vector (no panic, no NaN).
pub fn normalize(vector: &[f64]) -> Vec<f64> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return vec![0.0; vector.len()];
    }
    vector.iter().map(|value| value / norm).collect()
}

/// Front-truncate to at most `k` values, then right zero-pad to length `k`.
///
/// Ports `centroids.py:37-39`. E.g. `pad_coords(&[0.1, 0.2], 4)` →
/// `[0.1, 0.2, 0.0, 0.0]`; `pad_coords(&[0.1, 0.2, 0.3], 2)` → `[0.1, 0.2]`.
pub fn pad_coords(coords: &[f64], k: usize) -> Vec<f64> {
    let mut values: Vec<f64> = coords.iter().take(k).copied().collect();
    if values.len() < k {
        values.resize(k, 0.0);
    }
    values
}

/// Running-mean merge of an existing centroid with a new vector, renormalized.
///
/// Ports `centroids.py:42-52`: empty `old` → `normalize(new)`; else each paired
/// component becomes `(count * old + new) / (count + 1)`, then the result is
/// L2-normalized. Zip-truncates to the shorter input (Python `zip` semantics)
/// rather than padding or erroring on a length mismatch.
pub fn weighted_centroid(old: &[f64], count: usize, new: &[f64]) -> Vec<f64> {
    if old.is_empty() {
        return normalize(new);
    }
    let count = count as f64;
    let merged: Vec<f64> = old
        .iter()
        .zip(new.iter())
        .map(|(&old_value, &new_value)| (count * old_value + new_value) / (count + 1.0))
        .collect();
    normalize(&merged)
}

/// Deduplicate `(statement, vector)` pairs by learning id, sorted by id.
///
/// Ports `centroids.py:55-64`: skips statements that are empty after trimming,
/// keys by [`learning_id`] keeping the FIRST vector seen per id (like
/// `dict.setdefault`), and returns the items SORTED by id string (not insertion
/// order). Zip-truncates to the shorter of `statements`/`vectors`.
pub fn unique_learning_vectors(
    statements: &[String],
    vectors: &[Vec<f64>],
) -> Vec<(String, Vec<f64>)> {
    let mut unique: HashMap<String, Vec<f64>> = HashMap::new();
    for (statement, vector) in statements.iter().zip(vectors.iter()) {
        if statement.trim().is_empty() {
            continue;
        }
        unique
            .entry(learning_id(statement))
            .or_insert_with(|| vector.clone());
    }
    let mut pairs: Vec<(String, Vec<f64>)> = unique.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

/// Build a fresh set of centroid slots from scratch. Ports `centroids.py:67-81`.
///
/// Thin wrapper over [`extend_centroids_with_learning_vectors`] with no
/// existing centroids.
pub fn build_centroids_from_learning_vectors(
    dataset_id: &str,
    learning_vectors: &[(String, Vec<f64>)],
    truth_epoch: i64,
    updated_at: Option<i64>,
    k: usize,
) -> Vec<TruthCentroidPayload> {
    extend_centroids_with_learning_vectors(
        dataset_id,
        &[],
        learning_vectors,
        truth_epoch,
        updated_at,
        k,
    )
}

/// A mutable working slot used while assigning learning vectors.
struct WorkingSlot {
    centroid: Vec<f64>,
    count: usize,
    learning_ids: Vec<String>,
}

/// Assign learning vectors to (up to `k`) centroid slots, merging into the
/// nearest existing slot once full. Ports `centroids.py:84-137`.
///
/// - `updated_at` defaults to `chrono::Utc::now().timestamp_millis()` when
///   `None` (Python `int(datetime.now(timezone.utc).timestamp() * 1000)`).
/// - Working slots seed from `existing_centroids` SORTED by `.slot`, truncated
///   to `k` (silently drops any slot `>= k` from a stale higher-`k` run).
/// - `learning_vectors` are consumed IN THE CALLER'S GIVEN ORDER (callers
///   dedupe via [`unique_learning_vectors`] first if they want that). The raw
///   id from each pair is stored verbatim — it is NOT re-hashed.
/// - While `slots.len() < k`, each new distinct id gets its own slot. Once
///   full, a new vector merges into the nearest slot by MAX cosine similarity.
///
/// Tie-break: Python's `max(range(...), key=...)` is FIRST-wins on equal
/// cosine, whereas Rust's `Iterator::max_by` is LAST-wins. This implements
/// first-wins explicitly (manual best-index tracking, replacing only on a
/// STRICTLY greater score) so ties pick the lowest slot index.
pub fn extend_centroids_with_learning_vectors(
    dataset_id: &str,
    existing_centroids: &[TruthCentroidPayload],
    learning_vectors: &[(String, Vec<f64>)],
    truth_epoch: i64,
    updated_at: Option<i64>,
    k: usize,
) -> Vec<TruthCentroidPayload> {
    let updated_at = updated_at.unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

    let mut sorted_existing: Vec<&TruthCentroidPayload> = existing_centroids.iter().collect();
    sorted_existing.sort_by_key(|centroid| centroid.slot);
    let mut slots: Vec<WorkingSlot> = sorted_existing
        .into_iter()
        .take(k)
        .map(|centroid| WorkingSlot {
            centroid: centroid.centroid.clone(),
            count: centroid.count,
            learning_ids: centroid.learning_ids.clone(),
        })
        .collect();

    let mut seen_learning_ids: HashSet<String> = slots
        .iter()
        .flat_map(|slot| slot.learning_ids.iter().cloned())
        .collect();

    for (new_learning_id, vector) in learning_vectors {
        if seen_learning_ids.contains(new_learning_id) {
            continue;
        }
        let normalized_vector = normalize(vector);
        if slots.len() < k {
            slots.push(WorkingSlot {
                centroid: normalized_vector,
                count: 1,
                learning_ids: vec![new_learning_id.clone()],
            });
            seen_learning_ids.insert(new_learning_id.clone());
            continue;
        }

        // First-wins nearest-slot selection (see the tie-break note above).
        let mut best_index = 0;
        let mut best_score = cosine(&normalized_vector, &slots[0].centroid);
        for (index, slot) in slots.iter().enumerate().skip(1) {
            let score = cosine(&normalized_vector, &slot.centroid);
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }

        let slot = &mut slots[best_index];
        slot.centroid = weighted_centroid(&slot.centroid, slot.count, &normalized_vector);
        slot.count += 1;
        slot.learning_ids.push(new_learning_id.clone());
        seen_learning_ids.insert(new_learning_id.clone());
    }

    slots
        .into_iter()
        .enumerate()
        .map(|(slot_index, slot)| TruthCentroidPayload {
            dataset_id: dataset_id.to_string(),
            slot: slot_index,
            count: slot.count,
            truth_epoch,
            updated_at,
            centroid: slot.centroid,
            learning_ids: slot.learning_ids,
        })
        .collect()
}

/// Whether two centroid sets differ meaningfully. Ports `centroids.py:140-162`.
///
/// Returns `true` immediately if the lengths differ. Otherwise indexes `old`
/// by `.slot`; for each `new` centroid it is a change if there is no matching
/// old slot, the `count` differs, the `centroid` length differs, the
/// `learning_ids` differ (ORDER-sensitive vector equality), or any paired
/// centroid value differs by more than `tolerance` (call sites pass `1e-6`).
///
/// IGNORES `truth_epoch`/`updated_at` entirely — two sets differing only by
/// epoch/timestamp are "unchanged" (the invariant P2-03 relies on to skip a
/// redundant re-upsert).
pub fn centroids_changed(
    old: &[TruthCentroidPayload],
    new: &[TruthCentroidPayload],
    tolerance: f64,
) -> bool {
    if old.len() != new.len() {
        return true;
    }

    let old_by_slot: HashMap<usize, &TruthCentroidPayload> = old
        .iter()
        .map(|centroid| (centroid.slot, centroid))
        .collect();

    for new_centroid in new {
        let Some(old_centroid) = old_by_slot.get(&new_centroid.slot) else {
            return true;
        };
        if old_centroid.count != new_centroid.count {
            return true;
        }
        if old_centroid.centroid.len() != new_centroid.centroid.len() {
            return true;
        }
        if old_centroid.learning_ids != new_centroid.learning_ids {
            return true;
        }
        for (&old_value, &new_value) in old_centroid
            .centroid
            .iter()
            .zip(new_centroid.centroid.iter())
        {
            if (old_value - new_value).abs() > tolerance {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        reason = "test code — panics are acceptable failures"
    )]
    use super::*;

    const TOL: f64 = 1e-9;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "expected {a} ≈ {b}");
    }

    // 1 — centroid_id deterministic and slot-sensitive (test_centroids.py:19-24).
    #[test]
    fn centroid_id_is_deterministic_and_slot_sensitive() {
        let first = centroid_id("dataset-1", 0);
        let second = centroid_id("dataset-1", 0);
        assert_eq!(first, second);
        assert_ne!(first, centroid_id("dataset-1", 1));
    }

    // 2 — learning_id normalizes whitespace and case (test_centroids.py:27-28).
    #[test]
    fn learning_id_normalizes_whitespace_and_case() {
        assert_eq!(
            learning_id(" Coffee  Matters "),
            learning_id("coffee matters")
        );
    }

    // 3 — normalize zero vector preserves shape (test_centroids.py:31-32).
    #[test]
    fn normalize_zero_vector_preserves_shape() {
        assert_eq!(normalize(&[0.0, 0.0, 0.0]), vec![0.0, 0.0, 0.0]);
    }

    // 4 — pad_coords truncate/pad (test_centroids.py:35-37).
    #[test]
    fn pad_coords_keeps_fixed_slot_count() {
        assert_eq!(pad_coords(&[0.1, 0.2], 4), vec![0.1, 0.2, 0.0, 0.0]);
        assert_eq!(pad_coords(&[0.1, 0.2, 0.3], 2), vec![0.1, 0.2]);
    }

    // 5 — weighted_centroid updates toward the new vector (test_centroids.py:40-42).
    #[test]
    fn weighted_centroid_updates_toward_new_vector() {
        let updated = weighted_centroid(&[1.0, 0.0], 1, &[0.0, 1.0]);
        approx(updated[0], updated[1]);
    }

    // 6 — unique_learning_vectors dedupes by normalized text (test_centroids.py:45-49).
    #[test]
    fn unique_learning_vectors_deduplicates_by_statement_text() {
        let statements = vec![
            "Coffee matters".to_string(),
            " coffee   matters ".to_string(),
            "Tea matters".to_string(),
        ];
        let vectors = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let pairs = unique_learning_vectors(&statements, &vectors);
        assert_eq!(pairs.len(), 2);
    }

    // 7 — build_centroids creates slots until the limit (test_centroids.py:53-65).
    #[test]
    fn build_centroids_creates_slots_until_limit() {
        let learning_vectors: Vec<(String, Vec<f64>)> = (0..10)
            .map(|index| (index.to_string(), vec![1.0, index as f64, 0.0]))
            .collect();
        let centroids =
            build_centroids_from_learning_vectors("dataset-1", &learning_vectors, 3, Some(123), 8);

        assert_eq!(
            centroids.iter().map(|c| c.slot).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        assert!(centroids.iter().all(|c| c.truth_epoch == 3));
        assert_eq!(centroids.iter().map(|c| c.count).sum::<usize>(), 10);
    }

    // 8 — determinism for same inputs / differing epoch (test_centroids.py:68-74).
    #[test]
    fn build_centroids_is_deterministic_for_same_inputs() {
        let learning_vectors: Vec<(String, Vec<f64>)> = (0..10)
            .map(|index| (index.to_string(), vec![1.0, index as f64, 0.0]))
            .collect();
        let first =
            build_centroids_from_learning_vectors("dataset-1", &learning_vectors, 1, Some(123), 8);
        let second =
            build_centroids_from_learning_vectors("dataset-1", &learning_vectors, 9, Some(123), 8);

        assert_eq!(
            first.iter().map(|c| c.centroid.clone()).collect::<Vec<_>>(),
            second
                .iter()
                .map(|c| c.centroid.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            first.iter().map(|c| c.count).collect::<Vec<_>>(),
            second.iter().map(|c| c.count).collect::<Vec<_>>()
        );
    }

    // 9 — extend adds new ids without double-counting (test_centroids.py:77-105).
    #[test]
    fn extend_centroids_adds_new_learning_ids_without_double_counting() {
        let existing = build_centroids_from_learning_vectors(
            "dataset-1",
            &[("a".to_string(), vec![1.0, 0.0])],
            1,
            Some(123),
            2,
        );

        let first = extend_centroids_with_learning_vectors(
            "dataset-1",
            &existing,
            &[
                ("a".to_string(), vec![1.0, 0.0]),
                ("b".to_string(), vec![0.0, 1.0]),
            ],
            2,
            Some(456),
            2,
        );
        let second = extend_centroids_with_learning_vectors(
            "dataset-1",
            &first,
            &[("b".to_string(), vec![0.0, 1.0])],
            3,
            Some(789),
            2,
        );

        assert_eq!(first.iter().map(|c| c.count).sum::<usize>(), 2);
        assert_eq!(
            first
                .iter()
                .map(|c| c.learning_ids.clone())
                .collect::<Vec<_>>(),
            vec![vec!["a".to_string()], vec!["b".to_string()]]
        );
        assert_eq!(
            second.iter().map(|c| c.count).collect::<Vec<_>>(),
            first.iter().map(|c| c.count).collect::<Vec<_>>()
        );
    }

    // 10 — centroids_changed ignores epoch-only changes / detects real changes
    // (test_centroids.py:108-125).
    #[test]
    fn centroids_changed_ignores_epoch_but_detects_real_changes() {
        let learning_vectors = vec![
            ("a".to_string(), vec![1.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0]),
        ];
        let old =
            build_centroids_from_learning_vectors("dataset-1", &learning_vectors, 1, Some(123), 8);
        let new =
            build_centroids_from_learning_vectors("dataset-1", &learning_vectors, 2, Some(123), 8);
        assert!(!centroids_changed(&old, &new, 1e-6));

        let old2 = build_centroids_from_learning_vectors(
            "dataset-1",
            &[("a".to_string(), vec![1.0, 0.0])],
            1,
            Some(123),
            8,
        );
        let new2 = build_centroids_from_learning_vectors(
            "dataset-1",
            &[
                ("a".to_string(), vec![1.0, 0.0]),
                ("b".to_string(), vec![0.0, 1.0]),
            ],
            1,
            Some(123),
            8,
        );
        assert!(centroids_changed(&old2, &new2, 1e-6));
    }

    // 11 — Rust-specific: an equal-cosine tie picks the LOWEST-index slot,
    // guarding the Python-first-wins vs naive-Rust-last-wins divergence.
    #[test]
    fn nearest_slot_tie_break_picks_lowest_index() {
        // k=2: slot 0 seeds from [1,0], slot 1 from [0,1]. A third vector [1,1]
        // has equal cosine (√½) to both normalized slot centroids -> first wins.
        let centroids = extend_centroids_with_learning_vectors(
            "dataset-1",
            &[],
            &[
                ("a".to_string(), vec![1.0, 0.0]),
                ("b".to_string(), vec![0.0, 1.0]),
                ("c".to_string(), vec![1.0, 1.0]),
            ],
            1,
            Some(123),
            2,
        );

        assert_eq!(centroids.len(), 2);
        // "c" must merge into slot 0 (lowest index), not slot 1.
        assert_eq!(
            centroids[0].learning_ids,
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(centroids[0].count, 2);
        assert_eq!(centroids[1].learning_ids, vec!["b".to_string()]);
        assert_eq!(centroids[1].count, 1);
    }

    // 12 — Rust-specific: TruthCentroidPayload serde round-trip, including an
    // omitted learning_ids field defaulting to an empty vec.
    #[test]
    fn truth_centroid_payload_serde_round_trip() {
        let payload = TruthCentroidPayload {
            dataset_id: "dataset-1".to_string(),
            slot: 3,
            count: 7,
            truth_epoch: 42,
            updated_at: 1_700_000_000_000,
            centroid: vec![0.1, -0.2, 0.3],
            learning_ids: vec!["id-1".to_string(), "id-2".to_string()],
        };
        let json = serde_json::to_string(&payload).expect("serialize payload");
        let restored: TruthCentroidPayload =
            serde_json::from_str(&json).expect("deserialize payload");
        assert_eq!(payload, restored);

        // Omitting learning_ids entirely deserializes to an empty vec.
        let no_ids =
            r#"{"dataset_id":"d","slot":0,"count":0,"truth_epoch":0,"updated_at":0,"centroid":[]}"#;
        let parsed: TruthCentroidPayload =
            serde_json::from_str(no_ids).expect("deserialize without learning_ids");
        assert!(parsed.learning_ids.is_empty());
    }
}
