//! Pure truth-subspace alignment functions.
//!
//! No I/O, no database access, no LLM calls — just deterministic math over
//! plain slices. Everything here is NEUTRAL when inputs are
//! missing/empty/zero: [`truth_score`] returns `0.5` and [`truth_factor`]
//! returns `1.0` so callers that pass nothing leave baseline scoring
//! untouched. Ported 1:1 from Python `cognee/modules/truth_subspace/align.py`.

use sha2::Digest;

/// Cosine similarity of two vectors. Returns `0.0` for a zero/empty vector.
///
/// Mirrors the Python `zip` truncation semantics: if the two slices differ in
/// length, iteration stops at the shorter one (no padding, no error). Uses an
/// exact `== 0.0` norm check (no epsilon) for parity with the Python source.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Project `node_vec` onto each basis vector using cosine similarity.
///
/// The result is zero-padded to `basis_vecs.len()` so the coordinate vector
/// always has one entry per basis vector.
pub fn node_coords(node_vec: &[f64], basis_vecs: &[Vec<f64>]) -> Vec<f64> {
    let mut coords: Vec<f64> = basis_vecs
        .iter()
        .map(|basis_vec| cosine(node_vec, basis_vec))
        .collect();
    // `cosine` already yields 0.0 per vector, so `coords.len() == basis_vecs.len()`
    // always holds and this loop is dead code. Ported from the Python source and
    // kept for literal parity — it documents the "one coord per basis vector"
    // contract explicitly.
    while coords.len() < basis_vecs.len() {
        coords.push(0.0);
    }
    coords
}

/// Project a query vector onto each basis vector, zero-padded.
///
/// Pure passthrough to [`node_coords`] — identical projection math.
pub fn query_coords(q_vec: &[f64], basis_vecs: &[Vec<f64>]) -> Vec<f64> {
    node_coords(q_vec, basis_vecs)
}

/// Truth score in `[0, 1]`: the node's alignment with directions the query
/// cares about.
///
/// A query-relevance-weighted average of the node's per-direction alignments,
/// using the (clamped) query coordinates as weights. This is
/// magnitude-sensitive on purpose: a node strongly aligned with those
/// directions scores higher. Cosine of the two coord vectors does NOT work
/// here — every basis cosine is non-negative, so all coord vectors share one
/// octant and their cosine collapses to ~1 regardless of magnitude, erasing
/// the very signal we rank on.
///
/// Returns `0.5` (NEUTRAL) when either coord vector is empty, or when the
/// query aligns with no direction (no positive weight to spread).
pub fn truth_score(node_coords: &[f64], q_coords: &[f64]) -> f64 {
    if node_coords.is_empty() || q_coords.is_empty() {
        return 0.5;
    }

    let weights: Vec<f64> = q_coords.iter().map(|&q| q.max(0.0)).collect();
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.5;
    }

    let weighted: f64 = node_coords
        .iter()
        .zip(weights.iter())
        .map(|(&n, &w)| n * w)
        .sum();
    (weighted / total_weight).clamp(0.0, 1.0)
}

/// Multiplicative score factor in `[0.75, 1.25]`.
///
/// `0.75 + 0.5 * truth_score`. Returns `1.0` (NEUTRAL) when coords are
/// missing/zero, since [`truth_score`] is `0.5` there.
pub fn truth_factor(node_coords: &[f64], q_coords: &[f64]) -> f64 {
    0.75 + 0.5 * truth_score(node_coords, q_coords)
}

/// Stable sha256 hex signature of an ordered id sequence.
///
/// Joins the `Display` form of each id with `"|"` (order-sensitive by
/// construction) and returns the lowercase sha256 hex digest of the UTF-8
/// bytes. `Uuid::to_string()` matches Python `str(uuid.UUID(...))`, so
/// signatures over identical ordered id lists are stable across runs and SDKs.
pub fn stable_signature<T: std::fmt::Display>(ordered_ids: &[T]) -> String {
    let joined = ordered_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join("|");
    let result = sha2::Sha256::digest(joined.as_bytes());
    format!("{result:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {a} ≈ {b}");
    }

    // ---- cosine ----

    #[test]
    fn cosine_identical_vectors() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_opposite_vectors() {
        assert_eq!(cosine(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
    }

    #[test]
    fn cosine_scale_invariant() {
        approx(cosine(&[1.0, 1.0], &[2.0, 2.0]), 1.0);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[1.0, 1.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn cosine_empty_vector_returns_zero() {
        assert_eq!(cosine(&[], &[1.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[]), 0.0);
    }

    // ---- node_coords / query_coords ----

    #[test]
    fn node_coords_per_basis_vector() {
        let basis = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let coords = node_coords(&[1.0, 0.0], &basis);
        assert_eq!(coords, vec![1.0, 0.0]);
    }

    #[test]
    fn node_coords_zero_pad_to_basis_count() {
        let basis = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let coords = node_coords(&[0.0, 0.0], &basis);
        assert_eq!(coords.len(), basis.len());
        assert_eq!(coords, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn query_coords_matches_node_coords() {
        let basis = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            query_coords(&[0.0, 1.0], &basis),
            node_coords(&[0.0, 1.0], &basis)
        );
    }

    // ---- truth_factor ----

    #[test]
    fn truth_factor_neutral_when_coords_missing() {
        // NEUTRAL: empty coords => factor exactly 1.0
        assert_eq!(truth_factor(&[], &[]), 1.0);
        assert_eq!(truth_factor(&[0.1], &[]), 1.0);
        assert_eq!(truth_factor(&[], &[0.1]), 1.0);
    }

    #[test]
    fn truth_factor_neutral_when_coords_zero() {
        // Zero coord vectors => cosine 0.0 => score 0.5 => factor 1.0
        assert_eq!(truth_factor(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
    }

    // ---- truth_score ----

    #[test]
    fn truth_score_neutral_cases() {
        // NEUTRAL (0.5): empty coords, or a query with no positive weight.
        assert_eq!(truth_score(&[], &[]), 0.5);
        assert_eq!(truth_score(&[0.0, 0.0], &[0.0, 0.0]), 0.5);
        assert_eq!(truth_score(&[1.0, 1.0], &[0.0, 0.0]), 0.5);
        // Negative query coords contribute no weight -> still neutral.
        assert_eq!(truth_score(&[1.0, 1.0], &[-1.0, 0.0]), 0.5);
    }

    #[test]
    fn truth_score_weighted_alignment() {
        // Query-relevance-weighted average of the node's per-direction
        // alignment. Only the first direction has weight, so its coord wins.
        approx(truth_score(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        approx(truth_score(&[0.5, 0.0], &[1.0, 0.0]), 0.5);
        // Equal weights -> plain mean of the node coords.
        approx(truth_score(&[0.2, 0.8], &[0.5, 0.5]), 0.5);
    }

    #[test]
    fn truth_score_is_magnitude_sensitive() {
        // The whole point: a node aligned MORE strongly with the directions
        // scores higher, even though both point the same direction (cosine
        // would call them equal).
        let q = [0.3, 0.3];
        assert!(truth_score(&[0.4, 0.4], &q) > truth_score(&[0.2, 0.2], &q));
    }

    #[test]
    fn truth_factor_within_bounds() {
        let cases: [(&[f64], &[f64]); 4] = [
            (&[1.0, 0.0], &[1.0, 0.0]),
            (&[0.0, 0.0], &[1.0, 1.0]),
            (&[1.0, 1.0], &[1.0, 0.0]),
            (&[0.3, 0.7, 0.1], &[0.2, 0.9, 0.4]),
        ];
        for (nc, qc) in cases {
            let factor = truth_factor(nc, qc);
            assert!(
                (0.75..=1.25).contains(&factor),
                "factor {factor} out of bounds"
            );
        }

        // Extremes hit the bounds exactly.
        approx(truth_factor(&[1.0, 0.0], &[1.0, 0.0]), 1.25);
        approx(truth_factor(&[0.0, 0.0], &[1.0, 1.0]), 0.75);
    }

    // ---- stable_signature ----

    #[test]
    fn stable_signature_stability() {
        let ids = ["a", "b", "c"];
        let sig1 = stable_signature(&ids);
        let sig2 = stable_signature(&ids);
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), 64); // sha256 hex digest
    }

    #[test]
    fn stable_signature_order_sensitive() {
        assert_ne!(stable_signature(&["a", "b"]), stable_signature(&["b", "a"]));
    }

    // Rust-only: the generic `T: Display` path over `&[Uuid]` matches hashing
    // the manually "|"-joined `to_string()` values directly. Proves the
    // concrete type Phase-2 retriever code will actually pass behaves as
    // intended.
    #[test]
    fn stable_signature_uuid_matches_manual_join() {
        use uuid::Uuid;
        let ids = [
            Uuid::from_u128(0x3fa85f64_5717_4562_b3fc_2c963f66afa6),
            Uuid::from_u128(0x00112233_4455_6677_8899_aabbccddeeff),
        ];
        let manual = format!("{}|{}", ids[0], ids[1]);
        let result = sha2::Sha256::digest(manual.as_bytes());
        let expected = format!("{result:x}");
        assert_eq!(stable_signature(&ids), expected);
    }
}
