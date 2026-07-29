//! Server-side `belongs_to_set` (NodeSet) membership predicate for vector search.
//!
//! This is the vector-crate mirror of the search crate's
//! `crate::retrievers::hybrid::results::payload_matches_node_filter`. The two
//! must stay byte-for-byte semantically identical: an adapter that filters
//! server-side with [`metadata_matches_node_filter`] returns exactly the same
//! in-set rows the search crate would have kept client-side — just without ever
//! dropping an in-set row that was crowded out of an over-fetch window (finding
//! F9). The duplication is unavoidable because `cognee-search` depends on
//! `cognee-vector`, not the other way round.
//!
//! # Semantics (identical to `payload_matches_node_filter`)
//! * `node_name` `None` or empty → always matches (no filter requested).
//! * `belongs_to_set` absent or not a JSON array → never matches.
//! * Each `belongs_to_set` entry contributes a set-name: a bare JSON string is
//!   the name as-is; a JSON object contributes its `"name"` field when that is
//!   a string (skipped otherwise); any other entry shape is skipped. So a
//!   bare-string dataset-id entry only matches a request for that literal string
//!   — never a NodeSet name via an object's `"name"`.
//! * Operator `"AND"` → requested set must be a subset of the payload's set
//!   (every requested name present). Anything else (including garbage) → `"OR"`
//!   → a non-empty intersection.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

/// Metadata key holding the NodeSet / dataset membership array.
pub(crate) const BELONGS_TO_SET_KEY: &str = "belongs_to_set";

/// Whether `metadata` satisfies the requested node-set filter.
///
/// See the module docs for the exact semantics; this is the mirror of the
/// search crate's `payload_matches_node_filter`, operating on a
/// [`crate::models::SearchResult`]'s `metadata` map (which serializes into that
/// item's `payload`, so the two produce identical verdicts on the same row).
pub(crate) fn metadata_matches_node_filter(
    metadata: &HashMap<String, Value>,
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
) -> bool {
    let requested: &[String] = match node_name {
        Some(names) if !names.is_empty() => names,
        _ => return true,
    };

    let Some(Value::Array(entries)) = metadata.get(BELONGS_TO_SET_KEY) else {
        return false;
    };

    let payload_sets: HashSet<&str> = entries
        .iter()
        .filter_map(|entry| match entry {
            Value::String(name) => Some(name.as_str()),
            Value::Object(_) => entry.get("name").and_then(Value::as_str),
            _ => None,
        })
        .collect();

    let requested_sets: HashSet<&str> = requested.iter().map(String::as_str).collect();

    if node_name_filter_operator == "AND" {
        requested_sets.is_subset(&payload_sets)
    } else {
        !payload_sets.is_disjoint(&requested_sets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(belongs_to_set: Value) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(BELONGS_TO_SET_KEY.to_string(), belongs_to_set);
        m
    }

    #[test]
    fn none_or_empty_always_matches() {
        let m = meta(json!(["a"]));
        assert!(metadata_matches_node_filter(&m, None, "OR"));
        assert!(metadata_matches_node_filter(&m, Some(&[]), "AND"));
    }

    #[test]
    fn missing_set_fails_when_requested() {
        let m: HashMap<String, Value> = HashMap::new();
        assert!(!metadata_matches_node_filter(
            &m,
            Some(&["a".to_string()]),
            "OR"
        ));
    }

    #[test]
    fn bare_string_entries() {
        let m = meta(json!(["alpha", "beta"]));
        assert!(metadata_matches_node_filter(
            &m,
            Some(&["beta".to_string()]),
            "OR"
        ));
        assert!(!metadata_matches_node_filter(
            &m,
            Some(&["gamma".to_string()]),
            "OR"
        ));
    }

    #[test]
    fn object_entries_read_name() {
        let m = meta(json!([
            {"id": "1", "name": "alpha", "type": "NodeSet"},
            {"id": "2", "name": "beta", "type": "NodeSet"}
        ]));
        assert!(metadata_matches_node_filter(
            &m,
            Some(&["alpha".to_string()]),
            "OR"
        ));
        // An object entry without a usable `name` is skipped — its `id` must
        // NOT be treated as a set name.
        let no_name = meta(json!([{"id": "1", "type": "NodeSet"}]));
        assert!(!metadata_matches_node_filter(
            &no_name,
            Some(&["1".to_string()]),
            "OR"
        ));
    }

    #[test]
    fn bare_string_dataset_id_does_not_match_nodeset_name() {
        // A dataset-id bare string entry only matches a request for that exact
        // string, never a NodeSet name reached through an object's `name`.
        let m = meta(json!(["dataset-uuid-1234"]));
        assert!(!metadata_matches_node_filter(
            &m,
            Some(&["my_nodeset".to_string()]),
            "OR"
        ));
        assert!(metadata_matches_node_filter(
            &m,
            Some(&["dataset-uuid-1234".to_string()]),
            "OR"
        ));
    }

    #[test]
    fn and_requires_subset_or_requires_overlap() {
        let m = meta(json!(["alpha", "beta", "gamma"]));
        assert!(metadata_matches_node_filter(
            &m,
            Some(&["alpha".to_string(), "beta".to_string()]),
            "AND"
        ));
        assert!(!metadata_matches_node_filter(
            &m,
            Some(&["alpha".to_string(), "delta".to_string()]),
            "AND"
        ));
        // Any non-"AND" operator behaves as OR.
        assert!(metadata_matches_node_filter(
            &m,
            Some(&["delta".to_string(), "gamma".to_string()]),
            "garbage"
        ));
    }
}
