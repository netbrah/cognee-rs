//! Section formatting and used-id extraction for the hybrid retriever.
//!
//! Port of `cognee/modules/retrieval/hybrid/context.py`. Renders the flat
//! `Vec<SearchItem>` context the [`super::HybridRetriever`] assembles into the
//! sectioned markdown Python feeds the LLM, and re-derives the graph node ids a
//! hybrid answer drew on (facts excluded — see [`extract_used_ids`]).
//!
//! `format_entities` / `format_facts` are **not** re-implemented here: this
//! module reuses the already-landed ones from [`super::entities`] /
//! [`super::facts`]. `format_hybrid_context` therefore takes the
//! already-rendered passage/entity/fact section strings and only handles the
//! section ordering + join.

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use super::results::{display_value, payload, result_id};
use crate::types::SearchItem;

/// Render the ranked chunks as the "Relevant passages" markdown section.
///
/// Port of `format_passages` (`context.py:61-77`). Each chunk contributes its
/// raw `text` (skipped when blank); when a nonempty summary is paired to the
/// chunk (looked up by [`result_id`] in `chunk_summaries`) the entry becomes
/// `"[Passage Summary]: {summary}\n[Raw Passage]: {text}"`, otherwise the raw
/// text alone. Entries join with `"\n---\n"` under a `"## Relevant passages"`
/// header; an empty result yields `""`.
pub(crate) fn format_passages(
    chunks: &[SearchItem],
    chunk_summaries: &HashMap<String, String>,
) -> String {
    let mut texts: Vec<String> = Vec::new();
    for chunk in chunks {
        let Some(text) = payload(chunk).get("text").and_then(display_value) else {
            continue;
        };

        let summary = result_id(chunk).and_then(|id| chunk_summaries.get(&id));
        match summary {
            Some(summary) if !summary.is_empty() => {
                texts.push(format!(
                    "[Passage Summary]: {summary}\n[Raw Passage]: {text}"
                ));
            }
            _ => texts.push(text),
        }
    }

    if texts.is_empty() {
        return String::new();
    }
    format!("## Relevant passages\n{}", texts.join("\n---\n"))
}

/// Join the non-empty context sections into the final prompt context.
///
/// Port of `format_hybrid_context` (`context.py:8-30`). Pushes each non-empty
/// section in the order global → passages → entities → facts and joins them
/// with `"\n\n"`. The `global_context` slot is always `None` in Phase 1 (the
/// global-context index is unsupported), but the parameter is kept so the
/// signature is stable when Phase 3 lands it.
pub(crate) fn format_hybrid_context(
    global_context: Option<&str>,
    passages: &str,
    entities: &str,
    facts: &str,
) -> String {
    let mut sections: Vec<&str> = Vec::new();
    if let Some(global_context) = global_context
        && !global_context.is_empty()
    {
        sections.push(global_context);
    }
    if !passages.is_empty() {
        sections.push(passages);
    }
    if !entities.is_empty() {
        sections.push(entities);
    }
    if !facts.is_empty() {
        sections.push(facts);
    }
    sections.join("\n\n")
}

/// Collect the graph node ids a hybrid context drew on, **excluding facts**.
///
/// Port of `extract_context_object_ids` (`context.py:33-58`). Facts are
/// intentionally skipped: their ids are `EdgeType` vector rows, not graph nodes.
/// Walks the tagged context by its `"kind"` payload discriminator — chunk ids
/// via [`result_id`], entity ids plus each entity edge's `source_id`/`target_id`
/// — and returns them sorted and deduplicated. Orchestrator consumption
/// (`build_used_graph_element_ids`) is P1-10; this helper only provides the
/// stable extraction the later task wires in.
#[allow(
    dead_code,
    reason = "consumed by the orchestrator's used_graph_element_ids wiring in P1-10; exercised now by this module's tests"
)]
pub(crate) fn extract_used_ids(context: &[SearchItem]) -> Vec<String> {
    let mut node_ids: BTreeSet<String> = BTreeSet::new();

    for item in context {
        match item.payload.get("kind").and_then(Value::as_str) {
            Some("chunk") => {
                if let Some(id) = result_id(item) {
                    node_ids.insert(id);
                }
            }
            Some("entity") => {
                if let Some(id) = item.payload.get("id").and_then(display_value) {
                    node_ids.insert(id);
                }
                if let Some(Value::Array(edges)) = item.payload.get("edges") {
                    for edge in edges {
                        for key in ["source_id", "target_id"] {
                            if let Some(node_id) = edge.get(key).and_then(display_value) {
                                node_ids.insert(node_id);
                            }
                        }
                    }
                }
            }
            // Facts (and any unknown kind) contribute no graph node ids.
            _ => {}
        }
    }

    node_ids.into_iter().collect()
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

    fn chunk_item(id: &str, text: &str) -> SearchItem {
        SearchItem {
            id: None,
            score: None,
            payload: json!({ "kind": "chunk", "id": id, "text": text }),
        }
    }

    #[test]
    fn format_passages_summary_vs_raw_branch() {
        let chunks = vec![chunk_item("c1", "raw one"), chunk_item("c2", "raw two")];
        let mut summaries = HashMap::new();
        summaries.insert("c1".to_string(), "summary one".to_string());

        let rendered = format_passages(&chunks, &summaries);
        assert_eq!(
            rendered,
            "## Relevant passages\n[Passage Summary]: summary one\n[Raw Passage]: raw one\n---\nraw two"
        );
    }

    #[test]
    fn format_passages_empty_when_no_texts() {
        let chunks = vec![SearchItem {
            id: None,
            score: None,
            payload: json!({ "kind": "chunk", "id": "c1" }),
        }];
        assert_eq!(format_passages(&chunks, &HashMap::new()), "");
        assert_eq!(format_passages(&[], &HashMap::new()), "");
    }

    #[test]
    fn format_passages_blank_summary_falls_back_to_raw() {
        let chunks = vec![chunk_item("c1", "raw one")];
        let mut summaries = HashMap::new();
        summaries.insert("c1".to_string(), String::new());
        assert_eq!(
            format_passages(&chunks, &summaries),
            "## Relevant passages\nraw one"
        );
    }

    #[test]
    fn format_hybrid_context_orders_and_omits_sections() {
        // All four present -> global, passages, entities, facts in order.
        assert_eq!(
            format_hybrid_context(Some("## Global context\nG"), "P", "E", "F"),
            "## Global context\nG\n\nP\n\nE\n\nF"
        );

        // Some sections empty -> omitted, not joined as empty strings.
        assert_eq!(format_hybrid_context(None, "P", "", "F"), "P\n\nF");

        // All empty -> "".
        assert_eq!(format_hybrid_context(None, "", "", ""), "");

        // Empty global string is omitted just like a missing one.
        assert_eq!(format_hybrid_context(Some(""), "P", "", ""), "P");
    }

    #[test]
    fn extract_used_ids_excludes_facts_and_walks_edges() {
        let context = vec![
            chunk_item("chunk-1", "chunk text"),
            SearchItem {
                id: None,
                score: None,
                payload: json!({
                    "kind": "entity",
                    "id": "entity-1",
                    "name": "Alice",
                    "edges": [
                        { "source_id": "entity-1", "target_id": "acme-id" },
                        { "source_id": "entity-1", "target_id": "tennis-id" }
                    ]
                }),
            },
            SearchItem {
                id: None,
                score: None,
                payload: json!({ "kind": "fact", "id": "fact-1", "text": "Acme acquired Initech." }),
            },
        ];

        let ids = extract_used_ids(&context);
        assert!(ids.contains(&"chunk-1".to_string()));
        assert!(ids.contains(&"entity-1".to_string()));
        assert!(ids.contains(&"acme-id".to_string()));
        assert!(ids.contains(&"tennis-id".to_string()));
        // Fact ids are never contributed as graph node ids.
        assert!(!ids.contains(&"fact-1".to_string()));
        // Sorted and deduplicated.
        assert_eq!(
            ids,
            vec!["acme-id", "chunk-1", "entity-1", "tennis-id"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }
}
