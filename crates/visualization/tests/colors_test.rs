#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Integration-level assertions for the color contract. The color helpers are
//! module-private, so they are exercised through `preprocess()`, which is where
//! Python applies them (`preprocessor.py:1242-1246`, `1367-1377`).

use std::borrow::Cow;
use std::collections::HashMap;

use cognee_graph::GraphNode;
use cognee_visualization::preprocess;
use serde_json::{Value, json};

fn node(id: &str, props: Value) -> GraphNode {
    let map: HashMap<Cow<'static, str>, Value> = props
        .as_object()
        .expect("node fixture is a JSON object")
        .iter()
        .map(|(key, value)| (Cow::Owned(key.clone()), value.clone()))
        .collect();
    (id.to_string(), map)
}

/// The `color` the preprocessor stamped on the single node of a one-node graph.
fn color_of(props: Value) -> String {
    let result = preprocess(vec![node("n1", props)], vec![], None);
    result.nodes[0]
        .get("color")
        .and_then(Value::as_str)
        .expect("every node carries a string color")
        .to_string()
}

#[test]
fn canonical_type_colors() {
    // Pinned so a palette change cannot silently break the visual encoding.
    assert_eq!(color_of(json!({"type": "Entity"})), "#6510F4");
    assert_eq!(color_of(json!({"type": "EntityType"})), "#D5C2FF");
    assert_eq!(color_of(json!({"type": "DocumentChunk"})), "#0DFF00");
    assert_eq!(color_of(json!({"type": "TextDocument"})), "#A550FF");
    assert_eq!(color_of(json!({"type": "TextSummary"})), "#FFB454");
    assert_eq!(color_of(json!({"type": "GlobalContextSummary"})), "#00C2FF");
    // NodeSet used to be missing from the map and fell through to gray.
    assert_eq!(color_of(json!({"type": "NodeSet"})), "#94A3B8");
}

#[test]
fn ontology_valid_uses_its_own_pink_not_the_unknown_gray() {
    // `_ONTOLOGY_VALID_COLOR` (#FF5CA8) must differ from the #DBD8D8
    // unknown-type fallback, otherwise ontology matches disappear visually.
    assert_eq!(
        color_of(json!({"type": "Entity", "ontology_valid": true})),
        "#FF5CA8"
    );
    assert_eq!(
        color_of(json!({"type": "Mystery", "ontology_valid": true})),
        "#FF5CA8"
    );
    assert_ne!(color_of(json!({"type": "Mystery"})), "#FF5CA8");
    // Only a literal JSON `true` triggers the override (Python's `is True`).
    assert_eq!(
        color_of(json!({"type": "Entity", "ontology_valid": "yes"})),
        "#6510F4"
    );
}

#[test]
fn unknown_and_absent_types_are_distinguished() {
    assert_eq!(color_of(json!({"type": "Mystery"})), "#DBD8D8");
    assert_eq!(color_of(json!({"type": null})), "#DBD8D8");
    // The "default" entry is only reachable when the key is missing entirely.
    assert_eq!(color_of(json!({"name": "no type here"})), "#7c3aed");
}

#[test]
fn provenance_color_maps_are_deterministic_and_sorted() {
    let nodes = vec![
        node(
            "n1",
            json!({"type": "Entity", "source_task": "task-b", "source_user": "alice"}),
        ),
        node(
            "n2",
            json!({"type": "Entity", "source_task": "task-a", "source_user": "alice"}),
        ),
    ];
    let result = preprocess(nodes, vec![], None);
    let tasks: Vec<&String> = result.color_maps.task.keys().collect();
    assert_eq!(tasks, vec!["task-a", "task-b"]);
    // Golden-angle rotation over the sorted unique values.
    assert_eq!(
        result.color_maps.task.get("task-a"),
        Some(&"#db5656".to_string())
    );
    assert_eq!(
        result.color_maps.task.get("task-b"),
        Some(&"#56db7d".to_string())
    );
    assert_eq!(result.color_maps.user.len(), 1);
    assert!(result.color_maps.pipeline.is_empty());
    assert!(result.color_maps.node_set.is_empty());
}

#[test]
fn memory_node_sets_override_the_hue_rotation() {
    let nodes = vec![
        node(
            "a",
            json!({"type": "Entity", "source_node_set": "session_learnings"}),
        ),
        node(
            "b",
            json!({"type": "Entity", "source_node_set": "user_sessions_from_cache"}),
        ),
        node(
            "c",
            json!({"type": "Entity", "source_node_set": "agent_trace_feedbacks"}),
        ),
        node("d", json!({"type": "Entity", "source_node_set": "adhoc"})),
    ];
    let result = preprocess(nodes, vec![], None);
    for (name, color) in [
        ("session_learnings", "#FFC53D"),
        ("user_sessions_from_cache", "#00C2AA"),
        ("agent_trace_feedbacks", "#FF7A59"),
    ] {
        assert_eq!(
            result.color_maps.node_set.get(name),
            Some(&color.to_string()),
            "node set {name}"
        );
    }
    // Unpinned sets keep their generated hue.
    let adhoc = result
        .color_maps
        .node_set
        .get("adhoc")
        .expect("adhoc set is present");
    assert!(adhoc.starts_with('#') && adhoc.len() == 7);
}
