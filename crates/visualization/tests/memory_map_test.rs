#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Integration tests for the Memory-tab payload builder.
//!
//! Mirrors `cognee/tests/unit/modules/visualization/test_memory_map_payload.py`.
//! Pure tests over `preprocess()` with synthetic graphs — no DB, no LLM. They
//! pin the contract the JS renderer builds on:
//!
//!   - `t_created` is preserved from `created_at` before the pop;
//!   - chunk cells order by `chunk_index`; legacy chunks (attributed only via
//!     the `is_part_of` edge) append after; unattributable chunks are orphans;
//!   - entity groups come from `is_a` → EntityType edges; the top-8 of each
//!     group are flagged `important`;
//!   - summaries carry `chunk_ids` from `made_from` edges;
//!   - `context` is `null` when no GlobalContextSummary nodes exist;
//!   - the timeline gap-clusters `t_created` into run events;
//!   - the whole payload is deterministic: same input twice → identical JSON,
//!     and node insertion order does not change any ordering.

use std::borrow::Cow;
use std::collections::HashMap;

use cognee_graph::{EdgeData, GraphNode};
use cognee_visualization::preprocess;
use serde_json::{Value, json};

/// `MEMORY_TIMELINE_GAP_MS` (`preprocessor.py:900`).
const GAP_MS: i64 = 300_000;
/// `MEMORY_GROUP_TOP_MEMBERS` (`preprocessor.py:905`).
const TOP_MEMBERS: usize = 8;

const T0: i64 = 1_768_164_683_000; // first run batch (epoch ms)
const T1: i64 = T0 + 2 * GAP_MS; // second run, 10 minutes later

fn node(id: &str, props: Value) -> GraphNode {
    let map: HashMap<Cow<'static, str>, Value> = props
        .as_object()
        .expect("node fixture is a JSON object")
        .iter()
        .map(|(key, value)| (Cow::Owned(key.clone()), value.clone()))
        .collect();
    (id.to_string(), map)
}

fn edge(source: &str, target: &str, relation: &str, props: Value) -> EdgeData {
    let map: HashMap<Cow<'static, str>, Value> = props
        .as_object()
        .expect("edge fixture is a JSON object")
        .iter()
        .map(|(key, value)| (Cow::Owned(key.clone()), value.clone()))
        .collect();
    (
        source.to_string(),
        target.to_string(),
        relation.to_string(),
        map,
    )
}

/// Two documents across two run batches, a legacy chunk attributed via
/// `is_part_of`, an orphan chunk, one entity group, one ungrouped entity and
/// one summary. Port of `_memory_graph()` (`test_memory_map_payload.py:32-82`).
fn memory_graph() -> (Vec<GraphNode>, Vec<EdgeData>) {
    let nodes = vec![
        node(
            "doc1",
            json!({"type": "TextDocument", "name": "alice.md", "created_at": T0}),
        ),
        node(
            "doc2",
            json!({"type": "TextDocument", "name": "bob.md", "created_at": T1}),
        ),
        node(
            "c1",
            json!({
                "type": "DocumentChunk",
                "text": "chunk one",
                "chunk_index": 1,
                "document_id": "doc1",
                "created_at": T0 + 100,
                "source_pipeline": "cognify_pipeline",
            }),
        ),
        node(
            "c0",
            json!({
                "type": "DocumentChunk",
                "text": "chunk zero",
                "chunk_index": 0,
                "document_id": "doc1",
                "created_at": T0 + 200,
                "source_pipeline": "cognify_pipeline",
            }),
        ),
        // Legacy chunk: no document_id, no created_at — attributed via edge.
        node(
            "c_legacy",
            json!({"type": "DocumentChunk", "text": "legacy chunk"}),
        ),
        // Orphan chunk: no document_id and no is_part_of edge.
        node(
            "c_orphan",
            json!({"type": "DocumentChunk", "text": "orphan chunk", "created_at": T1 + 50}),
        ),
        node(
            "alice",
            json!({"type": "Entity", "name": "Alice", "created_at": T0 + 300}),
        ),
        node(
            "bob",
            json!({"type": "Entity", "name": "Bob", "created_at": T0 + 300}),
        ),
        // Ungrouped entity: no is_a edge to an EntityType.
        node(
            "zed",
            json!({"type": "Entity", "name": "Zed", "created_at": T1 + 100}),
        ),
        node(
            "person",
            json!({"type": "EntityType", "name": "Person", "created_at": T0 + 250}),
        ),
        node(
            "sum1",
            json!({"type": "TextSummary", "text": "summary one", "created_at": T0 + 400}),
        ),
    ];
    let edges = vec![
        edge("c0", "doc1", "is_part_of", json!({})),
        edge("c_legacy", "doc1", "is_part_of", json!({})),
        edge("c1", "alice", "contains", json!({})),
        edge("c0", "bob", "contains", json!({})),
        edge("alice", "person", "is_a", json!({})),
        edge("bob", "person", "is_a", json!({})),
        edge(
            "alice",
            "bob",
            "knows",
            json!({"relationship_name": "knows"}),
        ),
        edge("sum1", "c0", "made_from", json!({})),
    ];
    (nodes, edges)
}

fn payload_of(graph: (Vec<GraphNode>, Vec<EdgeData>)) -> Value {
    let (nodes, edges) = graph;
    preprocess(nodes, edges, None).memory_map
}

fn payload() -> Value {
    payload_of(memory_graph())
}

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

fn array<'a>(value: &'a Value, key: &str) -> &'a Vec<Value> {
    field(value, key)
        .as_array()
        .unwrap_or_else(|| panic!("payload key {key} is an array"))
}

fn ids(list: &[Value]) -> Vec<&str> {
    list.iter()
        .map(|entry| entry.get("id").and_then(Value::as_str).unwrap_or(""))
        .collect()
}

// ── t_created preservation ───────────────────────────────────────────────────

#[test]
fn t_created_preserved_and_created_at_dropped() {
    let (nodes, edges) = memory_graph();
    let result = preprocess(nodes, edges, None);
    let by_id: HashMap<&str, &Value> = result
        .nodes
        .iter()
        .map(|node| (node.get("id").and_then(Value::as_str).unwrap_or(""), node))
        .collect();
    assert_eq!(field(by_id["doc1"], "t_created"), &json!(T0));
    assert_eq!(field(by_id["c1"], "t_created"), &json!(T0 + 100));
    for node in &result.nodes {
        assert!(node.get("created_at").is_none());
        assert!(node.get("updated_at").is_none());
    }
    // Legacy node without created_at: no t_created key fabricated.
    assert!(by_id["c_legacy"].get("t_created").is_none());
}

// ── Documents and chunk cells ────────────────────────────────────────────────

#[test]
fn documents_sorted_by_t_first_then_name() {
    let payload = payload();
    let documents = array(&payload, "documents");
    assert_eq!(ids(documents), vec!["doc1", "doc2"]);
    assert_eq!(field(&documents[0], "t_first"), &json!(T0));
    assert_eq!(field(&documents[1], "t_first"), &json!(T1));
}

#[test]
fn chunks_ordered_by_chunk_index_with_legacy_appended() {
    let payload = payload();
    let doc1 = &array(&payload, "documents")[0];
    let chunks = array(doc1, "chunks");
    // c0 (index 0) before c1 (index 1) despite a later t_created and a later
    // input position; the legacy chunk (no chunk_index) appends last.
    assert_eq!(ids(chunks), vec!["c0", "c1", "c_legacy"]);
    let indexes: Vec<&Value> = chunks.iter().map(|c| field(c, "chunk_index")).collect();
    assert_eq!(indexes, vec![&json!(0), &json!(1), &Value::Null]);
    assert_eq!(field(&chunks[0], "t_created"), &json!(T0 + 200));
    assert_eq!(field(&chunks[2], "t_created"), &Value::Null);
}

#[test]
fn orphan_chunks_collected() {
    assert_eq!(array(&payload(), "orphan_chunks"), &vec![json!("c_orphan")]);
}

#[test]
fn doc_to_chunk_contains_counts_as_membership() {
    // Graphs where the doc→chunk edge is `contains` (not is_part_of) still
    // attribute chunks to their document.
    let nodes = vec![
        node("d", json!({"type": "TextDocument", "name": "d.md"})),
        node("c", json!({"type": "DocumentChunk", "text": "x"})),
    ];
    let edges = vec![edge("d", "c", "contains", json!({}))];
    let payload = payload_of((nodes, edges));
    let documents = array(&payload, "documents");
    assert_eq!(
        field(&documents[0], "chunks"),
        &json!([{"id": "c", "chunk_index": null, "t_created": null}])
    );
    assert!(array(&payload, "orphan_chunks").is_empty());
    assert_eq!(field(field(&payload, "edges"), "is_part_of"), &json!([0]));
    assert_eq!(field(field(&payload, "edges"), "contains"), &json!([]));
}

// ── Entity groups ────────────────────────────────────────────────────────────

#[test]
fn entity_grouping_via_is_a_edges() {
    let payload = payload();
    let groups = array(&payload, "entity_groups");
    assert_eq!(groups.len(), 1);
    let group = &groups[0];
    assert_eq!(field(group, "type_id"), &json!("person"));
    assert_eq!(field(group, "type_name"), &json!("Person"));
    let members = array(group, "members");
    assert_eq!(ids(members), vec!["alice", "bob"]);
    // Small group: everyone is within the top-8 budget.
    assert!(
        members
            .iter()
            .all(|member| field(member, "important") == &json!(true))
    );
}

#[test]
fn ungrouped_entities_listed() {
    assert_eq!(array(&payload(), "ungrouped_entities"), &vec![json!("zed")]);
}

#[test]
fn group_top_members_flagged_important() {
    // 12 equal-importance members in one group, plus high-degree ballast so
    // the 75th-percentile label threshold sits above the members — making the
    // top-8 `important` cut observable.
    let mut nodes = vec![node("t", json!({"type": "EntityType", "name": "Person"}))];
    let mut edges = Vec::new();
    for index in 0..12 {
        nodes.push(node(
            &format!("e{index}"),
            json!({"type": "Entity", "name": format!("E{index:02}")}),
        ));
        edges.push(edge(&format!("e{index}"), "t", "is_a", json!({})));
    }
    for index in 0..40 {
        nodes.push(node(
            &format!("h{index}"),
            json!({"type": "Entity", "name": format!("H{index:02}")}),
        ));
    }
    for index in 0..40 {
        for step in 0..3 {
            edges.push(edge(
                &format!("h{index}"),
                &format!("h{}", (index + step + 1) % 40),
                "rel",
                json!({"relationship_name": "rel"}),
            ));
        }
    }

    let payload = payload_of((nodes, edges));
    let groups = array(&payload, "entity_groups");
    let members = array(&groups[0], "members");
    assert_eq!(members.len(), 12);
    // Equal importance, no label_priority → deterministic name order, the
    // first MEMORY_GROUP_TOP_MEMBERS important, the tail collapsible.
    let expected: Vec<String> = (0..12).map(|index| format!("e{index}")).collect();
    assert_eq!(ids(members), expected);
    let important: Vec<bool> = members
        .iter()
        .map(|member| field(member, "important") == &json!(true))
        .collect();
    let mut want = vec![true; TOP_MEMBERS];
    want.extend(vec![false; 12 - TOP_MEMBERS]);
    assert_eq!(important, want);
}

#[test]
fn entity_to_entity_is_a_does_not_group() {
    // is_a edges between two Entity nodes must not create a group.
    let nodes = vec![
        node("a", json!({"type": "Entity", "name": "A"})),
        node("b", json!({"type": "Entity", "name": "B"})),
    ];
    let edges = vec![edge("a", "b", "is_a", json!({}))];
    let payload = payload_of((nodes, edges));
    assert!(array(&payload, "entity_groups").is_empty());
    assert_eq!(
        array(&payload, "ungrouped_entities"),
        &vec![json!("a"), json!("b")]
    );
    // The edge is still structural, so it never lands in the semantic bucket.
    assert_eq!(field(field(&payload, "edges"), "semantic"), &json!([]));
}

// ── Summaries ────────────────────────────────────────────────────────────────

#[test]
fn summaries_carry_chunk_ids_from_made_from() {
    assert_eq!(
        field(&payload(), "summaries"),
        &json!([{"id": "sum1", "chunk_ids": ["c0"], "bucket_id": null}])
    );
}

#[test]
fn summaries_sorted_by_t_created() {
    let nodes = vec![
        node(
            "s_late",
            json!({"type": "TextSummary", "text": "later", "created_at": T0 + 500}),
        ),
        node(
            "s_early",
            json!({"type": "TextSummary", "text": "earlier", "created_at": T0 + 100}),
        ),
    ];
    let payload = payload_of((nodes, vec![]));
    assert_eq!(ids(array(&payload, "summaries")), vec!["s_early", "s_late"]);
}

// ── Global context ───────────────────────────────────────────────────────────

#[test]
fn context_is_null_without_global_context_nodes() {
    assert_eq!(field(&payload(), "context"), &Value::Null);
}

#[test]
fn context_built_from_summarized_in_edges() {
    let nodes = vec![
        node(
            "sum1",
            json!({"type": "TextSummary", "text": "s", "created_at": T0}),
        ),
        node(
            "b1",
            json!({"type": "GlobalContextSummary", "text": "bucket", "level": 0}),
        ),
        node(
            "root",
            json!({"type": "GlobalContextSummary", "text": "root", "level": 1, "is_root": true}),
        ),
    ];
    let edges = vec![
        edge("sum1", "b1", "summarized_in", json!({})),
        edge("b1", "root", "summarized_in", json!({})),
    ];
    let payload = payload_of((nodes, edges));
    let context = field(&payload, "context");
    assert_eq!(field(context, "root_id"), &json!("root"));
    let buckets = array(context, "buckets");
    // Sorted by level ascending.
    assert_eq!(ids(buckets), vec!["b1", "root"]);
    assert_eq!(field(&buckets[0], "level"), &json!(0));
    assert_eq!(field(&buckets[0], "child_ids"), &json!(["sum1"]));
    assert_eq!(field(&buckets[1], "child_ids"), &json!(["b1"]));
}

// ── Structural edge index ────────────────────────────────────────────────────

#[test]
fn edge_index_points_into_links_array() {
    let (nodes, edges) = memory_graph();
    let result = preprocess(nodes, edges, None);
    let index = field(&result.memory_map, "edges");
    assert_eq!(field(index, "is_part_of"), &json!([0, 1]));
    assert_eq!(field(index, "contains"), &json!([2, 3]));
    assert_eq!(field(index, "semantic"), &json!([6]));
    assert_eq!(field(index, "made_from"), &json!([7]));
    assert_eq!(field(index, "summarized_in"), &json!([]));
    // Positions must resolve to the right links.
    let position = field(index, "semantic").as_array().expect("array")[0]
        .as_u64()
        .expect("link position is a number") as usize;
    let knows = &result.links[position];
    assert_eq!(field(knows, "source"), &json!("alice"));
    assert_eq!(field(knows, "target"), &json!("bob"));
    assert_eq!(field(knows, "relation"), &json!("knows"));
}

// ── Timeline ─────────────────────────────────────────────────────────────────

#[test]
fn timeline_two_batches_yield_two_events() {
    let payload = payload();
    let timeline = array(&payload, "timeline");
    assert_eq!(timeline.len(), 2);
    let (first, second) = (&timeline[0], &timeline[1]);
    assert_eq!(field(first, "index"), &json!(0));
    assert_eq!(field(second, "index"), &json!(1));
    assert_eq!(field(first, "kind"), &json!("run"));
    assert_eq!(field(second, "kind"), &json!("run"));
    assert_eq!(field(first, "t0"), &json!(T0));
    assert_eq!(field(first, "t1"), &json!(T0 + 400));
    assert_eq!(field(second, "t0"), &json!(T1));
    assert_eq!(field(second, "t1"), &json!(T1 + 100));
    // Batch 1: doc1, c1, c0, alice, bob, person, sum1 + untimed c_legacy.
    assert_eq!(field(first, "node_count"), &json!(8));
    let first_ids: Vec<&str> = field(first, "node_ids")
        .as_array()
        .expect("node_ids is an array")
        .iter()
        .map(|id| id.as_str().unwrap_or(""))
        .collect();
    // Untimed nodes join event 0, prepended.
    assert_eq!(first_ids[0], "c_legacy");
    let mut second_ids: Vec<&str> = field(second, "node_ids")
        .as_array()
        .expect("node_ids is an array")
        .iter()
        .map(|id| id.as_str().unwrap_or(""))
        .collect();
    second_ids.sort_unstable();
    assert_eq!(second_ids, vec!["c_orphan", "doc2", "zed"]);
    assert_eq!(field(second, "node_count"), &json!(3));
}

#[test]
fn timeline_labels_majority_pipeline_with_ingestion_fallback() {
    let payload = payload();
    let timeline = array(&payload, "timeline");
    assert_eq!(field(&timeline[0], "label"), &json!("cognify_pipeline"));
    assert_eq!(field(&timeline[1], "label"), &json!("ingestion"));
}

#[test]
fn timeline_label_refines_to_global_context_index() {
    let nodes = vec![
        node(
            "c",
            json!({
                "type": "DocumentChunk",
                "text": "x",
                "created_at": T0,
                "source_pipeline": "cognify_pipeline",
            }),
        ),
        node(
            "g",
            json!({"type": "GlobalContextSummary", "text": "ctx", "created_at": T0 + 10}),
        ),
    ];
    let payload = payload_of((nodes, vec![]));
    let timeline = array(&payload, "timeline");
    assert_eq!(timeline.len(), 1);
    assert_eq!(field(&timeline[0], "label"), &json!("global_context_index"));
}

#[test]
fn timeline_same_batch_merges_into_one_event() {
    // Consecutive gaps equal the threshold — chained into a single cluster
    // because each gap is measured against the previous member.
    let nodes = vec![
        node(
            "a",
            json!({"type": "Entity", "name": "A", "created_at": T0}),
        ),
        node(
            "b",
            json!({"type": "Entity", "name": "B", "created_at": T0 + GAP_MS}),
        ),
        node(
            "c",
            json!({"type": "Entity", "name": "C", "created_at": T0 + 2 * GAP_MS}),
        ),
    ];
    let payload = payload_of((nodes, vec![]));
    let timeline = array(&payload, "timeline");
    assert_eq!(timeline.len(), 1);
    assert_eq!(field(&timeline[0], "node_count"), &json!(3));
    assert_eq!(field(&timeline[0], "t0"), &json!(T0));
    assert_eq!(field(&timeline[0], "t1"), &json!(T0 + 2 * GAP_MS));
}

#[test]
fn timeline_splits_when_the_gap_is_exceeded() {
    let nodes = vec![
        node(
            "a",
            json!({"type": "Entity", "name": "A", "created_at": T0}),
        ),
        node(
            "b",
            json!({"type": "Entity", "name": "B", "created_at": T0 + GAP_MS + 1}),
        ),
    ];
    let payload = payload_of((nodes, vec![]));
    assert_eq!(array(&payload, "timeline").len(), 2);
}

#[test]
fn timeline_without_any_timestamps_emits_one_synthetic_event() {
    let nodes = vec![
        node("a", json!({"type": "Entity", "name": "A"})),
        node("b", json!({"type": "Entity", "name": "B"})),
    ];
    let payload = payload_of((nodes, vec![]));
    let timeline = array(&payload, "timeline");
    assert_eq!(timeline.len(), 1);
    assert_eq!(field(&timeline[0], "label"), &json!("ingestion"));
    assert_eq!(field(&timeline[0], "t0"), &json!(0));
    assert_eq!(field(&timeline[0], "t1"), &json!(0));
    assert_eq!(field(&timeline[0], "node_ids"), &json!(["a", "b"]));
}

// ── Empty graph and determinism ──────────────────────────────────────────────

#[test]
fn empty_graph_payload_has_all_keys_empty() {
    assert_eq!(
        payload_of((vec![], vec![])),
        json!({
            "documents": [],
            "orphan_chunks": [],
            "entity_groups": [],
            "ungrouped_entities": [],
            "summaries": [],
            "context": null,
            "edges": {
                "contains": [],
                "made_from": [],
                "is_part_of": [],
                "summarized_in": [],
                "semantic": [],
            },
            "timeline": [],
        })
    );
}

#[test]
fn payload_is_deterministic_across_runs() {
    let first = serde_json::to_string(&payload()).expect("payload serializes");
    let second = serde_json::to_string(&payload()).expect("payload serializes");
    assert_eq!(first, second);
}

#[test]
fn payload_ordering_independent_of_node_input_order() {
    // Reversing node insertion order must not change any ordering — every sort
    // key is intrinsic to the data. (Edge order is kept fixed because the
    // `edges` index stores positions into the links array.)
    let (nodes, edges) = memory_graph();
    let baseline = payload_of((nodes.clone(), edges.clone()));
    let reversed_nodes: Vec<GraphNode> = nodes.into_iter().rev().collect();
    let reversed = payload_of((reversed_nodes, edges));
    assert_eq!(
        serde_json::to_string(&baseline).expect("payload serializes"),
        serde_json::to_string(&reversed).expect("payload serializes")
    );
}
