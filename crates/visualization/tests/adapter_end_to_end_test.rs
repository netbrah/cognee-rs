#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! End-to-end: a **real graph adapter** → `preprocess()` → the Memory payload.
//!
//! Every other test in this crate feeds `preprocess()` hand-built `GraphNode`
//! maps, which is why the Memory timeline could pass its unit tests while being
//! dead on real data: `t_created` derives from a `created_at` property that no
//! production adapter returned. A hand-built fixture cannot catch that, and
//! neither can `MockGraphDB` — it echoes back whatever JSON it was handed, so it
//! never exercises the core-column/`properties`-blob split where the value was
//! being dropped.
//!
//! This test therefore writes through `LadybugAdapter` into a temp directory and
//! reads back through the same `get_graph_data()` the renderer calls. The
//! `cognee-graph/ladybug` dev-dependency feature exists for exactly this file;
//! dev-dependency features never reach the shipped library, so `cargo build -p
//! cognee-visualization` does not pull in Ladybug.

use cognee_graph::{GraphDBTrait, LadybugAdapter};
use serde_json::json;

/// Base timestamp, epoch ms — the shape `cognee_models::DataPoint` writes.
const T0: i64 = 1_768_164_683_000;

/// `MEMORY_TIMELINE_GAP_MS` is 5 minutes, so 10 minutes forces a second run
/// event rather than one merged cluster.
const SECOND_RUN_OFFSET_MS: i64 = 600_000;

async fn seeded_adapter(dir: &tempfile::TempDir) -> LadybugAdapter {
    let db = LadybugAdapter::new(
        dir.path()
            .join("graph.db")
            .to_str()
            .expect("temp dir path is UTF-8"),
    )
    .await
    .expect("create LadybugAdapter");
    db.initialize().await.expect("initialize Ladybug");

    // Run 1: a document and its chunk.
    db.add_node_raw(json!({
        "id": "doc1",
        "name": "alice.md",
        "type": "TextDocument",
        "created_at": T0,
    }))
    .await
    .expect("add doc1");
    db.add_node_raw(json!({
        "id": "c0",
        "name": "",
        "type": "DocumentChunk",
        "text": "Alice knows Bob.",
        "chunk_index": 0,
        "document_id": "doc1",
        "created_at": T0 + 100,
    }))
    .await
    .expect("add c0");
    // Run 2, ten minutes later: an entity extracted from that chunk.
    db.add_node_raw(json!({
        "id": "e1",
        "name": "Alice",
        "type": "Entity",
        "created_at": T0 + SECOND_RUN_OFFSET_MS,
    }))
    .await
    .expect("add e1");

    db.add_edge("c0", "doc1", "is_part_of", None)
        .await
        .expect("add is_part_of");
    db.add_edge("c0", "e1", "contains", None)
        .await
        .expect("add contains");
    db
}

#[tokio::test]
async fn t_created_survives_a_round_trip_through_the_graph_adapter() {
    let dir = tempfile::tempdir().expect("temp dir");
    let db = seeded_adapter(&dir).await;

    let (nodes, edges) = db.get_graph_data().await.expect("get_graph_data");
    let pre = cognee_visualization::preprocess(nodes, edges, None);

    for (id, expected) in [
        ("doc1", T0),
        ("c0", T0 + 100),
        ("e1", T0 + SECOND_RUN_OFFSET_MS),
    ] {
        let node = pre
            .nodes
            .iter()
            .find(|node| node["id"] == json!(id))
            .unwrap_or_else(|| panic!("node {id} in the preprocessed payload"));
        assert_eq!(
            node.get("t_created"),
            Some(&json!(expected)),
            "`{id}` lost its creation time between the adapter and the renderer"
        );
        // The raw audit key is still dropped from the emitted node.
        assert!(node.get("created_at").is_none(), "{id} leaked created_at");
    }
}

#[tokio::test]
async fn memory_timeline_clusters_real_runs_from_adapter_data() {
    let dir = tempfile::tempdir().expect("temp dir");
    let db = seeded_adapter(&dir).await;

    let (nodes, edges) = db.get_graph_data().await.expect("get_graph_data");
    let pre = cognee_visualization::preprocess(nodes, edges, None);

    let timeline = pre.memory_map["timeline"]
        .as_array()
        .expect("memory timeline is an array");

    // With `t_created` unavailable this collapsed to a single synthetic event
    // spanning `t0 = t1 = 0` (`memory_map.rs`'s "no node carries t_created"
    // branch), so the count *and* the timestamps are the regression signal.
    assert_eq!(
        timeline.len(),
        2,
        "two writes 10 minutes apart must gap-cluster into two run events: {timeline:#?}"
    );
    assert_eq!(timeline[0]["t0"], json!(T0));
    assert_eq!(timeline[0]["t1"], json!(T0 + 100));
    assert_eq!(timeline[0]["node_ids"], json!(["doc1", "c0"]));
    assert_eq!(timeline[1]["t0"], json!(T0 + SECOND_RUN_OFFSET_MS));
    assert_eq!(timeline[1]["node_ids"], json!(["e1"]));
}
