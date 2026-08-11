#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Regression test for `on_shutdown`: a graceful shutdown must release the
//! knowledge stores' OS resources, not just the relational pool.
//!
//! The relational half landed in #135. The stores were left out, and for the HTTP
//! server that gap is not closable by a `Drop`: `lib.graph_db` is an `Arc` clone
//! held by handlers and pipeline builders, so there is no owner to drop — the
//! store has to be closed through `&self` or not at all.
//!
//! What that costs, measured on the real binary and stated precisely because the
//! easy overstatement is wrong: on a **normal** exit the graph WAL does get
//! released either way, because process teardown eventually drops the last `Arc`.
//! The difference shows up whenever the process does not get to finish exiting —
//! a container whose grace period ends when the shutdown hook does, an in-process
//! embedder that rebuilds the router, or any Postgres store, whose pool a retained
//! `Arc` keeps open for the life of the process. SIGTERM followed by SIGKILL as
//! soon as `on_shutdown` finished its work: **1 orphan (`sys/graph.wal`, 1.1 KB)
//! before the fix, 0 after**, with `graph database closed` logged.
#![cfg(feature = "ladybug")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cognee_graph::{GraphDBTrait, LadybugAdapter};
use cognee_http_server::lifecycle::on_shutdown;

mod support;

/// Every `*.wal` under `root` — the embedded graph's sidecar.
fn wal_files(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("wal") {
                found.push(path.to_string_lossy().into_owned());
            }
        }
    }
    found
}

/// Warm an embedded graph under `dir` and write enough to leave a real WAL.
async fn graph_with_writes(dir: &Path) -> (Arc<dyn GraphDBTrait>, PathBuf) {
    let db_path = dir.join("graph.db");
    let adapter = LadybugAdapter::new(db_path.to_str().expect("utf-8 temp path"))
        .await
        .expect("open the embedded graph");
    adapter.initialize().await.expect("initialize");

    let nodes: Vec<_> = (0..500)
        .map(|i| {
            serde_json::json!({
                "id": format!("n{i}"),
                "name": format!("Node {i}"),
                "type": "TestNode",
                "properties": {"idx": i, "pad": "x".repeat(64)},
            })
        })
        .collect();
    adapter.add_nodes_raw(nodes).await.expect("add_nodes_raw");

    (Arc::new(adapter) as Arc<dyn GraphDBTrait>, db_path)
}

/// `on_shutdown` must leave no graph WAL behind.
///
/// Fails before the fix — `on_shutdown` closed only the relational pool, so
/// `graph.db.wal` (~660 KB for this workload) survived the shutdown and, with the
/// process being killed straight afterwards, survived the exit too.
#[tokio::test]
async fn shutdown_releases_the_embedded_graph_wal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (graph, _db_path) = graph_with_writes(dir.path()).await;

    // A second holder, standing in for the handler/pipeline clones that make a
    // drop-based teardown impossible here.
    let still_held = Arc::clone(&graph);

    let state = support::build_p4_state(None, None, Some(graph)).await;

    assert!(
        !wal_files(dir.path()).is_empty(),
        "precondition: the writes must leave an un-checkpointed WAL under {}",
        dir.path().display(),
    );

    on_shutdown(&state).await;

    let leftover = wal_files(dir.path());
    assert!(
        leftover.is_empty(),
        "on_shutdown must release the graph store's WAL, found: {leftover:?}"
    );

    // The surviving clone observes the closed store rather than reopening it.
    assert!(
        still_held.get_node("n1").await.is_err(),
        "a query after shutdown must fail rather than silently reopen the store"
    );
}

/// `on_shutdown` stays a no-op-safe path for a state with no stores wired, and
/// for one whose store owns nothing closable — both are normal configurations
/// (`lib: None` in most tests, an in-memory vector store in production).
#[tokio::test]
async fn shutdown_is_safe_without_stores() {
    let state = support::build_p4_state(None, None, None).await;
    on_shutdown(&state).await;
    // Twice, because a shutdown signal can arrive while one is already running.
    on_shutdown(&state).await;
}
