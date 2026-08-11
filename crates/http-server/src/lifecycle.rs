//! Server startup and shutdown lifecycle hooks.
//!
//! The closed `cognee-http-cloud` crate provides its own bootstrap that
//! seeds the `principals` / `users` / `tenants` tables; OSS keeps the
//! sync-registry sweep + pipeline-registry shutdown that are DB-free.

use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during server lifecycle transitions.
#[derive(Debug, Error)]
pub enum LifecycleError {
    /// Database migration failed.
    #[error("migration failed: {0}")]
    MigrationFailed(String),

    /// Bootstrap of default principals failed.
    #[error("bootstrap failed: {0}")]
    BootstrapFailed(String),
}

/// All-zero UUID — matches Python's `default_user_id`.
const DEFAULT_USER_ID_HEX: &str = "00000000000000000000000000000000";

/// How long [`on_shutdown`] waits for already-dispatched telemetry POSTs to leave
/// the process. Deliberately short: a SIGTERM must not be held up by an analytics
/// collector.
#[cfg(feature = "telemetry")]
const TELEMETRY_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

/// Called once before the router is handed to `axum::serve`.
///
/// OSS-side bootstrap is a no-op: the synthetic default user is
/// DB-free (no `principals`/`users`/`user_tenants` rows to seed). Closed
/// `cognee-http-cloud` provides its own startup hook that seeds the
/// `(default_user, default_tenant)` rows per `tenants.md §6`.
pub async fn on_startup(_state: &crate::state::AppState) -> Result<(), LifecycleError> {
    tracing::info!("Backend server has started");
    Ok(())
}

/// Convenience accessor — for callers that need the well-known IDs.
pub fn default_user_id() -> Uuid {
    Uuid::parse_str(DEFAULT_USER_ID_HEX).unwrap_or(Uuid::nil())
}

/// Called on graceful shutdown (SIGTERM / SIGINT).
///
/// Order is the whole design here, and it is the reverse of startup: **drain the
/// work first, then close the stores.** Closing a store while the pipeline
/// registry still has tasks running would make every in-flight cognify fail
/// against a closed handle and emit a burst of errors indistinguishable from a
/// crash — the exact failure the relational-close comment below was written to
/// avoid. So the registry shutdown and the sync abort stay first, and the graph /
/// vector / relational closes come after.
///
/// Closing through `&self` is what makes this possible at all: `lib.graph_db` and
/// `lib.vector_db` are `Arc` clones held by handlers and pipeline builders, so
/// there is nothing here to drop — and for a Postgres store a retained `Arc` is
/// precisely the case where the pool would otherwise stay open for the life of
/// the process.
///
/// # Deliberate limitation
///
/// [`crate::components::ComponentHandles`] stores its slots as plain
/// `Option<Arc<…>>`, so this hook **cannot** release the `reqwest` connection
/// pools behind `llm` / `embedding_engine` / `transcriber` / `responses_client`,
/// nor an ONNX session: doing so needs interior mutability in those fields, which
/// is a breaking change for embedders. The standalone binary exits immediately
/// after this returns, so the OS reclaims them; an **in-process embedder that
/// rebuilds the router without exiting keeps that gap**. Recorded here rather than
/// papered over with an interior-mutability layer.
///
/// Also pre-existing and worth knowing: without the `bin` feature there is no
/// graceful-shutdown wiring at all (see `lib.rs`), so this function never runs and
/// nothing is closed.
pub async fn on_shutdown(state: &crate::state::AppState) {
    tracing::info!("Backend server is shutting down");

    if let Err(e) = state.pipelines.shutdown().await {
        tracing::warn!("pipeline registry shutdown failed (non-fatal): {e}");
    } else {
        tracing::info!("pipeline registry shutdown complete");
    }

    // Abort every in-flight cloud sync — the durable-row "mark failed"
    // step moved closed alongside `SyncOperationRepository`.
    let aborted = state.sync.abort_all();
    if !aborted.is_empty() {
        tracing::info!(
            "aborted {} in-flight cloud sync(s) on shutdown",
            aborted.len()
        );
    }

    // Close the knowledge stores now that the work using them has drained.
    //
    // Same mechanism as the relational close below, different resources: an
    // embedded graph leaves an un-checkpointed `<db>.wal` and a write lock on its
    // database file, and a Postgres graph/vector adapter owns a pool of its own
    // that a drop would not close — and there is no drop here anyway, because
    // these are `Arc` clones. Both are no-ops for backends that own nothing
    // closable (the in-memory brute-force store, LanceDB).
    if let Some(lib) = state.lib.as_ref() {
        if let Some(graph) = lib.graph_db.as_ref() {
            match graph.close().await {
                Ok(()) => tracing::info!("graph database closed"),
                Err(e) => tracing::warn!("closing the graph database failed (non-fatal): {e}"),
            }
        }
        if let Some(vector) = lib.vector_db.as_ref() {
            match vector.close().await {
                Ok(()) => tracing::info!("vector database closed"),
                Err(e) => tracing::warn!("closing the vector database failed (non-fatal): {e}"),
            }
        }
    }

    // Close the relational pool last, once the work that uses it has stopped.
    //
    // Exiting without closing leaves a SQLite database's `-wal`/`-shm` sidecars
    // on disk: dropping the pool only flags it closed and lets its connections
    // tear down concurrently, and SQLite unlinks the sidecars only when the
    // *last* connection closes (issue #132). The next start recovers them, so
    // nothing is corrupt, but a server whose data directory is ephemeral (a
    // container, a test harness) leaves litter behind and looks like it crashed.
    if let Some(lib) = state.lib.as_ref() {
        match cognee_database::close(&lib.database).await {
            Ok(()) => tracing::info!("relational database closed"),
            Err(e) => tracing::warn!("closing the relational database failed (non-fatal): {e}"),
        }
    }

    // Last of all, let the analytics POSTs that are already in flight finish.
    // `send_telemetry` is fire-and-forget, so the shutdown event itself — the one
    // that says why the server stopped — is otherwise discarded when the process
    // exits (measured: 0 of 1 delivered without a flush, 1 of 1 with one).
    // Hard-bounded: a slow or blackholed collector must never hold up a SIGTERM.
    #[cfg(feature = "telemetry")]
    if !cognee_telemetry::flush(TELEMETRY_FLUSH_TIMEOUT).await {
        tracing::debug!(
            "telemetry still in flight after {TELEMETRY_FLUSH_TIMEOUT:?}; \
             dropping the remainder rather than delaying shutdown"
        );
    }
}
