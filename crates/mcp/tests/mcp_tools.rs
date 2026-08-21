#![cfg(feature = "runtime")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use cognee_mcp::config::{AgentConfig, EnvSource};
use cognee_mcp::detach::DrainSpawner;
use cognee_mcp::engine::{
    ApplyReceipt, EngineFactory, ForgetReceipt, ForgetTarget, ImproveReceipt, MemoryEngine,
    RecallItem, RecallRequest, RecallResponse, RecallSource,
};
use cognee_mcp::error::AgentError;
use cognee_mcp::event::{EventEnvelope, EventKind};
use cognee_mcp::generation::GenerationStore;
use cognee_mcp::lease::EngineLease;
use cognee_mcp::ledger::Ledger;
use cognee_mcp::spool::{Priority, Spool, SpoolRecord};
use cognee_mcp::tools::{McpTools, tool_descriptors};
use cognee_mcp::worker::{DrainBudget, Worker};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

fn descriptor<'a>(tools: &'a [Value], name: &str) -> &'a Value {
    tools
        .iter()
        .find(|tool| tool["name"] == name)
        .unwrap_or_else(|| panic!("missing {name} descriptor"))
}

#[test]
fn descriptors_publish_the_exact_memory_surface_and_defaults() {
    let tools = tool_descriptors();
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool["name"].as_str().expect("tool name"))
            .collect::<Vec<_>>(),
        ["remember", "recall", "forget"]
    );

    let remember = descriptor(&tools, "remember");
    assert_eq!(
        remember["inputSchema"]["required"],
        serde_json::json!(["data"])
    );
    assert_eq!(
        remember["inputSchema"]["properties"]["dataset_name"]["default"],
        "agent_sessions"
    );
    assert_eq!(
        remember["inputSchema"]["properties"]["self_improvement"]["default"],
        false
    );
    let remember_description = remember["description"].as_str().expect("description");
    for trigger in [
        "Please remember",
        "Save, note, or record this",
        "Keep this for next time",
        "Don't forget",
        "Going forward",
        "In future sessions",
        "My preference is",
        "standard workflow",
        "Always",
        "never",
    ] {
        assert!(
            remember_description.contains(trigger),
            "missing remember trigger: {trigger}"
        );
    }

    let recall = descriptor(&tools, "recall");
    assert_eq!(
        recall["inputSchema"]["required"],
        serde_json::json!(["query"])
    );
    assert_eq!(recall["inputSchema"]["properties"]["top_k"]["default"], 10);
    assert!(
        recall["inputSchema"]["properties"]["auto_route"]
            .get("default")
            .is_none(),
        "auto_route has a conditional runtime default"
    );
    assert_eq!(
        recall["inputSchema"]["properties"]["search_type"]["enum"],
        serde_json::json!([
            "GRAPH_COMPLETION",
            "RAG_COMPLETION",
            "CHUNKS",
            "SUMMARIES",
            "CODE",
            "FEELING_LUCKY"
        ])
    );
    let recall_description = recall["description"].as_str().expect("description");
    for trigger in [
        "yesterday",
        "earlier",
        "before",
        "last week",
        "last time",
        "previously",
        "previous session",
        "pick up where we left off",
        "continue this",
        "continue where we left off",
        "resume",
        "where were we?",
        "I told you",
        "you mentioned",
        "we discussed",
        "what did we try",
        "what was ruled out",
        "same issue",
        "recurring failure",
        "similar panic",
        "known problem",
        "artifact",
        "that command",
        "earlier test result",
        "preferences",
        "previous setup",
        "CONTAP",
        "case IDs",
        "PRs",
        "symbols",
        "cluster names",
        "panic signatures",
        "artifact paths",
    ] {
        assert!(
            recall_description.contains(trigger),
            "missing recall trigger: {trigger}"
        );
    }
    assert!(recall_description.contains("should not trigger broad recall by themselves"));

    let forget = descriptor(&tools, "forget");
    assert_eq!(
        forget["inputSchema"]["required"],
        serde_json::json!(["confirm"])
    );
    assert_eq!(
        forget["inputSchema"]["properties"]["everything"]["default"],
        false
    );
}

#[derive(Default)]
struct FakeEnv(BTreeMap<String, String>);

impl EnvSource for FakeEnv {
    fn get(&self, key: &str) -> Option<String> {
        self.0.get(key).cloned()
    }
}

fn config(root: &std::path::Path, allow_forget_all: bool) -> AgentConfig {
    let mut values = BTreeMap::from([("APEX_COGNEE_ROOT".to_owned(), root.display().to_string())]);
    if allow_forget_all {
        values.insert("APEX_COGNEE_ALLOW_FORGET_ALL".to_owned(), "true".to_owned());
    }
    AgentConfig::from_env(&FakeEnv(values)).expect("test config")
}

#[derive(Default)]
struct RecordingSpawner {
    calls: AtomicUsize,
}

impl DrainSpawner for RecordingSpawner {
    fn spawn(&self) -> std::io::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Default)]
struct EngineState {
    opens: AtomicUsize,
    closes: AtomicUsize,
    recalls: Mutex<Vec<RecallRequest>>,
    forgets: Mutex<Vec<ForgetTarget>>,
    applies: AtomicUsize,
    recall_items: Mutex<Vec<RecallItem>>,
    fail_forget: AtomicBool,
    required_fence: Mutex<Option<(GenerationStore, String, u64)>>,
}

struct FakeFactory {
    state: Arc<EngineState>,
}

#[async_trait]
impl EngineFactory for FakeFactory {
    async fn open(&self) -> Result<Box<dyn MemoryEngine>, AgentError> {
        self.state.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeEngine {
            state: Arc::clone(&self.state),
        }))
    }
}

struct FakeEngine {
    state: Arc<EngineState>,
}

#[async_trait]
impl MemoryEngine for FakeEngine {
    async fn contains_event(
        &mut self,
        _dataset: &str,
        _event_id: &str,
    ) -> Result<bool, AgentError> {
        Ok(false)
    }

    async fn apply_event(
        &mut self,
        _event: &cognee_mcp::event::EventEnvelope,
    ) -> Result<ApplyReceipt, AgentError> {
        self.state.applies.fetch_add(1, Ordering::SeqCst);
        Ok(ApplyReceipt::default())
    }

    async fn improve(
        &mut self,
        _dataset: &str,
        _session_ids: &[String],
    ) -> Result<ImproveReceipt, AgentError> {
        Ok(ImproveReceipt::default())
    }

    async fn recall(&mut self, request: RecallRequest) -> Result<RecallResponse, AgentError> {
        self.state
            .recalls
            .lock()
            .expect("recall log")
            .push(request.clone());
        Ok(RecallResponse {
            items: self.state.recall_items.lock().expect("items").clone(),
            search_type_used: request.search_type.clone(),
            auto_routed: request.auto_route,
        })
    }

    async fn forget(&mut self, target: ForgetTarget) -> Result<ForgetReceipt, AgentError> {
        if let Some((store, dataset, expected)) =
            self.state.required_fence.lock().expect("fence").as_ref()
        {
            if store.current(dataset).expect("current generation") != *expected {
                return Err(AgentError::Engine("generation_not_fenced"));
            }
        }
        self.state
            .forgets
            .lock()
            .expect("forget log")
            .push(target.clone());
        if self.state.fail_forget.load(Ordering::SeqCst) {
            return Err(AgentError::Retryable("delete_failed"));
        }
        Ok(ForgetReceipt {
            target: match target {
                ForgetTarget::Dataset(dataset) => format!("dataset:{dataset}"),
                ForgetTarget::All => "all".to_owned(),
            },
        })
    }

    async fn close(self: Box<Self>) {
        self.state.closes.fetch_add(1, Ordering::SeqCst);
    }
}

fn tools(
    temporary: &TempDir,
    allow_forget_all: bool,
) -> (
    AgentConfig,
    Arc<EngineState>,
    Arc<RecordingSpawner>,
    McpTools,
) {
    let config = config(&temporary.path().join("cognee"), allow_forget_all);
    let state = Arc::new(EngineState::default());
    let spawner = Arc::new(RecordingSpawner::default());
    let tools = McpTools::new(
        config.clone(),
        Arc::new(FakeFactory {
            state: Arc::clone(&state),
        }),
        spawner.clone(),
    )
    .with_identity("alice", "host-a", "/work/apex")
    .with_lease_wait(Duration::ZERO);
    (config, state, spawner, tools)
}

fn body(result: &Value) -> Value {
    let text = result["content"][0]["text"]
        .as_str()
        .expect("JSON text result");
    serde_json::from_str(text).expect("tool result JSON")
}

#[tokio::test]
async fn remember_queues_a_high_priority_event_without_opening_the_engine() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, spawner, tools) = tools(&temporary, false);

    let result = tools
        .call(
            "remember",
            json!({
                "data": "Use two build workers on the fleet.",
                "session_id": "session-17"
            }),
        )
        .await;

    assert_eq!(result["isError"], false);
    let body = body(&result);
    assert_eq!(body["status"], "queued");
    assert_eq!(body["dataset"], "agent_sessions");
    assert_eq!(body["session_id"], "session-17");
    assert_eq!(engine.opens.load(Ordering::SeqCst), 0);
    assert_eq!(spawner.calls.load(Ordering::SeqCst), 1);

    let spool = Spool::new(config.layout.clone(), config.limits.clone());
    let files = spool.pending_files().expect("pending files");
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].priority.as_str(), "high");
    let record: SpoolRecord =
        serde_json::from_slice(&std::fs::read(&files[0].path).expect("queued event bytes"))
            .expect("queued event");
    assert_eq!(record.envelope.event, EventKind::McpRemember);
    assert_eq!(
        record.envelope.payload["data"],
        "Use two build workers on the fleet."
    );
    assert_eq!(record.envelope.event_id, body["event_id"]);
}

#[tokio::test]
async fn recall_reads_a_matching_pending_memory_while_the_engine_is_busy() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, false);
    let remembered = tools
        .call(
            "remember",
            json!({"data": "The amber narwhal diagnostic ran yesterday."}),
        )
        .await;
    let event_id = body(&remembered)["event_id"]
        .as_str()
        .expect("event id")
        .to_owned();
    let blocker = EngineLease::new(config.layout.clone(), Duration::from_secs(180))
        .try_acquire("test-blocker")
        .expect("lease attempt")
        .expect("blocking lease");

    let recalled = tools
        .call("recall", json!({"query": "amber narwhal"}))
        .await;

    assert_eq!(recalled["isError"], false);
    let recalled = body(&recalled);
    assert_eq!(recalled["graph"]["status"], "busy");
    assert_eq!(recalled["pending"]["matched"], 1);
    assert_eq!(recalled["items"][0]["source"], "pending");
    assert_eq!(recalled["items"][0]["event_id"], event_id);
    assert_eq!(engine.opens.load(Ordering::SeqCst), 0);
    blocker.release().expect("release blocker");
}

#[tokio::test]
async fn recall_never_returns_pending_memory_from_a_superseded_generation() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, _engine, _spawner, tools) = tools(&temporary, true);
    let stale_generation = GenerationStore::new(config.layout.clone())
        .current("late_custom")
        .expect("generation before global forget");

    let forgotten = tools
        .call(
            "forget",
            json!({
                "everything": true,
                "confirm": "DELETE ALL COGNEE DATA"
            }),
        )
        .await;
    assert_eq!(forgotten["isError"], false);

    let stale = EventEnvelope::from_mcp_remember(
        "the deleted amber narwhal memory",
        None,
        false,
        "alice",
        "host-a",
        "2026-08-20T12:00:00.000000000Z".to_owned(),
        "/work/apex",
        "late_custom",
        stale_generation,
    );
    Spool::new(config.layout.clone(), config.limits.clone())
        .enqueue(&stale, Priority::High)
        .expect("enqueue stale request after global forget");
    let blocker = EngineLease::new(config.layout.clone(), Duration::from_secs(180))
        .try_acquire("test-blocker")
        .expect("lease attempt")
        .expect("blocking lease");

    let recalled = tools
        .call(
            "recall",
            json!({"query": "amber narwhal", "datasets": "late_custom"}),
        )
        .await;

    assert_eq!(recalled["isError"], true);
    let recalled = body(&recalled);
    assert_eq!(recalled["code"], "ENGINE_BUSY");
    blocker.release().expect("release blocker");
}

#[tokio::test]
async fn busy_recall_without_a_pending_match_is_retryable_and_reports_queue_depth() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, _engine, _spawner, tools) = tools(&temporary, false);
    let blocker = EngineLease::new(config.layout.clone(), Duration::from_secs(180))
        .try_acquire("test-blocker")
        .expect("lease attempt")
        .expect("blocking lease");

    let recalled = tools
        .call("recall", json!({"query": "nothing queued"}))
        .await;

    assert_eq!(recalled["isError"], true);
    let recalled = body(&recalled);
    assert_eq!(recalled["code"], "ENGINE_BUSY");
    assert_eq!(recalled["retryable"], true);
    assert_eq!(recalled["queue_depth"], 0);
    blocker.release().expect("release blocker");
}

#[tokio::test]
async fn pending_recall_rejects_a_queue_scan_above_its_hard_limit() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, _engine, _spawner, tools) = tools(&temporary, false);
    let spool = Spool::new(config.layout.clone(), config.limits.clone());
    for index in 0..257 {
        let event = EventEnvelope::from_mcp_remember(
            &format!("bounded recall record {index}"),
            None,
            false,
            "alice",
            "host-a",
            "2026-08-20T12:00:00.000000000Z".to_owned(),
            "/work/apex",
            "agent_sessions",
            0,
        );
        spool
            .enqueue(&event, Priority::High)
            .expect("enqueue bounded recall fixture");
    }

    let recalled = tools
        .call("recall", json!({"query": "bounded recall", "top_k": 3}))
        .await;

    assert_eq!(recalled["isError"], true);
    assert_eq!(body(&recalled)["code"], "PENDING_READ_ERROR");
}

#[tokio::test]
async fn recall_normalizes_defaults_and_deduplicates_pending_and_graph_sources() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (_config, engine, _spawner, tools) = tools(&temporary, false);
    tools
        .call(
            "remember",
            json!({"data": "Use the canary launcher for fleet validation."}),
        )
        .await;
    engine.recall_items.lock().expect("items").push(RecallItem {
        source: RecallSource::Graph,
        content: "Use the canary launcher for fleet validation.".to_owned(),
        score: None,
        dataset: "agent_sessions".to_owned(),
        session_id: None,
        timestamp: None,
        event_id: None,
        metadata: Map::new(),
    });

    let recalled = tools
        .call(
            "recall",
            json!({
                "query": "canary launcher",
                "search_type": "CHUNKS",
                "datasets": "agent_sessions"
            }),
        )
        .await;

    assert_eq!(recalled["isError"], false);
    let recalled = body(&recalled);
    assert_eq!(recalled["items"].as_array().expect("items").len(), 1);
    let item = &recalled["items"][0];
    for key in [
        "source",
        "content",
        "score",
        "dataset",
        "session_id",
        "timestamp",
        "event_id",
        "metadata",
    ] {
        assert!(item.get(key).is_some(), "missing stable key {key}");
    }
    assert_eq!(item["source"], "pending");
    assert_eq!(item["metadata"]["sources"], json!(["pending", "graph"]));
    assert_eq!(recalled["searchTypeUsed"], "CHUNKS");
    assert_eq!(recalled["autoRouted"], false);
    let requests = engine.recalls.lock().expect("requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].top_k, 10);
    assert!(!requests[0].auto_route);
    assert_eq!(engine.opens.load(Ordering::SeqCst), 1);
    assert_eq!(engine.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn recall_auto_routes_only_when_no_search_type_is_supplied() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (_config, engine, _spawner, tools) = tools(&temporary, false);

    let recalled = tools
        .call("recall", json!({"query": "what did we decide earlier?"}))
        .await;

    assert_eq!(recalled["isError"], false);
    let requests = engine.recalls.lock().expect("requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].search_type, None);
    assert!(requests[0].auto_route);
}

#[tokio::test]
async fn forget_rejects_ambiguous_or_unconfirmed_targets_without_mutation() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, false);

    for arguments in [
        json!({"confirm": "DELETE DATASET agent_sessions"}),
        json!({"dataset": "agent_sessions", "confirm": "yes"}),
        json!({"everything": true, "confirm": "DELETE ALL COGNEE DATA"}),
        json!({
            "dataset": "agent_sessions",
            "everything": true,
            "confirm": "DELETE DATASET agent_sessions"
        }),
    ] {
        let result = tools.call("forget", arguments).await;
        assert_eq!(result["isError"], true);
    }

    assert_eq!(engine.opens.load(Ordering::SeqCst), 0);
    assert_eq!(
        GenerationStore::new(config.layout)
            .current("agent_sessions")
            .expect("generation"),
        0
    );
}

#[tokio::test]
async fn dataset_forget_fences_and_quarantines_before_deleting() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, false);
    tools
        .call("remember", json!({"data": "obsolete canary fact"}))
        .await;
    *engine.required_fence.lock().expect("fence") = Some((
        GenerationStore::new(config.layout.clone()),
        "agent_sessions".to_owned(),
        1,
    ));

    let result = tools
        .call(
            "forget",
            json!({
                "dataset": "agent_sessions",
                "confirm": "DELETE DATASET agent_sessions"
            }),
        )
        .await;

    assert_eq!(result["isError"], false);
    let result = body(&result);
    assert_eq!(result["generation"]["previous"], 0);
    assert_eq!(result["generation"]["current"], 1);
    assert_eq!(result["generation"]["quarantined"], 1);
    assert_eq!(
        GenerationStore::new(config.layout.clone())
            .current("agent_sessions")
            .expect("generation"),
        1
    );
    assert_eq!(
        Spool::new(config.layout, config.limits)
            .depths()
            .expect("depths")
            .pending,
        0
    );
    assert_eq!(engine.opens.load(Ordering::SeqCst), 1);
    assert_eq!(engine.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn global_forget_fences_every_dataset_in_pending_and_processing_before_deleting() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, true);
    tools
        .call("remember", json!({"data": "obsolete default fact"}))
        .await;
    tools
        .call(
            "remember",
            json!({
                "data": "obsolete custom fact",
                "dataset_name": "project_notes"
            }),
        )
        .await;
    let spool = Spool::new(config.layout.clone(), config.limits.clone());
    let custom = spool
        .pending_files()
        .expect("pending files")
        .into_iter()
        .find(|file| {
            let record: SpoolRecord =
                serde_json::from_slice(&std::fs::read(&file.path).expect("pending record bytes"))
                    .expect("pending record");
            record.envelope.dataset == "project_notes"
        })
        .expect("custom dataset record");
    spool
        .claim(&custom)
        .expect("move custom record to processing");

    let result = tools
        .call(
            "forget",
            json!({
                "everything": true,
                "confirm": "DELETE ALL COGNEE DATA"
            }),
        )
        .await;

    assert_eq!(result["isError"], false);
    let result = body(&result);
    assert_eq!(result["generations"]["agent_sessions"]["current"], 1);
    assert_eq!(result["generations"]["project_notes"]["current"], 1);
    let generations = GenerationStore::new(config.layout.clone());
    assert_eq!(
        generations
            .current("agent_sessions")
            .expect("default generation"),
        1
    );
    assert_eq!(
        generations
            .current("project_notes")
            .expect("custom generation"),
        1
    );
    let depths = spool.depths().expect("spool depths");
    assert_eq!((depths.pending, depths.processing), (0, 0));
    assert_eq!(
        std::fs::read_dir(config.layout.spool_failed.join("superseded/generation-0"))
            .expect("superseded directory")
            .count(),
        2
    );
    assert_eq!(
        engine.forgets.lock().expect("forget calls").as_slice(),
        [ForgetTarget::All]
    );
}

#[tokio::test]
async fn global_forget_epoch_rejects_a_new_dataset_event_that_was_stamped_before_delete() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, true);
    let generations = GenerationStore::new(config.layout.clone());
    let stale_generation = generations
        .current("late_custom")
        .expect("generation before global forget");

    let forgotten = tools
        .call(
            "forget",
            json!({
                "everything": true,
                "confirm": "DELETE ALL COGNEE DATA"
            }),
        )
        .await;
    assert_eq!(forgotten["isError"], false);

    let stale = EventEnvelope::from_mcp_remember(
        "stamped before global deletion",
        Some("session-race"),
        false,
        "alice",
        "host-a",
        "2026-08-20T12:00:00.000000000Z".to_owned(),
        "/work/apex",
        "late_custom",
        stale_generation,
    );
    let spool = Spool::new(config.layout.clone(), config.limits.clone());
    spool
        .enqueue(&stale, Priority::High)
        .expect("enqueue request after global forget");
    let mut worker = Worker::new(
        config.layout.clone(),
        spool,
        EngineLease::new(
            config.layout.clone(),
            Duration::from_secs(u64::from(config.limits.lease_stale_seconds)),
        ),
        Ledger::open(config.layout.clone()).expect("worker ledger"),
        Arc::new(FakeFactory {
            state: engine.clone(),
        }),
        config.limits.clone(),
    );

    let report = worker.drain(DrainBudget::from_limits(&config.limits)).await;

    assert_eq!(
        generations
            .current("late_custom")
            .expect("generation after global forget"),
        1
    );
    assert_eq!(report.committed, 0);
    assert_eq!(report.quarantined, 1);
    assert_eq!(engine.applies.load(Ordering::SeqCst), 0);
    assert_eq!(
        std::fs::read_dir(config.layout.spool_failed.join("superseded/generation-0"))
            .expect("superseded directory")
            .count(),
        1
    );
}

#[tokio::test]
async fn global_forget_reports_the_durable_epoch_when_quarantine_fails() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, true);
    tools
        .call("remember", json!({"data": "obsolete default fact"}))
        .await;
    std::fs::write(
        config.layout.spool_failed.join("superseded"),
        b"deterministic quarantine blocker",
    )
    .expect("create quarantine blocker");

    let result = tools
        .call(
            "forget",
            json!({
                "everything": true,
                "confirm": "DELETE ALL COGNEE DATA"
            }),
        )
        .await;

    assert_eq!(result["isError"], true);
    let result = body(&result);
    assert_eq!(result["code"], "GENERATION_FENCE_ERROR");
    assert_eq!(result["global_generation"], 1);
    assert_eq!(result["generations"]["agent_sessions"]["current"], 1);
    assert!(
        result["message"]
            .as_str()
            .expect("message")
            .contains("fence advanced")
    );
    assert_eq!(
        GenerationStore::new(config.layout)
            .current("agent_sessions")
            .expect("generation after failed quarantine"),
        1
    );
    assert!(engine.forgets.lock().expect("forget calls").is_empty());
}

#[tokio::test]
async fn failed_dataset_delete_keeps_the_advanced_fence_and_is_retryable() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let (config, engine, _spawner, tools) = tools(&temporary, false);
    engine.fail_forget.store(true, Ordering::SeqCst);
    *engine.required_fence.lock().expect("fence") = Some((
        GenerationStore::new(config.layout.clone()),
        "agent_sessions".to_owned(),
        1,
    ));

    let result = tools
        .call(
            "forget",
            json!({
                "dataset": "agent_sessions",
                "confirm": "DELETE DATASET agent_sessions"
            }),
        )
        .await;

    assert_eq!(result["isError"], true);
    let result = body(&result);
    assert_eq!(result["retryable"], true);
    assert_eq!(result["generation"]["current"], 1);
    assert_eq!(
        GenerationStore::new(config.layout)
            .current("agent_sessions")
            .expect("generation"),
        1
    );
    assert_eq!(engine.closes.load(Ordering::SeqCst), 1);
}
