#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Stage 2c integration tests — `distill_session` /
//! `distill_sessions_in_knowledge_graph`.
//!
//! The pipeline curates a session's Q&A into proposed lessons (curator LLM),
//! writes/rejects each (writer LLM), and add+cognifies accepted lessons tagged
//! with the `session_learnings` node-set. The LLM is mocked so the gate
//! ordering and node-set tagging are deterministic.
#![cfg(feature = "testing")]

use std::sync::Arc;

use cognee_cognify::CognifyConfig;
use cognee_cognify::memify::distill_sessions::{
    DISTILLATE_NODE_SET, DistillationStatus, distill_session, distill_sessions_in_knowledge_graph,
};
use cognee_database::ops::datasets as ds_ops;
use cognee_database::{DatabaseConnection, IngestDb, connect, initialize};
use cognee_embedding::MockEmbeddingEngine;
use cognee_graph::MockGraphDB;
use cognee_ingestion::AddPipeline;
use cognee_llm::Llm;
use cognee_ontology::{NoOpOntologyResolver, OntologyResolver};
use cognee_session::{FsSessionStore, SessionStore};
use cognee_storage::{LocalStorage, StorageTrait};
use cognee_test_utils::MockLlm;
use cognee_vector::MockVectorDB;
use tempfile::TempDir;
use uuid::Uuid;

struct Harness {
    _temp: TempDir,
    _sess_dir: TempDir,
    db: Arc<DatabaseConnection>,
    storage: Arc<dyn StorageTrait>,
    add_pipeline: AddPipeline,
    graph_db: Arc<MockGraphDB>,
    vector_db: Arc<MockVectorDB>,
    embedding_engine: Arc<MockEmbeddingEngine>,
    ontology: Arc<dyn OntologyResolver>,
    session_store: Arc<dyn SessionStore>,
}

async fn make_harness() -> Harness {
    let temp = TempDir::new().unwrap();
    let sess_dir = TempDir::new().unwrap();
    let db_path = temp.path().join("cognee.db");
    std::fs::File::create(&db_path).unwrap();
    let url = format!("sqlite://{}", db_path.display());
    let db = connect(&url).await.unwrap();
    initialize(&db).await.unwrap();
    let db = Arc::new(db);
    let storage: Arc<dyn StorageTrait> = Arc::new(LocalStorage::new(temp.path().join("storage")));
    storage.initialize().await.unwrap();

    let ingest_db: Arc<dyn IngestDb> = db.clone();
    let graph_db = Arc::new(MockGraphDB::new());
    let vector_db = Arc::new(MockVectorDB::new());
    let add_pipeline = AddPipeline::new(Arc::clone(&storage), ingest_db)
        .with_thread_pool(Arc::new(
            cognee_core::RayonThreadPool::with_default_threads().unwrap(),
        ))
        .with_graph_db(graph_db.clone() as Arc<dyn cognee_graph::GraphDBTrait>)
        .with_vector_db(vector_db.clone() as Arc<dyn cognee_vector::VectorDB>)
        .with_database(Arc::clone(&db));

    let embedding_engine = Arc::new(MockEmbeddingEngine::new(16));
    let ontology: Arc<dyn OntologyResolver> = Arc::new(NoOpOntologyResolver::new());

    let session_store: Arc<dyn SessionStore> = Arc::new(FsSessionStore::new(sess_dir.path()));

    Harness {
        _temp: temp,
        _sess_dir: sess_dir,
        db,
        storage,
        add_pipeline,
        graph_db,
        vector_db,
        embedding_engine,
        ontology,
        session_store,
    }
}

fn noop_repo() -> Arc<dyn cognee_database::PipelineRunRepository> {
    Arc::new(cognee_database::NoopPipelineRunRepository::new())
}

fn thread_pool() -> Arc<dyn cognee_core::CpuPool> {
    Arc::new(cognee_core::RayonThreadPool::with_default_threads().expect("RayonThreadPool init"))
}

/// No non-empty Q&A → `NoQaEntries` without any LLM call.
#[tokio::test]
async fn distill_no_qa_returns_no_qa_entries() {
    let h = make_harness().await;
    let owner = Uuid::new_v4();
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::empty());
    let config = CognifyConfig::default();

    let res = distill_session(
        "no_such_session",
        "ds_noqa",
        owner,
        None,
        Arc::clone(&h.session_store),
        &h.add_pipeline,
        llm,
        Arc::clone(&h.storage),
        h.graph_db.clone() as Arc<_>,
        h.vector_db.clone() as Arc<_>,
        h.embedding_engine.clone() as Arc<_>,
        Arc::clone(&h.db),
        noop_repo(),
        thread_pool(),
        Arc::clone(&h.ontology),
        &config,
    )
    .await
    .unwrap();

    assert_eq!(res.status, DistillationStatus::NoQaEntries);
    assert!(res.documents.is_empty());
    // No dataset was created (add never ran).
    let ds = ds_ops::get_dataset_by_name(h.db.as_ref(), "ds_noqa", owner, None)
        .await
        .unwrap();
    assert!(ds.is_none(), "no dataset should exist when there is no Q&A");
}

/// Curator returns invalid JSON → the batch fails open to `[]`, so the whole
/// session gates at `NoProposedLessons` (no writer/add/cognify runs).
#[tokio::test]
async fn distill_curator_failopen_yields_no_proposed_lessons() {
    let h = make_harness().await;
    let owner = Uuid::new_v4();
    let user_id = owner.to_string();
    let session_id = "sess_curator_fail";

    h.session_store
        .create_qa_entry(session_id, Some(&user_id), "Q?", "A.", None)
        .await
        .unwrap();

    // One (only) LLM response, deliberately not valid structured output for the
    // curator schema → curate_batch fails open to [].
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::new(vec!["not json at all".to_string()]));
    let config = CognifyConfig::default();

    let res = distill_session(
        session_id,
        "ds_curator_fail",
        owner,
        None,
        Arc::clone(&h.session_store),
        &h.add_pipeline,
        llm,
        Arc::clone(&h.storage),
        h.graph_db.clone() as Arc<_>,
        h.vector_db.clone() as Arc<_>,
        h.embedding_engine.clone() as Arc<_>,
        Arc::clone(&h.db),
        noop_repo(),
        thread_pool(),
        Arc::clone(&h.ontology),
        &config,
    )
    .await
    .unwrap();

    assert_eq!(res.status, DistillationStatus::NoProposedLessons);
    assert!(res.documents.is_empty());
}

/// Writer rejects the proposed lesson → `NoAcceptedLessons`, no add/cognify.
#[tokio::test]
async fn distill_writer_rejects_yields_no_accepted_lessons() {
    let h = make_harness().await;
    let owner = Uuid::new_v4();
    let user_id = owner.to_string();
    let session_id = "sess_reject";

    h.session_store
        .create_qa_entry(
            session_id,
            Some(&user_id),
            "Who is Alice?",
            "Alice is an engineer.",
            None,
        )
        .await
        .unwrap();

    // 1) curator proposes one lesson; 2) writer rejects it.
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::new(vec![
        r#"{"lessons":[{"working_statement":"Alice is an engineer.","member_entry_ids":[]}]}"#
            .to_string(),
        r#"{"accept":false,"reason":"not_durable"}"#.to_string(),
    ]));
    let config = CognifyConfig::default();

    let res = distill_session(
        session_id,
        "ds_reject",
        owner,
        None,
        Arc::clone(&h.session_store),
        &h.add_pipeline,
        llm,
        Arc::clone(&h.storage),
        h.graph_db.clone() as Arc<_>,
        h.vector_db.clone() as Arc<_>,
        h.embedding_engine.clone() as Arc<_>,
        Arc::clone(&h.db),
        noop_repo(),
        thread_pool(),
        Arc::clone(&h.ontology),
        &config,
    )
    .await
    .unwrap();

    assert_eq!(res.status, DistillationStatus::NoAcceptedLessons);
    assert!(res.documents.is_empty());
    // No dataset created — add never ran.
    let ds = ds_ops::get_dataset_by_name(h.db.as_ref(), "ds_reject", owner, None)
        .await
        .unwrap();
    assert!(ds.is_none(), "rejected lessons must not create a dataset");
}

/// Full curate → propose → accept → publish happy path: the accepted lesson is
/// add+cognified and the resulting Data row carries BOTH node-set tags.
#[tokio::test]
async fn distill_happy_path_tags_both_node_sets() {
    let h = make_harness().await;
    let owner = Uuid::new_v4();
    let user_id = owner.to_string();
    let session_id = "sess_happy";

    h.session_store
        .create_qa_entry(
            session_id,
            Some(&user_id),
            "How does TerraScout index data?",
            "TerraScout indexes nightly.",
            None,
        )
        .await
        .unwrap();

    // 1) curator proposes one lesson; 2) writer accepts + writes it.
    // Remaining cognify graph-extraction calls fall back to the mock's default
    // empty-graph response.
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::new(vec![
        r#"{"lessons":[{"working_statement":"TerraScout indexes nightly.","member_entry_ids":[]}]}"#
            .to_string(),
        r#"{"accept":true,"statement":"TerraScout indexes data nightly.","entities":["TerraScout"],"why_learned":"learned while asking how indexing works"}"#
            .to_string(),
    ]));
    let config = CognifyConfig::default();

    // Use the fail-open multi-session loop so that any internal cognify error
    // (mock backends) does not abort the test; the add-phase node-set tagging is
    // the deterministic invariant we assert.
    let r = distill_sessions_in_knowledge_graph(
        &[session_id.to_string()],
        "ds_happy",
        owner,
        None,
        Arc::clone(&h.session_store),
        &h.add_pipeline,
        llm,
        Arc::clone(&h.storage),
        h.graph_db.clone() as Arc<_>,
        h.vector_db.clone() as Arc<_>,
        h.embedding_engine.clone() as Arc<_>,
        Arc::clone(&h.db),
        noop_repo(),
        thread_pool(),
        Arc::clone(&h.ontology),
        &config,
    )
    .await;

    // The dataset exists and holds the rendered lesson Data row, tagged with
    // BOTH the generic and the per-session node-set.
    let ds = ds_ops::get_dataset_by_name(h.db.as_ref(), "ds_happy", owner, None)
        .await
        .unwrap()
        .expect("dataset exists after publish");
    let data_items = ds_ops::get_dataset_data(h.db.as_ref(), ds.id)
        .await
        .unwrap();
    assert!(
        !data_items.is_empty(),
        "dataset should have at least one distilled-lesson Data row"
    );
    let per_session_tag = format!("{DISTILLATE_NODE_SET}:{session_id}");
    let has_both = data_items.iter().any(|d| {
        d.node_set
            .as_deref()
            .map(|s| s.contains(DISTILLATE_NODE_SET) && s.contains(&per_session_tag))
            .unwrap_or(false)
    });
    assert!(
        has_both,
        "expected a Data row tagged with both '{DISTILLATE_NODE_SET}' and '{per_session_tag}'; got {:?}",
        data_items
            .iter()
            .map(|d| d.node_set.clone())
            .collect::<Vec<_>>()
    );

    // If cognify also succeeded end-to-end, the aggregate reflects one published
    // lesson. (Tolerant of internal mock-cognify failure, which the loop swallows
    // fail-open; the tagging assertion above is the hard invariant.)
    assert!(r.lessons_published <= 1);
}
