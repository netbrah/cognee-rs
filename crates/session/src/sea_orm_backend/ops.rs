use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter,
    QueryOrder, QuerySelect,
};
use uuid::Uuid;

use super::entity;
use crate::error::SessionError;
use crate::session_store::SessionQAUpdate;
use crate::types::{norm_user_id, norm_user_id_opt, SessionTraceStep};

fn map_db_err(e: sea_orm::DbErr) -> SessionError {
    SessionError::StoreError(e.to_string())
}

pub async fn create_qa_entry(
    db: &DatabaseConnection,
    id: Uuid,
    session_id: &str,
    user_id: Option<&str>,
    question: &str,
    answer: &str,
    context: Option<&str>,
) -> Result<(), SessionError> {
    let model = entity::ActiveModel {
        id: Set(id.simple().to_string()),
        session_id: Set(session_id.to_string()),
        user_id: Set(norm_user_id_opt(user_id)),
        question: Set(question.to_string()),
        answer: Set(answer.to_string()),
        context: Set(context.map(|s| s.to_string())),
        created_at: Set(Utc::now()),
        feedback_text: Set(None),
        feedback_score: Set(None),
        used_graph_element_ids: Set(None),
        memify_metadata: Set(None),
    };
    model.insert(db).await.map_err(map_db_err)?;
    Ok(())
}

pub async fn get_latest_entries(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
    limit: usize,
) -> Result<Vec<entity::Model>, SessionError> {
    let user_id = norm_user_id_opt(user_id);
    let user_id = user_id.as_deref();

    // To get the last N entries ordered oldest-first, we query DESC with limit,
    // then reverse in memory.
    let mut query = entity::Entity::find().filter(entity::Column::SessionId.eq(session_id));

    if let Some(uid) = user_id {
        query = query.filter(entity::Column::UserId.eq(uid));
    }

    // `usize::MAX` is the callers' "no limit" sentinel (e.g. the sessions
    // detail router reads the whole history for parity with Python). Passing
    // it through `.limit()` panics in the Postgres driver binding u64::MAX
    // as i64; omit the LIMIT for the sentinel instead.
    let query = if limit < usize::MAX {
        query.order_by_desc(entity::Column::CreatedAt).limit(limit as u64)
    } else {
        query.order_by_desc(entity::Column::CreatedAt)
    };

    let mut entries = query.all(db).await.map_err(map_db_err)?;

    entries.reverse();
    Ok(entries)
}

pub async fn get_all_entries(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
) -> Result<Vec<entity::Model>, SessionError> {
    let user_id = norm_user_id_opt(user_id);
    let user_id = user_id.as_deref();

    let mut query = entity::Entity::find().filter(entity::Column::SessionId.eq(session_id));

    if let Some(uid) = user_id {
        query = query.filter(entity::Column::UserId.eq(uid));
    }

    query
        .order_by_asc(entity::Column::CreatedAt)
        .all(db)
        .await
        .map_err(map_db_err)
}

pub async fn delete_session(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
) -> Result<u64, SessionError> {
    let user_id = norm_user_id_opt(user_id);
    let user_id = user_id.as_deref();

    let mut delete = entity::Entity::delete_many().filter(entity::Column::SessionId.eq(session_id));

    if let Some(uid) = user_id {
        delete = delete.filter(entity::Column::UserId.eq(uid));
    }

    let result = delete.exec(db).await.map_err(map_db_err)?;

    // Also delete graph context for this session
    let mut gc_delete = entity::graph_context::Entity::delete_many()
        .filter(entity::graph_context::Column::SessionId.eq(session_id));
    if let Some(uid) = user_id {
        gc_delete = gc_delete.filter(entity::graph_context::Column::UserId.eq(uid));
    }
    let _ = gc_delete.exec(db).await.map_err(map_db_err)?;

    Ok(result.rows_affected)
}

pub async fn delete_qa_entry(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
    qa_id: &str,
) -> Result<bool, SessionError> {
    let user_id = norm_user_id_opt(user_id);
    let user_id = user_id.as_deref();

    let mut delete = entity::Entity::delete_many()
        .filter(entity::Column::Id.eq(qa_id))
        .filter(entity::Column::SessionId.eq(session_id));

    if let Some(uid) = user_id {
        delete = delete.filter(entity::Column::UserId.eq(uid));
    }

    let result = delete.exec(db).await.map_err(map_db_err)?;
    Ok(result.rows_affected > 0)
}

/// Delete all rows from the session_qa_entries and session_graph_context tables (prune).
pub async fn delete_all(db: &DatabaseConnection) -> Result<(), SessionError> {
    entity::Entity::delete_many()
        .exec(db)
        .await
        .map_err(map_db_err)?;
    entity::graph_context::Entity::delete_many()
        .exec(db)
        .await
        .map_err(map_db_err)?;
    Ok(())
}

/// Update fields on a QA entry. Returns true if the entry was found and updated.
pub async fn update_qa_entry(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
    qa_id: &str,
    updates: SessionQAUpdate,
) -> Result<bool, SessionError> {
    let user_id = norm_user_id_opt(user_id);
    let user_id = user_id.as_deref();

    // First find the existing entry
    let mut query = entity::Entity::find()
        .filter(entity::Column::Id.eq(qa_id))
        .filter(entity::Column::SessionId.eq(session_id));

    if let Some(uid) = user_id {
        query = query.filter(entity::Column::UserId.eq(uid));
    }

    let existing = query.one(db).await.map_err(map_db_err)?;
    let Some(existing) = existing else {
        return Ok(false);
    };

    // Build an ActiveModel with only the fields that need updating
    let mut active: entity::ActiveModel = existing.into();

    if let Some(ref q) = updates.question {
        active.question = Set(q.clone());
    }
    if let Some(ref a) = updates.answer {
        active.answer = Set(a.clone());
    }
    if let Some(ref ctx) = updates.context {
        active.context = Set(ctx.clone());
    }
    if let Some(ref ft) = updates.feedback_text {
        active.feedback_text = Set(ft.clone());
    }
    if let Some(ref fs) = updates.feedback_score {
        active.feedback_score = Set(*fs);
    }
    if let Some(ref ids) = updates.used_graph_element_ids {
        active.used_graph_element_ids = Set(ids
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default()));
    }
    if let Some(ref mm) = updates.memify_metadata {
        active.memify_metadata = Set(mm
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default()));
    }

    // The `id` field is already `Unchanged` from the `.into()` conversion,
    // which SeaORM uses as the WHERE clause for the UPDATE statement.
    active.update(db).await.map_err(map_db_err)?;
    Ok(true)
}

fn graph_context_id(user_id: Option<&str>, session_id: &str) -> String {
    let uid = norm_user_id_opt(user_id);
    let uid = uid.as_deref().unwrap_or("default");
    format!("{uid}:{session_id}")
}

/// Retrieve the graph context for a session.
pub async fn get_graph_context(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
) -> Result<Option<String>, SessionError> {
    let id = graph_context_id(user_id, session_id);
    let model = entity::graph_context::Entity::find_by_id(&id)
        .one(db)
        .await
        .map_err(map_db_err)?;
    Ok(model.map(|m| m.context))
}

/// Append one agent-trace step. Returns the persisted `trace_id`.
pub async fn save_trace_step(
    db: &DatabaseConnection,
    user_id: &str,
    session_id: &str,
    step: SessionTraceStep,
) -> Result<String, SessionError> {
    let user_id = norm_user_id(user_id);

    let method_params_json = serde_json::to_string(&step.method_params)
        .map_err(|e| SessionError::StoreError(format!("json error: {e}")))?;
    let method_return_value_json = match &step.method_return_value {
        Some(v) => Some(
            serde_json::to_string(v)
                .map_err(|e| SessionError::StoreError(format!("json error: {e}")))?,
        ),
        None => None,
    };

    // Assign next `seq` for this `(user_id, session_id)` so reads return
    // entries in stable insertion order (independent of timestamp resolution).
    let max_seq: Option<i64> = entity::trace_step::Entity::find()
        .filter(entity::trace_step::Column::UserId.eq(user_id.as_str()))
        .filter(entity::trace_step::Column::SessionId.eq(session_id))
        .order_by_desc(entity::trace_step::Column::Seq)
        .limit(1)
        .one(db)
        .await
        .map_err(map_db_err)?
        .map(|m| m.seq);
    let next_seq = max_seq.unwrap_or(0) + 1;

    let trace_id = step.trace_id.clone();
    let model = entity::trace_step::ActiveModel {
        trace_id: Set(trace_id.clone()),
        user_id: Set(user_id),
        session_id: Set(session_id.to_string()),
        seq: Set(next_seq),
        created_at: Set(Utc::now()),
        origin_function: Set(step.origin_function),
        status: Set(step.status),
        memory_query: Set(step.memory_query),
        memory_context: Set(step.memory_context),
        method_params: Set(method_params_json),
        method_return_value: Set(method_return_value_json),
        error_message: Set(step.error_message),
        session_feedback: Set(step.session_feedback),
    };
    model.insert(db).await.map_err(map_db_err)?;
    Ok(trace_id)
}

/// Read agent-trace steps for `(user_id, session_id)`, ordered oldest-first.
pub async fn read_trace_steps(
    db: &DatabaseConnection,
    user_id: &str,
    session_id: &str,
) -> Result<Vec<SessionTraceStep>, SessionError> {
    let user_id = norm_user_id(user_id);

    let models = entity::trace_step::Entity::find()
        .filter(entity::trace_step::Column::UserId.eq(user_id.as_str()))
        .filter(entity::trace_step::Column::SessionId.eq(session_id))
        .order_by_asc(entity::trace_step::Column::Seq)
        .all(db)
        .await
        .map_err(map_db_err)?;

    let mut out = Vec::with_capacity(models.len());
    for m in models {
        let method_params: serde_json::Value = serde_json::from_str(&m.method_params)
            .map_err(|e| SessionError::StoreError(format!("json parse error: {e}")))?;
        let method_return_value = match m.method_return_value {
            Some(s) => Some(
                serde_json::from_str::<serde_json::Value>(&s)
                    .map_err(|e| SessionError::StoreError(format!("json parse error: {e}")))?,
            ),
            None => None,
        };
        out.push(SessionTraceStep {
            trace_id: m.trace_id,
            origin_function: m.origin_function,
            status: m.status,
            memory_query: m.memory_query,
            memory_context: m.memory_context,
            method_params,
            method_return_value,
            error_message: m.error_message,
            session_feedback: m.session_feedback,
        });
    }
    Ok(out)
}

/// Store (or overwrite) the graph context for a session.
pub async fn set_graph_context(
    db: &DatabaseConnection,
    session_id: &str,
    user_id: Option<&str>,
    context: &str,
) -> Result<(), SessionError> {
    let id = graph_context_id(user_id, session_id);

    // Try to find existing, then update or insert
    let existing = entity::graph_context::Entity::find_by_id(&id)
        .one(db)
        .await
        .map_err(map_db_err)?;

    if let Some(existing) = existing {
        let mut active: entity::graph_context::ActiveModel = existing.into();
        active.context = Set(context.to_string());
        active.updated_at = Set(Utc::now());
        active.update(db).await.map_err(map_db_err)?;
    } else {
        let model = entity::graph_context::ActiveModel {
            id: Set(id),
            session_id: Set(session_id.to_string()),
            user_id: Set(norm_user_id_opt(user_id)),
            context: Set(context.to_string()),
            updated_at: Set(Utc::now()),
        };
        model.insert(db).await.map_err(map_db_err)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sea_orm::{ConnectOptions, ConnectionTrait, Database};

    /// In-memory sqlite with the production table shapes. `:memory:` is
    /// per-connection, so the pool is pinned to one connection for the
    /// schema to be visible to every query.
    async fn setup() -> DatabaseConnection {
        let mut opt = ConnectOptions::new("sqlite::memory:".to_string());
        opt.max_connections(1);
        let db = Database::connect(opt).await.unwrap();
        db.execute_unprepared(
            "CREATE TABLE session_qa_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TEXT NOT NULL,
                feedback_text TEXT,
                feedback_score INTEGER,
                used_graph_element_ids TEXT,
                memify_metadata TEXT
            )",
        )
        .await
        .unwrap();
        db.execute_unprepared(
            "CREATE TABLE session_graph_context (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                context TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .await
        .unwrap();
        db.execute_unprepared(
            "CREATE TABLE session_trace_steps (
                trace_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                origin_function TEXT NOT NULL,
                status TEXT NOT NULL,
                memory_query TEXT NOT NULL,
                memory_context TEXT NOT NULL,
                method_params TEXT NOT NULL,
                method_return_value TEXT,
                error_message TEXT NOT NULL,
                session_feedback TEXT NOT NULL
            )",
        )
        .await
        .unwrap();
        db
    }

    // The remember write path stores `Uuid::to_string()` (hyphenated) while
    // the session routers read with the hex form; the store boundary
    // normalizes both sides, so a row written in either format is visible to
    // a read in the other. Regression for the `qas: []`-despite-rows bug on
    // /sessions/{id}.
    #[tokio::test]
    async fn qa_entries_round_trip_across_user_id_formats() {
        let db = setup().await;
        let hex = "7d8f1beccc275d17ab84fe1f058cf3fa";
        let hyphenated = "7d8f1bec-cc27-5d17-ab84-fe1f058cf3fa";

        create_qa_entry(&db, Uuid::new_v4(), "s1", Some(hyphenated), "q", "a", None)
            .await
            .unwrap();
        let rows = get_latest_entries(&db, "s1", Some(hex), 10).await.unwrap();
        assert_eq!(rows.len(), 1, "hex read must see the hyphenated write");
        assert_eq!(rows[0].user_id.as_deref(), Some(hex), "storage keeps hex");

        create_qa_entry(&db, Uuid::new_v4(), "s2", Some(hex), "q", "a", None)
            .await
            .unwrap();
        let rows = get_all_entries(&db, "s2", Some(hyphenated)).await.unwrap();
        assert_eq!(rows.len(), 1, "hyphenated read must see the hex write");
    }

    #[tokio::test]
    async fn trace_steps_round_trip_across_user_id_formats() {
        let db = setup().await;
        let hex = "7d8f1beccc275d17ab84fe1f058cf3fa";
        let hyphenated = "7d8f1bec-cc27-5d17-ab84-fe1f058cf3fa";

        let step = SessionTraceStep {
            trace_id: "t1".into(),
            origin_function: "f".into(),
            status: "success".into(),
            memory_query: String::new(),
            memory_context: String::new(),
            method_params: serde_json::json!({}),
            method_return_value: None,
            error_message: String::new(),
            session_feedback: String::new(),
        };
        save_trace_step(&db, hyphenated, "s1", step).await.unwrap();
        let steps = read_trace_steps(&db, hex, "s1").await.unwrap();
        assert_eq!(steps.len(), 1, "hex read must see the hyphenated trace write");
    }

    #[tokio::test]
    async fn graph_context_round_trips_across_user_id_formats() {
        let db = setup().await;
        let hex = "7d8f1beccc275d17ab84fe1f058cf3fa";
        let hyphenated = "7d8f1bec-cc27-5d17-ab84-fe1f058cf3fa";

        set_graph_context(&db, "s1", Some(hyphenated), "ctx").await.unwrap();
        assert_eq!(
            get_graph_context(&db, "s1", Some(hex)).await.unwrap().as_deref(),
            Some("ctx")
        );
    }
}
