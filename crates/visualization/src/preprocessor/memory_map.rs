//! Memory-tab payload builder.
//!
//! Port of `_build_memory_map()` and its helpers from
//! `cognee/modules/visualization/preprocessor.py:896-1175`.
//!
//! The payload carries *structure only*: ids, deterministic ordering, grouping,
//! an index of structural edges (integer positions into the links array) and
//! the run timeline. Node/link details are read at render time from the Story
//! view's already-embedded data (`window._vizNodeById` / `window._vizLinks`),
//! so nothing is duplicated here.
//!
//! Deterministic-position rule: every list is sorted by keys intrinsic to the
//! data (`chunk_index`, names, ids, `t_created`), so the layout the JS derives
//! from it is reproducible and append-stable as the graph grows.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::{Map, Value};

use super::{ENTITY_TYPE_RELATION, is_truthy, link_relation, py_str};

/// Gap (ms) between consecutive `t_created` values beyond which the timeline
/// starts a new run event. 5 minutes cleanly separates pipeline runs while
/// merging the sub-second spread within one cognify batch
/// (`preprocessor.py:900`).
pub(crate) const MEMORY_TIMELINE_GAP_MS: i64 = 300_000;

/// Members per entity group flagged `important` even when they did not earn a
/// Key-mode `label_priority` slot (`preprocessor.py:905`).
pub(crate) const MEMORY_GROUP_TOP_MEMBERS: usize = 8;

/// `node["id"]` — always present on preprocessed nodes.
fn id_of(node: &Value) -> &str {
    node.get("id").and_then(Value::as_str).unwrap_or("")
}

/// `node["stage"]` — always present on preprocessed nodes.
fn stage_of(node: &Value) -> &str {
    node.get("stage").and_then(Value::as_str).unwrap_or("")
}

/// `node.get("name")` with Python's `or ""` fallback for sorting.
fn name_of(node: &Value) -> &str {
    node.get("name").and_then(Value::as_str).unwrap_or("")
}

/// `node.get("t_created")` — `None` for legacy nodes with no timestamp.
fn t_created_of(node: &Value) -> Option<i64> {
    node.get("t_created").and_then(Value::as_i64)
}

/// `node.get("chunk_index")` restricted to real integers, mirroring Python's
/// `isinstance(index, int) and not isinstance(index, bool)`
/// (`preprocessor.py:1002-1004`).
fn chunk_index_of(node: &Value) -> Option<i64> {
    match node.get("chunk_index") {
        Some(Value::Number(number)) => number.as_i64(),
        _ => None,
    }
}

/// `node.get("importance") or 0.0`.
fn importance_of(node: &Value) -> f64 {
    node.get("importance")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

/// `bool(node.get("label_priority"))`.
fn label_priority_of(node: &Value) -> bool {
    node.get("label_priority")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// `str(node.get(key))` for id-like properties, `None` when missing or null
/// (mirrors `str(raw) if raw is not None else None`).
fn optional_id_string(node: &Value, key: &str) -> Option<String> {
    match node.get(key) {
        None | Some(Value::Null) => None,
        Some(value) => Some(py_str(value)),
    }
}

/// Sort key tolerant of missing timestamps: untimed values sort last
/// (`_t_sort_key`, `preprocessor.py:908-910`).
fn t_sort_key(t_created: Option<i64>) -> (bool, i64) {
    (t_created.is_none(), t_created.unwrap_or(0))
}

/// Build the Memory-tab payload (embedded via the `__MEMORY_DATA__` token).
///
/// `nodes` / `links` are the already-enriched values produced by
/// [`super::preprocess`], so `id`, `stage`, `edge_class`, `source_stage` and
/// `target_stage` are guaranteed present.
pub(crate) fn build(nodes: &[Value], links: &[Value]) -> Value {
    let nodes_by_id: HashMap<&str, &Value> = nodes.iter().map(|node| (id_of(node), node)).collect();

    let by_stage = |stage: &'static str| -> Vec<&Value> {
        nodes
            .iter()
            .filter(|node| stage_of(node) == stage)
            .collect()
    };
    let doc_nodes = by_stage("document");
    let chunk_nodes = by_stage("chunk");
    let entity_nodes = by_stage("entity");
    let summary_nodes = by_stage("summary");
    let context_nodes = by_stage("context");

    // A local `fn` rather than a closure: the returned `&str` borrows from the
    // `Value`s (`'v`), not from the slice that holds the references, and closure
    // lifetime elision cannot express that distinction.
    fn ids_of<'v>(list: &[&'v Value]) -> HashSet<&'v str> {
        list.iter().copied().map(id_of).collect()
    }
    let doc_ids = ids_of(&doc_nodes);
    let chunk_ids = ids_of(&chunk_nodes);
    let entity_ids = ids_of(&entity_nodes);
    let summary_ids = ids_of(&summary_nodes);
    let context_ids = ids_of(&context_nodes);

    // ── Structural edge index + relation maps (single pass over links) ───────
    let mut edge_index: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for bucket in [
        "contains",
        "made_from",
        "is_part_of",
        "summarized_in",
        "semantic",
    ] {
        edge_index.insert(bucket, Vec::new());
    }
    let mut chunk_doc_via_edge: HashMap<&str, &str> = HashMap::new();
    let mut summary_chunks: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut bucket_children: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut members_by_type: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();

    for (position, link) in links.iter().enumerate() {
        let source = link.get("source").and_then(Value::as_str).unwrap_or("");
        let target = link.get("target").and_then(Value::as_str).unwrap_or("");
        let relation = link_relation(link).to_lowercase();
        let source_stage = link.get("source_stage").and_then(Value::as_str);
        let target_stage = link.get("target_stage").and_then(Value::as_str);
        // Python compares the *set* {source_stage, target_stage} to
        // {"document", "chunk"}, so one endpoint must be each.
        let doc_chunk_pair = matches!(
            (source_stage, target_stage),
            (Some("document"), Some("chunk")) | (Some("chunk"), Some("document"))
        );

        // The order of this chain is load-bearing (Python's elif ladder).
        if relation == "is_part_of"
            || relation == "part_of"
            || (relation == "contains" && doc_chunk_pair)
        {
            // Chunk↔document membership: modern graphs use is_part_of
            // (chunk→doc); some legacy graphs use contains (doc→chunk).
            push_edge(&mut edge_index, "is_part_of", position);
            let chunk_end = if chunk_ids.contains(source) {
                source
            } else {
                target
            };
            let doc_end = if doc_ids.contains(target) {
                target
            } else {
                source
            };
            if chunk_ids.contains(chunk_end) && doc_ids.contains(doc_end) {
                // First-wins, like Python's `setdefault`.
                chunk_doc_via_edge.entry(chunk_end).or_insert(doc_end);
            }
        } else if relation == "contains" {
            push_edge(&mut edge_index, "contains", position);
        } else if relation == "made_from" {
            push_edge(&mut edge_index, "made_from", position);
            let summary_end = if summary_ids.contains(source) {
                source
            } else {
                target
            };
            let chunk_end = if chunk_ids.contains(target) {
                target
            } else {
                source
            };
            if summary_ids.contains(summary_end) && chunk_ids.contains(chunk_end) {
                summary_chunks
                    .entry(summary_end)
                    .or_default()
                    .insert(chunk_end);
            }
        } else if relation == "summarized_in" {
            push_edge(&mut edge_index, "summarized_in", position);
            if context_ids.contains(target) {
                bucket_children.entry(target).or_default().insert(source);
            }
        } else if relation == ENTITY_TYPE_RELATION {
            // Entity → EntityType grouping edge. Entity→Entity is_a edges do
            // NOT group (only true EntityType targets count).
            let target_is_entity_type = nodes_by_id
                .get(target)
                .and_then(|node| node.get("type"))
                .and_then(Value::as_str)
                == Some("EntityType");
            if entity_ids.contains(source) && target_is_entity_type {
                members_by_type.entry(target).or_default().insert(source);
            }
        } else if link.get("edge_class").and_then(Value::as_str) == Some("semantic")
            && entity_ids.contains(source)
            && entity_ids.contains(target)
        {
            push_edge(&mut edge_index, "semantic", position);
        }
    }

    // ── Documents with ordered chunk cells ───────────────────────────────────
    let mut chunks_by_doc: HashMap<String, Vec<&Value>> = HashMap::new();
    let mut legacy_chunks_by_doc: HashMap<String, Vec<&Value>> = HashMap::new();
    let mut orphan_chunks: Vec<&str> = Vec::new();
    for chunk in &chunk_nodes {
        let doc_id = optional_id_string(chunk, "document_id");
        match doc_id {
            Some(doc_id) if doc_ids.contains(doc_id.as_str()) => {
                chunks_by_doc.entry(doc_id).or_default().push(chunk);
            }
            _ => match chunk_doc_via_edge.get(id_of(chunk)) {
                Some(doc_end) => legacy_chunks_by_doc
                    .entry((*doc_end).to_string())
                    .or_default()
                    .push(chunk),
                None => orphan_chunks.push(id_of(chunk)),
            },
        }
    }
    orphan_chunks.sort_unstable();

    let mut documents: Vec<Value> = Vec::with_capacity(doc_nodes.len());
    for doc in &doc_nodes {
        let mut primary = chunks_by_doc.get(id_of(doc)).cloned().unwrap_or_default();
        sort_chunks(&mut primary);
        // Legacy chunks (attributed only via the is_part_of edge) append after
        // the chunk_index-ordered run so existing cells never shift.
        let mut legacy = legacy_chunks_by_doc
            .get(id_of(doc))
            .cloned()
            .unwrap_or_default();
        sort_chunks(&mut legacy);
        let ordered: Vec<&Value> = primary.into_iter().chain(legacy).collect();

        let t_first = std::iter::once(t_created_of(doc))
            .chain(ordered.iter().map(|chunk| t_created_of(chunk)))
            .flatten()
            .min();

        let mut document = Map::new();
        document.insert("id".to_string(), Value::String(id_of(doc).to_string()));
        let display_name = match doc.get("name") {
            Some(name) if is_truthy(name) => py_str(name),
            _ => id_of(doc).to_string(),
        };
        document.insert("name".to_string(), Value::String(display_name));
        document.insert("t_first".to_string(), optional_int(t_first));
        document.insert(
            "chunks".to_string(),
            Value::Array(ordered.iter().map(|chunk| chunk_cell(chunk)).collect()),
        );
        documents.push(Value::Object(document));
    }
    documents.sort_by(|left, right| {
        let key = |document: &Value| {
            let t_first = document.get("t_first").and_then(Value::as_i64);
            (
                t_sort_key(t_first),
                document
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                document
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            )
        };
        key(left).cmp(&key(right))
    });

    // ── Entity groups (one per EntityType node) ──────────────────────────────
    let mut type_nodes: Vec<&Value> = nodes
        .iter()
        .filter(|node| node.get("type").and_then(Value::as_str) == Some("EntityType"))
        .collect();
    type_nodes
        .sort_by(|left, right| (name_of(left), id_of(left)).cmp(&(name_of(right), id_of(right))));

    let mut entity_groups: Vec<Value> = Vec::with_capacity(type_nodes.len());
    let mut grouped_entity_ids: HashSet<&str> = HashSet::new();
    for type_node in &type_nodes {
        let mut member_nodes: Vec<&Value> = members_by_type
            .get(id_of(type_node))
            .map(|members| {
                members
                    .iter()
                    .filter_map(|member_id| nodes_by_id.get(member_id).copied())
                    .collect()
            })
            .unwrap_or_default();
        // (not label_priority, -importance, name, id)
        member_nodes.sort_by(|left, right| {
            label_priority_of(right)
                .cmp(&label_priority_of(left))
                .then(importance_of(right).total_cmp(&importance_of(left)))
                .then(name_of(left).cmp(name_of(right)))
                .then(id_of(left).cmp(id_of(right)))
        });
        grouped_entity_ids.extend(member_nodes.iter().map(|node| id_of(node)));

        let members: Vec<Value> = member_nodes
            .iter()
            .enumerate()
            .map(|(rank, member)| {
                let mut entry = Map::new();
                entry.insert("id".to_string(), Value::String(id_of(member).to_string()));
                entry.insert(
                    "important".to_string(),
                    Value::Bool(label_priority_of(member) || rank < MEMORY_GROUP_TOP_MEMBERS),
                );
                Value::Object(entry)
            })
            .collect();

        let mut group = Map::new();
        group.insert(
            "type_id".to_string(),
            Value::String(id_of(type_node).to_string()),
        );
        let type_name = match type_node.get("name") {
            Some(name) if is_truthy(name) => py_str(name),
            _ => id_of(type_node).to_string(),
        };
        group.insert("type_name".to_string(), Value::String(type_name));
        group.insert("members".to_string(), Value::Array(members));
        entity_groups.push(Value::Object(group));
    }

    let mut ungrouped: Vec<&Value> = entity_nodes
        .iter()
        .copied()
        .filter(|node| !grouped_entity_ids.contains(id_of(node)))
        .collect();
    ungrouped
        .sort_by(|left, right| (name_of(left), id_of(left)).cmp(&(name_of(right), id_of(right))));
    let ungrouped_entities: Vec<Value> = ungrouped
        .iter()
        .map(|node| Value::String(id_of(node).to_string()))
        .collect();

    // ── Summaries ────────────────────────────────────────────────────────────
    let mut sorted_summaries = summary_nodes.clone();
    sorted_summaries.sort_by(|left, right| {
        (t_sort_key(t_created_of(left)), id_of(left))
            .cmp(&(t_sort_key(t_created_of(right)), id_of(right)))
    });
    let summaries: Vec<Value> = sorted_summaries
        .iter()
        .map(|summary| {
            let mut entry = Map::new();
            entry.insert("id".to_string(), Value::String(id_of(summary).to_string()));
            entry.insert(
                "chunk_ids".to_string(),
                Value::Array(
                    summary_chunks
                        .get(id_of(summary))
                        .map(|chunks| {
                            chunks
                                .iter()
                                .map(|chunk| Value::String((*chunk).to_string()))
                                .collect()
                        })
                        .unwrap_or_default(),
                ),
            );
            entry.insert(
                "bucket_id".to_string(),
                optional_id_string(summary, "global_context_bucket_id")
                    .map_or(Value::Null, Value::String),
            );
            Value::Object(entry)
        })
        .collect();

    // ── Global context (null → the view renders its empty state) ────────────
    let context = if context_nodes.is_empty() {
        Value::Null
    } else {
        let mut root_ids: Vec<&str> = context_nodes
            .iter()
            .filter(|node| node.get("is_root").is_some_and(is_truthy))
            .map(|node| id_of(node))
            .collect();
        root_ids.sort_unstable();

        let mut sorted_context = context_nodes.clone();
        sorted_context.sort_by(|left, right| {
            (level_of(left).unwrap_or(-1), id_of(left))
                .cmp(&(level_of(right).unwrap_or(-1), id_of(right)))
        });
        let buckets: Vec<Value> = sorted_context
            .iter()
            .map(|node| {
                let mut bucket = Map::new();
                bucket.insert("id".to_string(), Value::String(id_of(node).to_string()));
                bucket.insert("level".to_string(), optional_int(level_of(node)));
                bucket.insert(
                    "child_ids".to_string(),
                    Value::Array(
                        bucket_children
                            .get(id_of(node))
                            .map(|children| {
                                children
                                    .iter()
                                    .map(|child| Value::String((*child).to_string()))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    ),
                );
                Value::Object(bucket)
            })
            .collect();

        let mut context = Map::new();
        context.insert(
            "root_id".to_string(),
            root_ids
                .first()
                .map_or(Value::Null, |id| Value::String((*id).to_string())),
        );
        context.insert("buckets".to_string(), Value::Array(buckets));
        Value::Object(context)
    };

    // ── Timeline: gap-cluster t_created into run events ──────────────────────
    let mut timed: Vec<&Value> = nodes
        .iter()
        .filter(|node| t_created_of(node).is_some())
        .collect();
    timed.sort_by(|left, right| {
        (t_created_of(left).unwrap_or(0), id_of(left))
            .cmp(&(t_created_of(right).unwrap_or(0), id_of(right)))
    });
    let mut untimed_ids: Vec<&str> = nodes
        .iter()
        .filter(|node| t_created_of(node).is_none())
        .map(id_of)
        .collect();
    untimed_ids.sort_unstable();

    let mut clusters: Vec<Vec<&Value>> = Vec::new();
    for node in timed {
        let t = t_created_of(node).unwrap_or(0);
        // The gap is measured against the *previous member*, so a chain of
        // 5-minute steps merges into a single event.
        let joins_last = clusters
            .last()
            .and_then(|cluster| cluster.last().copied())
            .and_then(t_created_of)
            .is_some_and(|previous| t - previous <= MEMORY_TIMELINE_GAP_MS);
        if joins_last {
            if let Some(cluster) = clusters.last_mut() {
                cluster.push(node);
            }
        } else {
            clusters.push(vec![node]);
        }
    }

    let mut timeline: Vec<Value> = Vec::with_capacity(clusters.len());
    for (index, cluster) in clusters.iter().enumerate() {
        let mut node_ids: Vec<&str> = cluster.iter().map(|node| id_of(node)).collect();
        if index == 0 && !untimed_ids.is_empty() {
            // Nodes without t_created (legacy/defensive) join the first event.
            let mut merged = untimed_ids.clone();
            merged.extend(node_ids);
            node_ids = merged;
        }

        let label = if cluster.iter().any(|node| stage_of(node) == "context") {
            "global_context_index".to_string()
        } else {
            modal_pipeline(cluster).unwrap_or_else(|| "ingestion".to_string())
        };

        let mut event = Map::new();
        event.insert("index".to_string(), Value::from(index));
        event.insert("kind".to_string(), Value::String("run".to_string()));
        event.insert("label".to_string(), Value::String(label));
        event.insert(
            "t0".to_string(),
            optional_int(cluster.first().and_then(|node| t_created_of(node))),
        );
        event.insert(
            "t1".to_string(),
            optional_int(cluster.last().and_then(|node| t_created_of(node))),
        );
        event.insert("node_count".to_string(), Value::from(node_ids.len()));
        event.insert(
            "node_ids".to_string(),
            Value::Array(
                node_ids
                    .iter()
                    .map(|id| Value::String((*id).to_string()))
                    .collect(),
            ),
        );
        timeline.push(Value::Object(event));
    }
    if timeline.is_empty() && !untimed_ids.is_empty() {
        // No node carries t_created at all: emit one synthetic event so the
        // view always has a "current state" selection on non-empty graphs.
        let mut event = Map::new();
        event.insert("index".to_string(), Value::from(0));
        event.insert("kind".to_string(), Value::String("run".to_string()));
        event.insert("label".to_string(), Value::String("ingestion".to_string()));
        event.insert("t0".to_string(), Value::from(0));
        event.insert("t1".to_string(), Value::from(0));
        event.insert("node_count".to_string(), Value::from(untimed_ids.len()));
        event.insert(
            "node_ids".to_string(),
            Value::Array(
                untimed_ids
                    .iter()
                    .map(|id| Value::String((*id).to_string()))
                    .collect(),
            ),
        );
        timeline.push(Value::Object(event));
    }

    // ── Payload (exactly the eight keys of preprocessor.py:1166-1175) ────────
    let mut edges = Map::new();
    for (bucket, positions) in edge_index {
        edges.insert(
            bucket.to_string(),
            Value::Array(positions.into_iter().map(Value::from).collect()),
        );
    }

    let mut payload = Map::new();
    payload.insert("documents".to_string(), Value::Array(documents));
    payload.insert(
        "orphan_chunks".to_string(),
        Value::Array(
            orphan_chunks
                .iter()
                .map(|id| Value::String((*id).to_string()))
                .collect(),
        ),
    );
    payload.insert("entity_groups".to_string(), Value::Array(entity_groups));
    payload.insert(
        "ungrouped_entities".to_string(),
        Value::Array(ungrouped_entities),
    );
    payload.insert("summaries".to_string(), Value::Array(summaries));
    payload.insert("context".to_string(), context);
    payload.insert("edges".to_string(), Value::Object(edges));
    payload.insert("timeline".to_string(), Value::Array(timeline));
    Value::Object(payload)
}

/// Append a link position to one of the five structural buckets.
fn push_edge(edge_index: &mut BTreeMap<&str, Vec<usize>>, bucket: &str, position: usize) {
    if let Some(positions) = edge_index.get_mut(bucket) {
        positions.push(position);
    }
}

/// `node.get("level") if isinstance(level, int) else None`.
fn level_of(node: &Value) -> Option<i64> {
    match node.get("level") {
        Some(Value::Number(number)) => number.as_i64(),
        _ => None,
    }
}

/// `Some(v)` → JSON number, `None` → JSON null.
fn optional_int(value: Option<i64>) -> Value {
    value.map_or(Value::Null, Value::from)
}

/// `_chunk_sort_key()` — `(index is None, index or 0, *_t_sort_key(t), id)`
/// (`preprocessor.py:1006-1008`).
fn sort_chunks(chunks: &mut [&Value]) {
    chunks.sort_by(|left, right| {
        let key = |chunk: &Value| {
            let index = chunk_index_of(chunk);
            (
                index.is_none(),
                index.unwrap_or(0),
                t_sort_key(t_created_of(chunk)),
                id_of(chunk).to_string(),
            )
        };
        key(left).cmp(&key(right))
    });
}

/// `_chunk_cell()` (`preprocessor.py:1010-1015`).
fn chunk_cell(chunk: &Value) -> Value {
    let mut cell = Map::new();
    cell.insert("id".to_string(), Value::String(id_of(chunk).to_string()));
    cell.insert(
        "chunk_index".to_string(),
        optional_int(chunk_index_of(chunk)),
    );
    cell.insert("t_created".to_string(), optional_int(t_created_of(chunk)));
    Value::Object(cell)
}

/// The cluster's dominant `source_pipeline`, chosen by Python's
/// `min(counts.items(), key=lambda kv: (-count, name))` — highest count wins,
/// ties break on the name (never on iteration order).
fn modal_pipeline(cluster: &[&Value]) -> Option<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for node in cluster {
        if let Some(pipeline) = node.get("source_pipeline").and_then(Value::as_str)
            && !pipeline.is_empty()
        {
            *counts.entry(pipeline).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        // BTreeMap iterates in name order, so `max_by_key` on the count alone
        // would pick the *last* name on a tie; fold with a strict `>` instead.
        .fold(
            None,
            |best: Option<(&str, usize)>, (name, count)| match best {
                Some((_, best_count)) if best_count >= count => best,
                _ => Some((name, count)),
            },
        )
        .map(|(name, _)| name.to_string())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn modal_pipeline_breaks_ties_on_name() {
        let nodes = [
            json!({"source_pipeline": "zeta"}),
            json!({"source_pipeline": "alpha"}),
        ];
        let cluster: Vec<&Value> = nodes.iter().collect();
        assert_eq!(modal_pipeline(&cluster), Some("alpha".to_string()));

        let nodes = [
            json!({"source_pipeline": "alpha"}),
            json!({"source_pipeline": "zeta"}),
            json!({"source_pipeline": "zeta"}),
        ];
        let cluster: Vec<&Value> = nodes.iter().collect();
        assert_eq!(modal_pipeline(&cluster), Some("zeta".to_string()));

        assert_eq!(modal_pipeline(&[]), None);
    }

    #[test]
    fn chunk_cell_shape() {
        let chunk = json!({"id": "c1", "chunk_index": 2, "t_created": 5});
        assert_eq!(
            chunk_cell(&chunk),
            json!({"id": "c1", "chunk_index": 2, "t_created": 5})
        );
        let bare = json!({"id": "c2", "chunk_index": true});
        assert_eq!(
            chunk_cell(&bare),
            json!({"id": "c2", "chunk_index": null, "t_created": null})
        );
    }

    #[test]
    fn empty_graph_payload_shape() {
        let payload = build(&[], &[]);
        assert_eq!(
            payload,
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
}
