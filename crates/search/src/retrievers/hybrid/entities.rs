//! Entity neighborhood → edge-bullet rendering for the hybrid retriever.
//!
//! Port of `cognee/modules/retrieval/hybrid/entities.py`. Turns already-fetched
//! `Entity_name` vector hits plus their one-hop graph neighborhood into the
//! per-entity edge-bullet blocks the hybrid retriever renders into the LLM
//! prompt. Only [`build_entities`]'s `get_neighborhood` call is I/O; the
//! dedupe / sort / truncate / formatting logic is pure.
//!
//! # Known limitation — `belongs_to_set` is not inherited by entities
//!
//! Rust's `expand_with_nodes_and_edges` port constructs entities with
//! `Entity::new(&node.name, None, &node.name, Some(dataset_id))`
//! (`crates/cognify/src/graph_integration/expansion.rs:471`), which never
//! propagates the source chunk's NodeSet names into `belongs_to_set`; that
//! field only carries the dataset-UUID default from the `DataPoint` base
//! constructor (`crates/models/src/data_point.rs:135`). A dataset-UUID string
//! never matches a `node_name` filter of NodeSet names, so on this workspace a
//! `node_name`-scoped hybrid search's `Entity_name` query structurally returns
//! zero entity hits — a strictly narrower miss than Python, whose entities
//! inherit `data_chunk.belongs_to_set` and therefore survive a scoped search
//! (only facts are hidden). This is out of scope here and tracked as a
//! follow-up backfill task, not fixed in this module.

use std::collections::{HashMap, HashSet};

use cognee_graph::{EdgeData, GraphDBTrait, GraphNode};
use serde_json::{Map, Value};

use super::facts::{EdgeLite, connection_edge_type_id};
use super::results::{display_value, first_display_value, payload, result_id};
use crate::types::SearchItem;

/// A node in a rebuilt connection triple: a JSON object carrying at least an
/// `"id"` (Python's `nodes_by_id.get(id, {"id": id})` dict).
type NodeLite = Value;

/// A resolved entity plus its ranked edge bullets.
///
/// Port of the Python entity dict (`_entity_from_result`, `entities.py:86-98`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EntityResult {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub entity_type: Option<String>,
    pub edges: Vec<EdgeBullet>,
}

/// A single rendered edge bullet for an entity.
///
/// Port of the Python edge dict (`_edge_bullet`, `entities.py:183-201`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EdgeBullet {
    pub text: String,
    pub source: Option<String>,
    pub target: Option<String>,
    pub source_id: Option<String>,
    pub relationship: Option<String>,
    pub target_id: Option<String>,
    pub edge_type_id: Option<String>,
}

/// Build the entity blocks for the given `Entity_name` hits.
///
/// Port of `build_entities` (`entities.py:15-44`). Returns `[]` for no hits.
/// Builds one entity per hit; if no hit has a nonempty id, returns the entities
/// unedited (no neighborhood call). Otherwise fetches the one-hop neighborhood
/// and attaches ranked edge bullets. **Fail-open:** a `get_neighborhood` error
/// is logged and the entities are returned with empty edges — the error is
/// never propagated (mirrors Python's `try/except` returning bare entities).
pub(crate) async fn build_entities(
    graph: &dyn GraphDBTrait,
    entity_hits: &[SearchItem],
    max_edges_per_entity: usize,
    edge_ranks: &HashMap<String, usize>,
) -> Vec<EntityResult> {
    if entity_hits.is_empty() {
        return vec![];
    }

    let mut entities: Vec<EntityResult> = entity_hits.iter().map(entity_from_result).collect();
    let entity_ids: Vec<String> = entities
        .iter()
        .filter(|entity| !entity.id.is_empty())
        .map(|entity| entity.id.clone())
        .collect();
    if entity_ids.is_empty() {
        return entities;
    }

    let (nodes, edges) = match graph.get_neighborhood(&entity_ids, 1).await {
        Ok(neighborhood) => neighborhood,
        Err(error) => {
            tracing::warn!(
                %error,
                "Graph neighborhood retrieval failed; returning entities without edges"
            );
            return entities;
        }
    };

    let connections_by_entity_id = partition_neighborhood(&entity_ids, nodes, edges);
    for entity in &mut entities {
        let connections = connections_by_entity_id
            .get(&entity.id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        entity.edges = edge_bullets_from_connections(connections, max_edges_per_entity, edge_ranks);
    }
    entities
}

/// Resolve a single entity hit into an [`EntityResult`] with empty edges.
///
/// Port of `_entity_from_result` (`entities.py:86-98`). `name` falls back
/// `name` → `text` → `id`, never empty.
fn entity_from_result(item: &SearchItem) -> EntityResult {
    let result_payload = payload(item);
    let entity_id = result_id(item).unwrap_or_default();

    let id_value = Value::String(entity_id.clone());
    let mut name_candidates: Vec<&Value> = Vec::new();
    if let Some(value) = result_payload.get("name") {
        name_candidates.push(value);
    }
    if let Some(value) = result_payload.get("text") {
        name_candidates.push(value);
    }
    name_candidates.push(&id_value);
    let name = first_display_value(&name_candidates).unwrap_or_default();

    EntityResult {
        id: entity_id,
        name,
        description: result_payload.get("description").and_then(display_value),
        entity_type: entity_type(result_payload),
        edges: vec![],
    }
}

/// Resolve an entity's domain type from a payload/entity object.
///
/// Port of `_entity_type` (`entities.py:122-127`): the first nonblank
/// `display_value` of `is_a` then `type`, suppressing the literal
/// `"IndexSchema"` structural type; else `None`.
fn entity_type(result_payload: &Value) -> Option<String> {
    for key in ["is_a", "type"] {
        if let Some(value) = result_payload.get(key)
            && let Some(entity_type) = display_value(value)
            && entity_type != "IndexSchema"
        {
            return Some(entity_type);
        }
    }
    None
}

/// Rebuild per-entity `(source, edge, target)` connection triples from the flat
/// one-hop subgraph.
///
/// Port of `_partition_neighborhood` (`entities.py:47-72`). Each seed id gets an
/// (initially empty) connection list. A triple is pushed onto its `source_id`'s
/// list (when that id is a seed) and onto its `target_id`'s list (when that id
/// is a seed **and** differs from `source_id`, deduping self-loops). Edges with
/// neither endpoint among the seeds (neighbor-to-neighbor) are silently dropped.
fn partition_neighborhood(
    entity_ids: &[String],
    nodes: Vec<GraphNode>,
    edges: Vec<EdgeData>,
) -> HashMap<String, Vec<(NodeLite, EdgeLite, NodeLite)>> {
    let mut nodes_by_id: HashMap<String, Value> = HashMap::new();
    for (id, data) in nodes {
        let mut object = Map::new();
        object.insert("id".to_string(), Value::String(id.clone()));
        for (key, value) in data {
            object.insert(key.into_owned(), value);
        }
        nodes_by_id.insert(id, Value::Object(object));
    }

    let mut connections: HashMap<String, Vec<(NodeLite, EdgeLite, NodeLite)>> = entity_ids
        .iter()
        .map(|id| (id.clone(), Vec::new()))
        .collect();

    for (source_id, target_id, relationship_name, properties) in edges {
        let source = nodes_by_id
            .get(&source_id)
            .cloned()
            .unwrap_or_else(|| Value::Object(id_only_object(&source_id)));
        let target = nodes_by_id
            .get(&target_id)
            .cloned()
            .unwrap_or_else(|| Value::Object(id_only_object(&target_id)));

        let mut property_object = Map::new();
        for (key, value) in properties {
            property_object.insert(key.into_owned(), value);
        }
        let edge = EdgeLite {
            relationship_name: Some(Value::String(relationship_name)),
            edge_text: None,
            properties: Some(Value::Object(property_object)),
        };
        let triple = (source, edge, target);

        let push_source = connections.contains_key(&source_id);
        let push_target = connections.contains_key(&target_id) && target_id != source_id;
        match (push_source, push_target) {
            (true, true) => {
                if let Some(list) = connections.get_mut(&source_id) {
                    list.push(triple.clone());
                }
                if let Some(list) = connections.get_mut(&target_id) {
                    list.push(triple);
                }
            }
            (true, false) => {
                if let Some(list) = connections.get_mut(&source_id) {
                    list.push(triple);
                }
            }
            (false, true) => {
                if let Some(list) = connections.get_mut(&target_id) {
                    list.push(triple);
                }
            }
            (false, false) => {}
        }
    }
    connections
}

/// A `{"id": id}` JSON object for a node absent from the neighborhood.
fn id_only_object(id: &str) -> Map<String, Value> {
    let mut object = Map::new();
    object.insert("id".to_string(), Value::String(id.to_string()));
    object
}

/// Build ranked, deduped, capped edge bullets for one entity's connections.
///
/// Port of `_edge_bullets_from_connections` (`entities.py:130-161`). `max_edges
/// == 0` yields `[]`. Empty-text bullets are skipped. Dedupe runs on two
/// **independent** tracks: a keyed `(source_id, relationship, target_id)` set
/// and a text-only set — a keyed bullet is never checked against the text set
/// or vice versa. A **stable** sort by [`edge_sort_key`] preserves connection
/// order among equal keys, then the list is truncated to `max_edges`.
fn edge_bullets_from_connections(
    connections: &[(NodeLite, EdgeLite, NodeLite)],
    max_edges: usize,
    edge_ranks: &HashMap<String, usize>,
) -> Vec<EdgeBullet> {
    if max_edges == 0 {
        return vec![];
    }

    let mut edges: Vec<EdgeBullet> = Vec::new();
    let mut seen_keys: HashSet<(String, String, String)> = HashSet::new();
    let mut seen_texts: HashSet<String> = HashSet::new();

    for (source, edge, target) in connections {
        let Some(bullet) = edge_bullet(source, edge, target) else {
            continue;
        };

        let dedupe_key = edge_dedupe_key(&bullet);
        match &dedupe_key {
            Some(key) if seen_keys.contains(key) => continue,
            None if seen_texts.contains(&bullet.text) => continue,
            _ => {}
        }
        match dedupe_key {
            Some(key) => {
                seen_keys.insert(key);
            }
            None => {
                seen_texts.insert(bullet.text.clone());
            }
        }
        edges.push(bullet);
    }

    edges.sort_by_key(|edge| edge_sort_key(edge, edge_ranks));
    edges.truncate(max_edges);
    edges
}

/// Sort key: type edges first, then query-ranked edges, then legacy order.
///
/// Port of `_edge_sort_key` (`entities.py:164-171`): `(0, 0)` for a type edge;
/// `(1, rank)` when the edge's `edge_type_id` is in `edge_ranks`; else `(2, 0)`.
fn edge_sort_key(edge: &EdgeBullet, edge_ranks: &HashMap<String, usize>) -> (u8, usize) {
    if is_type_edge(edge) {
        return (0, 0);
    }
    match edge.edge_type_id.as_ref().and_then(|id| edge_ranks.get(id)) {
        Some(rank) => (1, *rank),
        None => (2, 0),
    }
}

/// Render a single connection triple into an [`EdgeBullet`], or `None` to drop.
///
/// Port of `_edge_bullet` (`entities.py:183-201`). Text prefers the top-level
/// `edge_text` (absent from graph triples in practice, kept for fidelity), then
/// the nested `properties.edge_text`, then a synthesized
/// `"{source} -- {relationship} -- {target}"` when all three labels are present;
/// if still empty the bullet is dropped. `edge_type_id` is recomputed via
/// [`connection_edge_type_id`] (edge-text-first), never from the raw
/// relationship name.
fn edge_bullet(source: &NodeLite, edge: &EdgeLite, target: &NodeLite) -> Option<EdgeBullet> {
    let source_label = node_label(source);
    let target_label = node_label(target);
    let relationship = edge.relationship_name.as_ref().and_then(display_value);

    let mut text = edge
        .edge_text
        .as_ref()
        .and_then(display_value)
        .or_else(|| nested_edge_text(edge));
    if text.is_none()
        && let (Some(source_label), Some(relationship), Some(target_label)) =
            (&source_label, &relationship, &target_label)
    {
        text = Some(format!(
            "{source_label} -- {relationship} -- {target_label}"
        ));
    }
    let text = text?;

    Some(EdgeBullet {
        text,
        source: source_label,
        target: target_label,
        source_id: source.get("id").and_then(display_value),
        relationship,
        target_id: target.get("id").and_then(display_value),
        edge_type_id: connection_edge_type_id(edge),
    })
}

/// The dedupe key for a bullet, or `None` when any component is blank.
///
/// Port of `_edge_dedupe_key` (`entities.py:204-210`).
fn edge_dedupe_key(edge: &EdgeBullet) -> Option<(String, String, String)> {
    match (&edge.source_id, &edge.relationship, &edge.target_id) {
        (Some(source_id), Some(relationship), Some(target_id)) => {
            Some((source_id.clone(), relationship.clone(), target_id.clone()))
        }
        _ => None,
    }
}

/// Whether a bullet is an `is a` / type edge.
///
/// Port of `_is_type_edge` (`entities.py:213-221`). The relationship is
/// normalized (lowercase, `_`/`-` → space, trimmed) and compared to `"is a"`;
/// otherwise the bullet text (lowercased and padded) is scanned for `" is a "`.
fn is_type_edge(edge: &EdgeBullet) -> bool {
    if let Some(relationship) = edge.relationship.as_deref() {
        let normalized = relationship.to_lowercase().replace(['_', '-'], " ");
        if normalized.trim() == "is a" {
            return true;
        }
    }
    let padded = format!(" {} ", edge.text.to_lowercase());
    padded.contains(" is a ")
}

/// The nested `properties.edge_text` of an edge, or `None`.
///
/// Port of `_nested_edge_text` (`entities.py:224-228`).
fn nested_edge_text(edge: &EdgeLite) -> Option<String> {
    edge.properties
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|map| map.get("edge_text"))
        .and_then(display_value)
}

/// A node's display label: its `name`, then its `id`.
///
/// Port of `_node_label` (`entities.py:231-233`).
fn node_label(node: &NodeLite) -> Option<String> {
    let mut candidates: Vec<&Value> = Vec::new();
    if let Some(value) = node.get("name") {
        candidates.push(value);
    }
    if let Some(value) = node.get("id") {
        candidates.push(value);
    }
    first_display_value(&candidates)
}

/// Render the entity blocks as the "Relevant entities" markdown section.
///
/// Port of `format_entities` (`entities.py:75-83`). Empty if no entity yields a
/// nonempty block; otherwise a `"## Relevant entities"` header followed by the
/// blocks joined by a blank line.
pub(crate) fn format_entities(entities: &[EntityResult]) -> String {
    let blocks: Vec<String> = entities
        .iter()
        .map(format_entity)
        .filter(|block| !block.is_empty())
        .collect();
    if blocks.is_empty() {
        return String::new();
    }
    format!("## Relevant entities\n{}", blocks.join("\n\n"))
}

/// Render a single entity block, or `""` when its name is blank.
///
/// Port of `_format_entity` (`entities.py:101-119`). Header is
/// `"### {name} ({type})"` or `"### {name}"` (the `IndexSchema` structural type
/// is suppressed), followed by the description line if present and one
/// `"- {text}"` per edge with nonblank text.
fn format_entity(entity: &EntityResult) -> String {
    let name = entity.name.trim();
    if name.is_empty() {
        return String::new();
    }

    let entity_type = entity
        .entity_type
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "IndexSchema");
    let header = match entity_type {
        Some(entity_type) => format!("### {name} ({entity_type})"),
        None => format!("### {name}"),
    };

    let mut lines = vec![header];
    if let Some(description) = entity.description.as_deref().map(str::trim)
        && !description.is_empty()
    {
        lines.push(description.to_string());
    }
    for edge in &entity.edges {
        let edge_text = edge.text.trim();
        if !edge_text.is_empty() {
            lines.push(format!("- {edge_text}"));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use std::borrow::Cow;

    use async_trait::async_trait;
    use cognee_graph::{EdgeData, GraphDBError, GraphDBResult, GraphNode, MockGraphDB, NodeData};
    use serde_json::json;

    use super::super::facts::edge_rank_by_id;
    use super::*;

    fn entity_hit(payload: Value) -> SearchItem {
        SearchItem {
            id: None,
            score: None,
            payload,
        }
    }

    fn edge_hit(text: &str) -> SearchItem {
        SearchItem {
            id: Some(cognee_models::EdgeType::deterministic_id(text)),
            score: None,
            payload: json!({ "text": text }),
        }
    }

    /// Insert a `{id, name}` node into a MockGraphDB.
    async fn add_node(graph: &MockGraphDB, id: &str, name: &str) {
        graph
            .add_node_raw(json!({ "id": id, "name": name }))
            .await
            .unwrap();
    }

    /// Insert a directed edge (with optional `edge_text` property).
    async fn add_edge(
        graph: &MockGraphDB,
        src: &str,
        tgt: &str,
        rel: &str,
        edge_text: Option<&str>,
    ) {
        let mut props: HashMap<Cow<'static, str>, Value> = HashMap::new();
        if let Some(text) = edge_text {
            props.insert(Cow::from("edge_text"), json!(text));
        }
        graph.add_edge(src, tgt, rel, Some(props)).await.unwrap();
    }

    /// A graph double whose `get_neighborhood` always errors (fail-open test).
    struct FailingGraphDB {
        inner: MockGraphDB,
    }

    impl FailingGraphDB {
        fn new() -> Self {
            Self {
                inner: MockGraphDB::new(),
            }
        }
    }

    #[async_trait]
    impl GraphDBTrait for FailingGraphDB {
        async fn get_neighborhood(
            &self,
            _node_ids: &[String],
            _depth: usize,
        ) -> GraphDBResult<(Vec<GraphNode>, Vec<EdgeData>)> {
            Err(GraphDBError::QueryError("graph unavailable".into()))
        }
        async fn initialize(&self) -> GraphDBResult<()> {
            self.inner.initialize().await
        }
        async fn is_empty(&self) -> GraphDBResult<bool> {
            self.inner.is_empty().await
        }
        async fn query(
            &self,
            q: &str,
            params: Option<HashMap<Cow<'static, str>, Value>>,
        ) -> GraphDBResult<Vec<Vec<Value>>> {
            self.inner.query(q, params).await
        }
        async fn delete_graph(&self) -> GraphDBResult<()> {
            self.inner.delete_graph().await
        }
        async fn has_node(&self, id: &str) -> GraphDBResult<bool> {
            self.inner.has_node(id).await
        }
        async fn add_node_raw(&self, node: Value) -> GraphDBResult<()> {
            self.inner.add_node_raw(node).await
        }
        async fn add_nodes_raw(&self, nodes: Vec<Value>) -> GraphDBResult<()> {
            self.inner.add_nodes_raw(nodes).await
        }
        async fn delete_node(&self, id: &str) -> GraphDBResult<()> {
            self.inner.delete_node(id).await
        }
        async fn delete_nodes(&self, ids: &[String]) -> GraphDBResult<()> {
            self.inner.delete_nodes(ids).await
        }
        async fn get_node(&self, id: &str) -> GraphDBResult<Option<NodeData>> {
            self.inner.get_node(id).await
        }
        async fn get_nodes(&self, ids: &[String]) -> GraphDBResult<Vec<NodeData>> {
            self.inner.get_nodes(ids).await
        }
        async fn has_edge(&self, s: &str, t: &str, r: &str) -> GraphDBResult<bool> {
            self.inner.has_edge(s, t, r).await
        }
        async fn has_edges(&self, edges: &[EdgeData]) -> GraphDBResult<Vec<EdgeData>> {
            self.inner.has_edges(edges).await
        }
        async fn add_edge(
            &self,
            s: &str,
            t: &str,
            r: &str,
            p: Option<HashMap<Cow<'static, str>, Value>>,
        ) -> GraphDBResult<()> {
            self.inner.add_edge(s, t, r, p).await
        }
        async fn add_edges(&self, edges: &[EdgeData]) -> GraphDBResult<()> {
            self.inner.add_edges(edges).await
        }
        async fn get_edges(&self, id: &str) -> GraphDBResult<Vec<EdgeData>> {
            self.inner.get_edges(id).await
        }
        async fn get_neighbors(&self, id: &str) -> GraphDBResult<Vec<NodeData>> {
            self.inner.get_neighbors(id).await
        }
        async fn get_connections(
            &self,
            id: &str,
        ) -> GraphDBResult<Vec<(NodeData, HashMap<Cow<'static, str>, Value>, NodeData)>> {
            self.inner.get_connections(id).await
        }
        async fn get_graph_data(&self) -> GraphDBResult<(Vec<GraphNode>, Vec<EdgeData>)> {
            self.inner.get_graph_data().await
        }
        async fn get_graph_metrics(
            &self,
            include_optional: bool,
        ) -> GraphDBResult<HashMap<Cow<'static, str>, Value>> {
            self.inner.get_graph_metrics(include_optional).await
        }
        async fn get_filtered_graph_data(
            &self,
            filters: &HashMap<Cow<'static, str>, Vec<Value>>,
        ) -> GraphDBResult<(Vec<GraphNode>, Vec<EdgeData>)> {
            self.inner.get_filtered_graph_data(filters).await
        }
        async fn get_nodeset_subgraph(
            &self,
            node_type: &str,
            node_names: &[String],
            op: &str,
        ) -> GraphDBResult<(Vec<GraphNode>, Vec<EdgeData>)> {
            self.inner
                .get_nodeset_subgraph(node_type, node_names, op)
                .await
        }
    }

    #[test]
    fn entity_name_falls_back_from_name_to_text_to_id() {
        let named = entity_from_result(&entity_hit(json!({"id": "entity-1", "name": "Named"})));
        assert_eq!(named.name, "Named");

        let texted = entity_from_result(&entity_hit(json!({"id": "entity-1", "text": "Texted"})));
        assert_eq!(texted.name, "Texted");

        let id_only = entity_from_result(&entity_hit(json!({"id": "entity-1"})));
        assert_eq!(id_only.name, "entity-1");
    }

    #[test]
    fn entity_type_prefers_is_a_and_suppresses_index_schema() {
        let payload = json!({"id": "e", "type": "IndexSchema", "is_a": "Office"});
        assert_eq!(entity_type(&payload), Some("Office".to_string()));

        let only_index = json!({"id": "e", "type": "IndexSchema"});
        assert_eq!(entity_type(&only_index), None);

        let domain = json!({"id": "e", "type": "Office"});
        assert_eq!(entity_type(&domain), Some("Office".to_string()));
    }

    #[tokio::test]
    async fn edge_text_explicit_and_synthesized_fallback() {
        let graph = MockGraphDB::new();
        add_node(&graph, "entity-1", "Source").await;
        add_node(&graph, "target-1", "Target").await;
        add_edge(&graph, "entity-1", "target-1", "REL", Some("Edge text")).await;
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Entity"}))];
        let entities = build_entities(&graph, &hits, 5, &HashMap::new()).await;
        assert_eq!(entities[0].edges[0].text, "Edge text");

        let graph2 = MockGraphDB::new();
        add_node(&graph2, "entity-1", "Source").await;
        add_node(&graph2, "target-1", "Target").await;
        add_edge(&graph2, "entity-1", "target-1", "REL", None).await;
        let entities = build_entities(&graph2, &hits, 5, &HashMap::new()).await;
        assert_eq!(entities[0].edges[0].text, "Source -- REL -- Target");
    }

    #[tokio::test]
    async fn duplicate_edges_removed_and_max_edges_caps_results() {
        // Blank relationship_name -> text-only dedupe track. Two "same" collapse,
        // "other" survives; cap of 1 keeps a single bullet.
        let graph = MockGraphDB::new();
        add_node(&graph, "entity-1", "Entity").await;
        add_node(&graph, "t1", "T1").await;
        add_node(&graph, "t2", "T2").await;
        add_node(&graph, "t3", "T3").await;
        add_edge(&graph, "entity-1", "t1", "", Some("same")).await;
        add_edge(&graph, "entity-1", "t2", "", Some("same")).await;
        add_edge(&graph, "entity-1", "t3", "", Some("other")).await;
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Entity"}))];
        let entities = build_entities(&graph, &hits, 1, &HashMap::new()).await;
        let texts: Vec<&str> = entities[0].edges.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(texts, ["same"]);
    }

    #[tokio::test]
    async fn same_edge_text_does_not_collapse_distinct_relationships() {
        // Distinct target_ids -> distinct dedupe keys -> both survive; the literal
        // duplicate triple (entity-1, REL, t1) collapses.
        let graph = MockGraphDB::new();
        add_node(&graph, "entity-1", "Entity").await;
        add_node(&graph, "t1", "T1").await;
        add_node(&graph, "t2", "T2").await;
        add_edge(&graph, "entity-1", "t1", "REL", Some("related")).await;
        add_edge(&graph, "entity-1", "t2", "REL", Some("related")).await;
        add_edge(&graph, "entity-1", "t1", "REL", Some("related")).await;
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Entity"}))];
        let entities = build_entities(&graph, &hits, 5, &HashMap::new()).await;
        let target_ids: Vec<&str> = entities[0]
            .edges
            .iter()
            .filter_map(|e| e.target_id.as_deref())
            .collect();
        assert_eq!(target_ids, ["t1", "t2"]);
    }

    #[tokio::test]
    async fn is_a_edge_is_prioritized_before_edge_cap() {
        // Unranked is_a edge beats a ranked-looking owns edge for the single slot.
        let graph = MockGraphDB::new();
        add_node(&graph, "entity-1", "Lisbon office").await;
        add_node(&graph, "project-1", "HarborLens").await;
        add_node(&graph, "type-1", "Office").await;
        add_edge(
            &graph,
            "entity-1",
            "project-1",
            "owns",
            Some("Lisbon office owns HarborLens"),
        )
        .await;
        add_edge(
            &graph,
            "entity-1",
            "type-1",
            "is_a",
            Some("Lisbon office is a Office"),
        )
        .await;
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Entity"}))];
        let entities = build_entities(&graph, &hits, 1, &HashMap::new()).await;
        let texts: Vec<&str> = entities[0].edges.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(texts, ["Lisbon office is a Office"]);
    }

    #[tokio::test]
    async fn edge_hits_rank_entity_bullets_and_fill_facts_section() {
        let ranked_bullet = "Alice works at Acme.";
        let unranked_bullet = "Alice plays tennis.";
        let fact = "Acme acquired Initech.";
        let graph = MockGraphDB::new();
        add_node(&graph, "entity-1", "Alice").await;
        add_node(&graph, "tennis-id", "Tennis").await;
        add_node(&graph, "acme-id", "Acme").await;
        add_node(&graph, "person-id", "Person").await;
        add_edge(
            &graph,
            "entity-1",
            "tennis-id",
            "plays",
            Some(unranked_bullet),
        )
        .await;
        add_edge(
            &graph,
            "entity-1",
            "acme-id",
            "works_at",
            Some(ranked_bullet),
        )
        .await;
        add_edge(&graph, "entity-1", "person-id", "is_a", None).await;

        let edge_hits = vec![
            edge_hit(fact),
            edge_hit(ranked_bullet),
            edge_hit("works at"),
        ];
        let edge_ranks = edge_rank_by_id(&edge_hits);
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Alice"}))];
        let entities = build_entities(&graph, &hits, 5, &edge_ranks).await;

        let bullets: Vec<&str> = entities[0].edges.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(
            bullets,
            ["Alice -- is_a -- Person", ranked_bullet, unranked_bullet]
        );
    }

    #[tokio::test]
    async fn edge_between_two_retrieved_entities_appears_under_both() {
        let graph = MockGraphDB::new();
        add_node(&graph, "alice-id", "Alice").await;
        add_node(&graph, "acme-id", "Acme").await;
        add_edge(
            &graph,
            "alice-id",
            "acme-id",
            "works_at",
            Some("Alice works at Acme."),
        )
        .await;
        let hits = vec![
            entity_hit(json!({"id": "alice-id", "name": "Alice"})),
            entity_hit(json!({"id": "acme-id", "name": "Acme"})),
        ];
        let entities = build_entities(&graph, &hits, 5, &HashMap::new()).await;
        let bullet_texts: Vec<Vec<&str>> = entities
            .iter()
            .map(|e| e.edges.iter().map(|b| b.text.as_str()).collect())
            .collect();
        assert_eq!(
            bullet_texts,
            vec![vec!["Alice works at Acme."], vec!["Alice works at Acme."]]
        );
    }

    #[tokio::test]
    async fn malformed_row_skipped_without_dropping_entity() {
        // An edge whose target node is absent from the neighborhood: the entity
        // survives and the bullet uses the target id as its label.
        let graph = MockGraphDB::new();
        add_node(&graph, "entity-1", "Source").await;
        add_edge(&graph, "entity-1", "target-1", "REL", None).await;
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Entity"}))];
        let entities = build_entities(&graph, &hits, 5, &HashMap::new()).await;
        assert_eq!(entities[0].name, "Entity");
        assert_eq!(entities[0].edges[0].text, "Source -- REL -- target-1");
    }

    #[tokio::test]
    async fn build_entities_fails_open_on_neighborhood_error() {
        let graph = FailingGraphDB::new();
        let hits = vec![entity_hit(json!({"id": "entity-1", "name": "Alice"}))];
        let entities = build_entities(&graph, &hits, 5, &HashMap::new()).await;
        assert_eq!(
            entities,
            vec![EntityResult {
                id: "entity-1".to_string(),
                name: "Alice".to_string(),
                description: None,
                entity_type: None,
                edges: vec![],
            }]
        );
    }

    #[tokio::test]
    async fn build_entities_returns_empty_for_no_hits() {
        let graph = MockGraphDB::new();
        let entities = build_entities(&graph, &[], 5, &HashMap::new()).await;
        assert!(entities.is_empty());
    }

    #[test]
    fn format_entities_renders_minimal_blocks() {
        // No optional fields -> header only.
        let minimal = EntityResult {
            id: "entity-1".to_string(),
            name: "Entity".to_string(),
            description: None,
            entity_type: None,
            edges: vec![],
        };
        assert_eq!(
            format_entities(&[minimal]),
            "## Relevant entities\n### Entity"
        );

        // IndexSchema type is suppressed from the header.
        let index_schema = EntityResult {
            id: "entity-1".to_string(),
            name: "lisbon office logistics intelligence project".to_string(),
            description: None,
            entity_type: Some("IndexSchema".to_string()),
            edges: vec![],
        };
        assert_eq!(
            format_entities(&[index_schema]),
            "## Relevant entities\n### lisbon office logistics intelligence project"
        );

        // Domain type + a description + an edge bullet.
        let full = EntityResult {
            id: "entity-1".to_string(),
            name: "Alice".to_string(),
            description: Some("An engineer.".to_string()),
            entity_type: Some("Person".to_string()),
            edges: vec![EdgeBullet {
                text: "Alice works at Acme.".to_string(),
                source: Some("Alice".to_string()),
                target: Some("Acme".to_string()),
                source_id: Some("alice-id".to_string()),
                relationship: Some("works_at".to_string()),
                target_id: Some("acme-id".to_string()),
                edge_type_id: None,
            }],
        };
        assert_eq!(
            format_entities(&[full]),
            "## Relevant entities\n### Alice (Person)\nAn engineer.\n- Alice works at Acme."
        );

        // A blank-name entity yields no block, so the whole section is empty.
        let blank = EntityResult {
            id: String::new(),
            name: "   ".to_string(),
            description: None,
            entity_type: None,
            edges: vec![],
        };
        assert_eq!(format_entities(&[blank]), "");
    }
}
