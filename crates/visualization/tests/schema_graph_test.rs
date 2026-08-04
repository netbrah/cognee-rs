#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Parity tests for the visualization preprocessor's schema graph.
//!
//! Mirrors Python's
//! `cognee/tests/unit/modules/visualization/test_preprocessor.py:340-551`
//! (the type-graph, rollup and operation-layer cases). The Python tests drive
//! `preprocess()` end to end; these build the *already-normalized* node/link
//! objects that `extract_schema_graph_data` consumes, so the assertions pin the
//! schema-graph half on its own.

use std::collections::HashMap;

use cognee_visualization::preprocessor::operations_catalog::OPERATIONS;
use cognee_visualization::preprocessor::schema_graph::{
    OTHER_ENTITY_TYPES_LABEL, SCHEMA_MAX_ENTITY_TYPES, build_operation_layer,
    extract_schema_graph_data, node_type_rank, resolve_semantic_types,
};
use serde_json::{Value, json};

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Normalized node object. `degree` is what the real preprocessor stamps in
/// pass 2; the sample ordering reads it.
fn node(id: &str, node_type: &str, name: &str, degree: u64) -> Value {
    json!({ "id": id, "type": node_type, "name": name, "degree": degree })
}

/// Normalized link object, matching the shape `preprocess()` emits.
fn link(source: &str, target: &str, relation: &str) -> Value {
    json!({
        "source": source,
        "target": target,
        "relation": relation,
        "weight": Value::Null,
        "all_weights": {},
        "relationship_type": Value::Null,
        "edge_info": {},
        "edge_class": if relation == "knows" { "semantic" } else { "structural" },
    })
}

/// The canonical Alice example, already normalized: one document, two chunks,
/// three entities of two semantic types, two EntityTypes and one summary.
/// Mirrors `_alice_like_graph()` (`test_preprocessor.py:28-105`) after
/// `preprocess()` has run over it.
fn alice_like_graph() -> (Vec<Value>, Vec<Value>) {
    let chunk = |id: &str, name: &str, degree: u64| {
        json!({
            "id": id,
            "type": "DocumentChunk",
            "name": name,
            "degree": degree,
            "source_pipeline": "cognify_pipeline",
            "source_task": "extract_chunks_from_documents",
            "topological_rank": 2,
        })
    };
    let entity = |id: &str, name: &str, degree: u64| {
        json!({
            "id": id,
            "type": "Entity",
            "name": name,
            "degree": degree,
            "source_pipeline": "cognify_pipeline",
            "source_task": "extract_graph_from_data",
            "topological_rank": 3,
        })
    };

    let nodes = vec![
        node("doc1", "TextDocument", "alice.md", 2),
        chunk("c1", "Alice knows Bob.", 4),
        chunk("c2", "NLP is a subfield of CS.", 2),
        entity("alice", "Alice", 3),
        entity("bob", "Bob", 3),
        entity("nlp", "NLP", 2),
        node("person", "EntityType", "Person", 2),
        node("field", "EntityType", "Field", 1),
        node("sum1", "TextSummary", "Alice and Bob in NLP.", 1),
    ];
    let links = vec![
        link("doc1", "c1", "contains"),
        link("doc1", "c2", "contains"),
        link("c1", "alice", "contains"),
        link("c1", "bob", "contains"),
        link("c2", "nlp", "contains"),
        link("alice", "person", "is_a"),
        link("bob", "person", "is_a"),
        link("nlp", "field", "is_a"),
        link("alice", "bob", "knows"),
        link("c1", "sum1", "made_from"),
    ];
    (nodes, links)
}

/// `num_types` semantic entity types with strictly descending instance counts:
/// `Type00` has `num_types` instances, `Type01` has `num_types - 1`, … down to 1.
/// Mirrors `_many_entity_types_graph` (`test_preprocessor.py:396-409`).
fn many_entity_types_graph(num_types: usize) -> (Vec<Value>, Vec<Value>) {
    let mut nodes = Vec::new();
    let mut links = Vec::new();
    for index in 0..num_types {
        let type_name = format!("Type{index:02}");
        let type_id = format!("etype{index}");
        nodes.push(node(
            &type_id,
            "EntityType",
            &type_name,
            (num_types - index) as u64,
        ));
        for instance in 0..(num_types - index) {
            let entity_id = format!("e{index}_{instance}");
            nodes.push(node(
                &entity_id,
                "Entity",
                &format!("{type_name}_inst{instance}"),
                1,
            ));
            links.push(link(&entity_id, &type_id, "is_a"));
        }
    }
    (nodes, links)
}

/// Index the `GraphNodeType` cards of a schema graph by their display name.
fn type_cards(schema: &Value) -> HashMap<String, Value> {
    schema["nodes"]
        .as_array()
        .expect("schema nodes is an array")
        .iter()
        .filter(|card| card["type"] == "GraphNodeType")
        .map(|card| {
            (
                card["name"]
                    .as_str()
                    .expect("card name is a string")
                    .to_string(),
                card.clone(),
            )
        })
        .collect()
}

/// All `GraphRelationshipType` nodes of a schema graph.
fn relationship_nodes(schema: &Value) -> Vec<Value> {
    schema["nodes"]
        .as_array()
        .expect("schema nodes is an array")
        .iter()
        .filter(|card| card["type"] == "GraphRelationshipType")
        .cloned()
        .collect()
}

/// `(relation, to_type) -> count` for a card's relationship distribution.
fn relationship_map(card: &Value) -> HashMap<(String, String), u64> {
    card["relationships"]
        .as_array()
        .expect("relationships is an array")
        .iter()
        .map(|entry| {
            (
                (
                    entry["relation"]
                        .as_str()
                        .expect("relation is a string")
                        .to_string(),
                    entry["to_type"]
                        .as_str()
                        .expect("to_type is a string")
                        .to_string(),
                ),
                entry["count"].as_u64().expect("count is a number"),
            )
        })
        .collect()
}

// ── node_type_rank ───────────────────────────────────────────────────────────

#[test]
fn node_type_rank_places_actors_left_of_the_document_pipeline() {
    assert_eq!(node_type_rank(Some("Tenant")), -5);
    assert_eq!(node_type_rank(Some("User")), -4);
    assert_eq!(node_type_rank(Some("Agent")), -3);
    assert_eq!(node_type_rank(Some("Session")), -2);
    assert_eq!(node_type_rank(Some("Dataset")), -1);
}

#[test]
fn node_type_rank_follows_the_document_pipeline_and_schema_taxonomy() {
    assert_eq!(node_type_rank(Some("TextDocument")), 0);
    assert_eq!(node_type_rank(Some("DocumentChunk")), 1);
    assert_eq!(node_type_rank(Some("Entity")), 2);
    assert_eq!(node_type_rank(Some("EntityType")), 3);
    assert_eq!(node_type_rank(Some("TextSummary")), 4);
    assert_eq!(node_type_rank(Some("GlobalContextSummary")), 5);

    assert_eq!(node_type_rank(Some("DatabaseSchema")), 0);
    assert_eq!(node_type_rank(Some("SchemaTable")), 1);
    assert_eq!(node_type_rank(Some("SchemaRelationship")), 2);
    assert_eq!(node_type_rank(Some("TableType")), 1);
    assert_eq!(node_type_rank(Some("TableRow")), 2);
    assert_eq!(node_type_rank(Some("ColumnValue")), 3);
}

#[test]
fn node_type_rank_defaults_unknown_types_to_four() {
    assert_eq!(node_type_rank(Some("Person")), 4);
    assert_eq!(node_type_rank(Some("NodeSet")), 4);
    assert_eq!(node_type_rank(Some("")), 4);
    assert_eq!(node_type_rank(None), 4);
}

// ── Type graph: semantic types, samples, relationship distribution ───────────

#[test]
fn schema_type_nodes_resolve_semantic_types_via_is_a() {
    let (nodes, links) = alice_like_graph();
    let cards = type_cards(&extract_schema_graph_data(&nodes, &links));

    assert!(
        !cards.contains_key("Entity"),
        "the literal Entity type never surfaces"
    );
    assert!(cards.contains_key("Person"));
    assert!(cards.contains_key("Field"));
    assert_eq!(cards["Person"]["instance_count"], 2);
    assert_eq!(cards["Field"]["instance_count"], 1);
    // EntityType is surfaced as its own card (_INTERNAL_TYPES is empty).
    assert_eq!(cards["EntityType"]["instance_count"], 2);

    // Semantic entity types share the Entity column; EntityType keeps its own.
    assert_eq!(cards["Person"]["rank"], node_type_rank(Some("Entity")));
    assert_eq!(cards["Field"]["rank"], node_type_rank(Some("Entity")));
    assert_eq!(
        cards["EntityType"]["rank"],
        node_type_rank(Some("EntityType"))
    );

    let resolved = resolve_semantic_types(&nodes, &links);
    assert_eq!(resolved["alice"], "Person");
    assert_eq!(resolved["nlp"], "Field");
    assert_eq!(resolved["field"], "EntityType");
    assert_eq!(resolved["doc1"], "TextDocument");
}

#[test]
fn schema_type_nodes_carry_bounded_deterministic_samples() {
    let (nodes, links) = alice_like_graph();
    let cards = type_cards(&extract_schema_graph_data(&nodes, &links));

    // Alice and Bob both have degree 3; the tie breaks to name order.
    assert_eq!(cards["Person"]["samples"], json!(["Alice", "Bob"]));
    assert_eq!(cards["Person"]["sample_size"], 2);
    assert_eq!(cards["Field"]["samples"], json!(["NLP"]));

    for card in cards.values() {
        let samples = card["samples"].as_array().expect("samples is an array");
        let sample_size = card["sample_size"]
            .as_u64()
            .expect("sample_size is a number");
        assert!(sample_size <= 5, "samples never exceed the per-type cap");
        assert_eq!(samples.len() as u64, sample_size);
    }
}

#[test]
fn schema_type_nodes_carry_full_relationship_distribution() {
    let (nodes, links) = alice_like_graph();
    let cards = type_cards(&extract_schema_graph_data(&nodes, &links));

    let person = relationship_map(&cards["Person"]);
    // Alice + Bob both is_a the Person EntityType node; alice knows bob.
    assert_eq!(person[&("is_a".into(), "EntityType".into())], 2);
    assert_eq!(person[&("knows".into(), "Person".into())], 1);
    // Incoming edges are recorded with the U+2190 arrow + space prefix, so a
    // type whose connections are all inbound is never drawn isolated.
    assert_eq!(
        person[&("\u{2190} contains".into(), "DocumentChunk".into())],
        2
    );
    assert_eq!(person[&("\u{2190} knows".into(), "Person".into())], 1);

    // DocumentChunk contains Person twice (alice, bob), Field once (nlp).
    let chunk = relationship_map(&cards["DocumentChunk"]);
    assert_eq!(chunk[&("contains".into(), "Person".into())], 2);
    assert_eq!(chunk[&("contains".into(), "Field".into())], 1);
    assert_eq!(
        chunk[&("\u{2190} contains".into(), "TextDocument".into())],
        2
    );

    // The EntityType card is reached only by inbound is_a edges.
    let entity_type = relationship_map(&cards["EntityType"]);
    assert_eq!(entity_type[&("\u{2190} is_a".into(), "Person".into())], 2);
    assert_eq!(entity_type[&("\u{2190} is_a".into(), "Field".into())], 1);
    assert_eq!(entity_type.len(), 2);

    // Distribution entries sort by (-count, to_type, relation).
    let counts: Vec<u64> = cards["DocumentChunk"]["relationships"]
        .as_array()
        .expect("relationships is an array")
        .iter()
        .map(|entry| entry["count"].as_u64().expect("count is a number"))
        .collect();
    let mut sorted = counts.clone();
    sorted.sort_by(|left, right| right.cmp(left));
    assert_eq!(counts, sorted);
}

// ── Relationship-type nodes ──────────────────────────────────────────────────

#[test]
fn relationship_nodes_carry_exactly_three_fields() {
    let (nodes, links) = alice_like_graph();
    let schema = extract_schema_graph_data(&nodes, &links);
    let rel_nodes = relationship_nodes(&schema);
    assert!(!rel_nodes.is_empty());

    let chunk_to_person = rel_nodes
        .iter()
        .find(|rel| rel["source_type"] == "DocumentChunk" && rel["target_type"] == "Person")
        .expect("DocumentChunk -> Person pair exists");
    assert_eq!(chunk_to_person["name"], "DocumentChunk to Person");
    assert_eq!(chunk_to_person["edge_count"], 2);
    assert_eq!(chunk_to_person["relationship_label"], "contains (2)");
    assert_eq!(
        chunk_to_person["fields"],
        json!([
            { "name": "edges", "type": "2", "required": true },
            { "name": "top relation", "type": "contains", "required": true },
            { "name": "relation types", "type": "1", "required": true },
        ])
    );
    // Rank sits between the two type columns.
    assert_eq!(chunk_to_person["rank"], json!(1.5));

    // Every rel node contributes exactly one "from" and one "to" link.
    let schema_links = schema["links"].as_array().expect("links is an array");
    let rel_id = chunk_to_person["id"].as_str().expect("rel id is a string");
    assert!(schema_links.iter().any(|edge| {
        edge["source"] == "type:DocumentChunk"
            && edge["target"] == rel_id
            && edge["label"] == "from"
    }));
    assert!(schema_links.iter().any(|edge| {
        edge["source"] == rel_id && edge["target"] == "type:Person" && edge["label"] == "to"
    }));
    assert_eq!(schema_links.len(), rel_nodes.len() * 2);
}

#[test]
fn self_link_relationship_node_sits_half_a_column_right() {
    let (nodes, links) = alice_like_graph();
    let schema = extract_schema_graph_data(&nodes, &links);
    let cards = type_cards(&schema);
    let person_rank = cards["Person"]["rank"].as_f64().expect("rank is a number");

    let self_link = relationship_nodes(&schema)
        .into_iter()
        .find(|rel| rel["source_type"] == "Person" && rel["target_type"] == "Person")
        .expect("alice knows bob is a Person self-link");
    assert_eq!(self_link["name"], "Person self-links");
    assert_eq!(self_link["rank"].as_f64(), Some(person_rank + 0.5));
    assert_eq!(self_link["edge_count"], 1);
}

// ── Instance drill-down ──────────────────────────────────────────────────────

#[test]
fn instance_index_records_adjacency_in_link_order() {
    let (nodes, links) = alice_like_graph();
    let schema = extract_schema_graph_data(&nodes, &links);

    let instances = &schema["instances_by_type"];
    let people: Vec<&str> = instances["Person"]
        .as_array()
        .expect("Person instances is an array")
        .iter()
        .map(|record| record["name"].as_str().expect("instance name is a string"))
        .collect();
    assert_eq!(people, vec!["Alice", "Bob"], "instances sort by name");

    let alice = &schema["instance_index"]["alice"];
    assert_eq!(alice["type"], "Person");
    assert_eq!(alice["name"], "Alice");
    assert_eq!(
        alice["out"],
        json!([
            { "relation": "is_a", "id": "person" },
            { "relation": "knows", "id": "bob" },
        ])
    );
    assert_eq!(alice["in"], json!([{ "relation": "contains", "id": "c1" }]));
}

// ── Entity long-tail rollup ──────────────────────────────────────────────────

#[test]
fn entity_type_long_tail_rolls_up_into_other_entities() {
    let num_types = SCHEMA_MAX_ENTITY_TYPES + 3;
    let (nodes, links) = many_entity_types_graph(num_types);
    let schema = extract_schema_graph_data(&nodes, &links);
    let cards = type_cards(&schema);

    let semantic_cards = cards
        .keys()
        .filter(|name| name.starts_with("Type") || *name == OTHER_ENTITY_TYPES_LABEL)
        .count();
    assert_eq!(semantic_cards, SCHEMA_MAX_ENTITY_TYPES);

    // Top types keep their own cards; the smallest types are rolled up.
    assert!(cards.contains_key("Type00"));
    assert!(!cards.contains_key(&format!("Type{:02}", num_types - 1)));

    let rollup = &cards[OTHER_ENTITY_TYPES_LABEL];
    assert_eq!(rollup["rollup"], true);
    let tail_size = num_types - (SCHEMA_MAX_ENTITY_TYPES - 1);
    let rolled_up_types = rollup["rolled_up_types"]
        .as_array()
        .expect("rolled_up_types is an array");
    assert_eq!(rolled_up_types.len(), tail_size);
    // Tail of descending counts ends at 1: tail_size + (tail_size-1) + … + 1.
    assert_eq!(
        rollup["instance_count"],
        (tail_size * (tail_size + 1) / 2) as u64
    );
    // The rollup keeps the Entity column rank of the cards it replaces.
    assert_eq!(rollup["rank"], cards["Type00"]["rank"]);
    assert_eq!(rollup["rank"], node_type_rank(Some("Entity")));

    // The lead field (index 1, right after "count") announces the rollup.
    let fields = rollup["fields"].as_array().expect("fields is an array");
    assert_eq!(fields[0]["name"], "count");
    assert_eq!(fields[1]["name"], "entity types");
    assert_eq!(fields[1]["required"], true);
    let lead = fields[1]["type"].as_str().expect("field type is a string");
    assert!(
        lead.starts_with(&format!("{tail_size} rolled up: ")),
        "unexpected lead field: {lead}"
    );
    assert!(lead.ends_with(", …"), "unexpected lead field: {lead}");
    // The three largest tail types are named, largest first.
    assert!(lead.contains("Type11 (4)"), "unexpected lead field: {lead}");
    assert!(lead.contains("Type12 (3)"), "unexpected lead field: {lead}");
    assert!(lead.contains("Type13 (2)"), "unexpected lead field: {lead}");

    // rolled_up_types is ordered by descending count.
    let tail_counts: Vec<u64> = rolled_up_types
        .iter()
        .map(|entry| entry["count"].as_u64().expect("count is a number"))
        .collect();
    assert_eq!(tail_counts, vec![4, 3, 2, 1]);

    // Pair-relationship nodes never reference a rolled-up type name.
    let rolled_names: Vec<&str> = rolled_up_types
        .iter()
        .map(|entry| entry["name"].as_str().expect("name is a string"))
        .collect();
    for rel in relationship_nodes(&schema) {
        let source = rel["source_type"]
            .as_str()
            .expect("source_type is a string");
        let target = rel["target_type"]
            .as_str()
            .expect("target_type is a string");
        assert!(
            !rolled_names.contains(&source),
            "{source} should have been rolled up"
        );
        assert!(
            !rolled_names.contains(&target),
            "{target} should have been rolled up"
        );
    }
    // …and the rollup itself is wired up as an ordinary type.
    assert!(relationship_nodes(&schema).iter().any(|rel| {
        rel["source_type"] == OTHER_ENTITY_TYPES_LABEL && rel["target_type"] == "EntityType"
    }));

    // Instance drill-down still reaches the rolled-up instances.
    let instances = schema["instances_by_type"][OTHER_ENTITY_TYPES_LABEL]
        .as_array()
        .expect("rollup instances is an array");
    assert_eq!(
        instances.len() as u64,
        (tail_size * (tail_size + 1) / 2) as u64
    );
}

#[test]
fn entity_types_under_the_cap_are_not_rolled_up() {
    let (nodes, links) = alice_like_graph();
    let cards = type_cards(&extract_schema_graph_data(&nodes, &links));
    assert!(!cards.contains_key(OTHER_ENTITY_TYPES_LABEL));

    let (nodes, links) = many_entity_types_graph(SCHEMA_MAX_ENTITY_TYPES);
    let cards = type_cards(&extract_schema_graph_data(&nodes, &links));
    assert!(!cards.contains_key(OTHER_ENTITY_TYPES_LABEL));
    assert_eq!(
        cards.keys().filter(|name| name.starts_with("Type")).count(),
        SCHEMA_MAX_ENTITY_TYPES
    );
}

// ── DLT branch ───────────────────────────────────────────────────────────────

#[test]
fn schema_table_nodes_switch_to_the_dlt_branch() {
    let nodes = vec![
        json!({
            "id": "orders",
            "type": "SchemaTable",
            "name": "orders",
            "degree": 1,
            "description": "one row per order",
            "columns": [
                { "name": "id", "data_type": "bigint", "primary_key": true },
                { "name": "customer_id", "data_type": "bigint", "nullable": false },
            ],
        }),
        json!({
            "id": "orders_customers",
            "type": "SchemaRelationship",
            "name": "orders -> customers",
            "degree": 1,
            "source_table": "orders",
            "target_table": "customers",
            "relationship_type": "many_to_one",
        }),
        // A regular graph node coexisting with the schema nodes is ignored.
        node("alice", "Entity", "Alice", 1),
    ];
    let links = vec![
        link("orders", "orders_customers", "has_relationship"),
        link("orders", "alice", "contains"),
    ];
    let schema = extract_schema_graph_data(&nodes, &links);

    // The DLT branch emits nodes + links only.
    assert!(schema.get("instances_by_type").is_none());
    assert!(schema.get("instance_index").is_none());

    let schema_nodes = schema["nodes"].as_array().expect("nodes is an array");
    assert_eq!(schema_nodes.len(), 2);
    assert_eq!(schema_nodes[0]["name"], "orders");
    assert_eq!(schema_nodes[0]["description"], "one row per order");
    assert_eq!(
        schema_nodes[0]["fields"],
        json!([
            { "name": "id", "type": "bigint", "required": true },
            { "name": "customer_id", "type": "bigint", "required": true },
        ])
    );
    assert_eq!(schema_nodes[1]["source_table"], "orders");
    assert_eq!(schema_nodes[1]["target_table"], "customers");
    assert_eq!(schema_nodes[1]["relationship_type"], "many_to_one");

    // Only links whose endpoints are both schema nodes survive.
    assert_eq!(
        schema["links"],
        json!([{
            "source": "orders",
            "target": "orders_customers",
            "label": "has_relationship",
        }])
    );
}

// ── Operation impact layer ───────────────────────────────────────────────────

#[test]
fn operation_layer_maps_operations_to_present_types() {
    let nodes = vec![
        node("d1", "TextDocument", "a.txt", 0),
        node("p1", "Entity", "Carlos", 1),
        node("t_person", "EntityType", "Person", 1),
    ];
    let links = vec![link("p1", "t_person", "is_a")];
    let mut schema = extract_schema_graph_data(&nodes, &links);
    build_operation_layer(&mut schema, &nodes, &links);

    let operations = schema["operations"]
        .as_array()
        .expect("operations is an array");
    let operation_links = schema["operation_links"]
        .as_array()
        .expect("operation_links is an array");
    let op_ids: Vec<&str> = operations
        .iter()
        .map(|op| op["id"].as_str().expect("operation id is a string"))
        .collect();
    assert!(op_ids.contains(&"op:cognify"));

    let targets_for = |op_id: &str| -> Vec<(String, String)> {
        operation_links
            .iter()
            .filter(|entry| entry["source"] == op_id)
            .map(|entry| {
                (
                    entry["target"]
                        .as_str()
                        .expect("target is a string")
                        .to_string(),
                    entry["effect"]
                        .as_str()
                        .expect("effect is a string")
                        .to_string(),
                )
            })
            .collect()
    };

    let cognify = targets_for("op:cognify");
    assert!(cognify.contains(&("type:TextDocument".into(), "produces".into())));
    // "Entity" expands to the semantic entity types actually present.
    assert!(cognify.contains(&("type:Person".into(), "produces".into())));
    assert!(cognify.contains(&("type:EntityType".into(), "produces".into())));
    // Absent types (DocumentChunk, TextSummary) never appear.
    assert_eq!(cognify.len(), 3);

    // An operation whose only target is absent (Rule) is dropped entirely.
    assert!(!op_ids.contains(&"op:coding_rule_associations"));

    // Modify effects are surfaced, and carry their property.
    let feedback = targets_for("op:apply_feedback_weights");
    assert!(feedback.contains(&("type:Person".into(), "modifies".into())));
    assert!(feedback.contains(&("type:EntityType".into(), "modifies".into())));
    let feedback_property = operation_links
        .iter()
        .find(|entry| {
            entry["source"] == "op:apply_feedback_weights" && entry["target"] == "type:Person"
        })
        .expect("feedback weighting touches Person");
    assert_eq!(feedback_property["property"], "feedback_weight");
    assert_eq!(
        feedback_property["observed"], false,
        "no pipeline_name means never observed"
    );

    // Operation nodes carry the catalog metadata verbatim.
    let cognify_node = operations
        .iter()
        .find(|op| op["id"] == "op:cognify")
        .expect("cognify is emitted");
    assert_eq!(cognify_node["name"], "cognify");
    assert_eq!(cognify_node["type"], "GraphOperation");
    assert_eq!(cognify_node["op_kind"], "pipeline");
    assert_eq!(cognify_node["scope"], "subset");
    assert_eq!(
        cognify_node["summary"],
        "Extracts a knowledge graph from raw documents."
    );

    // Operation nodes appear in catalog declaration order.
    let catalog_order: Vec<&str> = OPERATIONS
        .iter()
        .map(|op| op.name)
        .filter(|name| op_ids.contains(&format!("op:{name}").as_str()))
        .collect();
    let emitted_order: Vec<&str> = op_ids
        .iter()
        .map(|id| id.trim_start_matches("op:"))
        .collect();
    assert_eq!(emitted_order, catalog_order);
}

#[test]
fn operation_layer_flags_observed_links_from_live_provenance() {
    let nodes = vec![
        json!({
            "id": "d1",
            "type": "TextDocument",
            "name": "a.txt",
            "degree": 0,
            "source_pipeline": "cognify_pipeline",
        }),
        json!({
            "id": "gc1",
            "type": "GlobalContextSummary",
            "name": "bucket",
            "degree": 0,
            "source_pipeline": "cognify_pipeline",
        }),
    ];
    let mut schema = extract_schema_graph_data(&nodes, &[]);
    build_operation_layer(&mut schema, &nodes, &[]);
    let operation_links = schema["operation_links"]
        .as_array()
        .expect("operation_links is an array");

    let observed = |op_id: &str, target: &str| -> bool {
        operation_links
            .iter()
            .find(|entry| entry["source"] == op_id && entry["target"] == target)
            .and_then(|entry| entry["observed"].as_bool())
            .expect("link exists")
    };
    // cognify_pipeline stamped on the node matches cognify's pipeline_name.
    assert!(observed("op:cognify", "type:TextDocument"));
    // global_context_index runs under memify_pipeline, so this is unobserved.
    assert!(!observed(
        "op:global_context_index",
        "type:GlobalContextSummary"
    ));
}

#[test]
fn operation_layer_is_always_present_even_when_empty() {
    let mut schema = extract_schema_graph_data(&[], &[]);
    build_operation_layer(&mut schema, &[], &[]);
    assert_eq!(schema["operations"], json!([]));
    assert_eq!(schema["operation_links"], json!([]));
    assert_eq!(schema["nodes"], json!([]));
    assert_eq!(schema["links"], json!([]));
    assert_eq!(schema["instances_by_type"], json!({}));
    assert_eq!(schema["instance_index"], json!({}));
}
