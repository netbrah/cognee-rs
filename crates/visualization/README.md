# Cognee Visualization

Interactive HTML knowledge-graph visualization for Cognee-Rust.

Reads all nodes and edges from any `GraphDBTrait` implementation, enriches them,
and renders a single HTML file. The frontend is **vendored verbatim** from the
Python `cognee` repo — this crate is the preprocessor plus the orchestrator that
stitches the vendored assets around the data.

Parity mapping:

| Rust | Python source |
| --- | --- |
| `src/preprocessor/` | `cognee/modules/visualization/preprocessor.py` |
| `src/html.rs` | `cognee/modules/visualization/cognee_network_visualization.py` |
| `src/colors.rs` | the `color_map` dict + `_generate_provenance_colors()` |
| `assets/template.html`, `assets/views/*.js` | vendored byte-for-byte, not ported |

## The four tabs

| Tab | Rendered by | State in Rust |
| --- | --- | --- |
| **Graph** | `views/story_view.js` | Full. Canvas renderer with Story / Flow / Force layouts, quadtree hit-testing, minimap, label budget, density layers, search and hover. The Flow layout keys off the preprocessor's `visual_rank` (the runtime-stamped `topological_rank` when present, otherwise a stage-order fallback). |
| **Schema** | `views/schema_view.js` + `views/inspector.js` | Full. Ontology diagram of node types in canonical pipeline order, labelled relationship curves, edge-type cards, and a side-panel inspector with type → instance drill-down. Fed by the preprocessor's derived type-schema graph, so it works without any caller-supplied schema. |
| **Memory** | `views/memory_map.js` | Renders. Deterministic column map: documents (with ordered chunk cells) → entity-type groups → summaries → global context, plus the run timeline the preprocessor gap-clusters from each node's `t_created`. Only the *search-retrieval* half of the timeline rail (and the recall overlay) is inert, because `__SEARCH_EVENTS__` is empty (see Deferred). |
| **Semantic** | `views/semantic_map.js` | Empty state only. See Deferred. |

## The 20 tokens

The vendored shell and JS chunks declare 20 `__TOKEN__` slots.
`src/html.rs::build_html` substitutes all of them, and **the order is
load-bearing**: the eight JS chunks go first, because ten of the twelve data
tokens live *inside* those chunks and would otherwise never be reached.

**1. JS chunks** (in this order):

| Token | Filled with |
| --- | --- |
| `__UI_CHROME_JS__` | `assets/views/ui_chrome.js` |
| `__SCHEMA_VIEW_JS__` | `assets/views/schema_view.js` |
| `__STORY_VIEW_JS__` | `assets/views/story_view.js` |
| `__PIPELINE_LAYOUT_JS__` | the **empty string** — upstream `layouts/pipeline_layout.py:16-17` is a deliberate stub whose `emit_js()` returns `""`. The slot is kept, not filled with invented content. |
| `__INSPECTOR_JS__` | `assets/views/inspector.js` |
| `__MEMORY_VIEW_JS__` | `assets/views/memory_map.js` |
| `__SEMANTIC_LAYOUT_JS__` | the one line `window._semanticPositions = __SEMANTIC_POSITIONS__;` (upstream `layouts/semantic_layout.py:200-206`) — which is why this chunk must land before the data tokens |
| `__SEMANTIC_VIEW_JS__` | `assets/views/semantic_map.js` |

**2. Data payloads** (after the chunks):

| Token | Filled with |
| --- | --- |
| `__NODES_DATA__` | `pre.nodes` |
| `__LINKS_DATA__` | `pre.links` |
| `__TASK_COLORS__` | `pre.color_maps.task` |
| `__PIPELINE_COLORS__` | `pre.color_maps.pipeline` |
| `__NODESET_COLORS__` | `pre.color_maps.node_set` |
| `__USER_COLORS__` | `pre.color_maps.user` |
| `__SCHEMA_DATA__` | the caller-supplied opaque passthrough, or the literal `null` |
| `__SCHEMA_GRAPH_DATA__` | `pre.schema_graph`, falling back to `{"nodes":[],"links":[]}` |
| `__MEMORY_DATA__` | `pre.memory_map`, falling back to `{}` |
| `__SEARCH_EVENTS__` | the literal `[]` (deferred) |
| `__SEMANTIC_POSITIONS__` | the literal `null` (deferred) |
| `__SEMANTIC_CLUSTERS__` | the literal `null` (deferred) |

Every data token is substituted **unconditionally, with a JSON fallback**. A
leaked `__TOKEN__` is a hard failure — it lands as literal garbage inside a
`<script>` block and breaks the page — so `tests/html_test.rs` scans the whole
output for any surviving `__[A-Z][A-Z0-9_]*__`.

All embedded JSON goes through `safe_json_embed`: `serde_json::to_string` then
`</` → `<\/`. That is the only XSS guard on the payload; without it a node whose
`name` contains `</script>` would close the script block early.

## Vendored assets: re-sync, never hand-edit

`assets/template.html` and `assets/views/*.js` are byte-for-byte copies of the
upstream Python files. **Do not hand-edit them.** Change the frontend upstream,
then re-copy the whole set and bump the recorded commit in
[`assets/README.md`](assets/README.md), which holds the source paths and the
resync script.

This rule exists because it was broken once already: Rust used to carry
`assets/graph_template.html`, a 1680-line hand-fork of Python's pre-refactor
monolith. Edited in place, it drifted silently and fell several features behind
(two tabs instead of four, no Memory or Semantic view, no layout switch, no
inspector, and a Schema tab that could never render). `tests/html_test.rs`
asserts each chunk's distinctive identifiers survived the copy, so a bad or
partial resync fails the test suite instead of shipping quietly.

Note that the vendored `template.html` fetches d3 v7 and the Inter webfont from
CDNs. That matches Python exactly, so it is not a parity gap — but the output is
not fully self-contained and degrades offline / on air-gapped edge devices.

## Deferred

- **Semantic tab** — `__SEMANTIC_POSITIONS__` and `__SEMANTIC_CLUSTERS__` are
  both the literal `null`, so `semantic_map.js` shows its friendly empty state.
  Filling them needs each node's stored embedding vector, but Rust's
  `cognee_vector::SearchResult` has no `vector` field so stored vectors cannot be
  read back at all; a projection (PCA/SVD) and k-means seeded to match NumPy's
  PCG64 would also be required. This is the same state Python reaches whenever
  its own best-effort `_semantic_payload` raises, so it is a legal, upstream
  test-passing configuration.
- **Session events** — `__SEARCH_EVENTS__` is the literal `[]`. There is no Rust
  counterpart to Python's `visualization/session_events.py`, so the Semantic
  recall overlay has nothing to display and the Memory timeline rail shows only
  its run events (`memory_map.js:464-471` concatenates the two sources). The run
  events themselves are *not* deferred: they are gap-clustered from each node's
  `t_created`, which the graph adapters surface as the `DataPoint`'s epoch-ms
  `created_at` (`crates/graph/tests/common/mod.rs::
  test_get_graph_data_surfaces_created_at` pins that contract).
- **Subgraph bounding** — Python's `subgraph_data.py` (bounded per-dataset
  subgraph extraction) is not ported; Rust always renders the whole graph
  returned by `get_graph_data()`.

## Usage

```rust
use cognee_graph::GraphDBTrait;
use cognee_visualization::visualize;
use std::path::Path;

async fn example(graph_db: &dyn GraphDBTrait) -> Result<(), Box<dyn std::error::Error>> {
    // Write the visualization to a caller-specified file.
    let path = visualize(graph_db, Some(Path::new("/tmp/graph.html"))).await?;

    // Or write to ~/graph_visualization.html (matches Python behavior).
    let path = visualize(graph_db, None).await?;

    println!("wrote {}", path.display());
    Ok(())
}
```

## API

- `visualize(graph_db, output_path) -> PathBuf` — render and write the HTML file.
  When `output_path` is `None`, writes to `~/graph_visualization.html`
  (`%USERPROFILE%` on Windows). Returns the path written.
- `render(graph_db) -> String` — render the HTML string without writing it
  (useful for streaming over HTTP or embedding in a larger page).
- `render_multi_user(pairs) -> String` — aggregate multiple
  `(user_label, Arc<dyn GraphDBTrait>)` pairs into one HTML document. Nodes are
  deduplicated by stringified id (first-write-wins) and tagged with a
  `source_user` attribute so the renderer can color-code by user; edges are
  deduplicated by `(source, target, relation)`. Mirrors Python's
  `aggregate_multi_user_graphs()`.
- `preprocess(nodes, edges, schema_data) -> PreprocessedGraph` — the renderer
  snapshot on its own, re-exported from the public `preprocessor` module (Python
  `preprocessor.preprocess`). Exposed so callers can inspect or post-process the
  enriched nodes/links/color-maps without going through the HTML shell; the
  parity tests drive it directly.

Errors are surfaced via `VisualizationError`.
