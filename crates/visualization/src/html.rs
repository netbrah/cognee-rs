//! HTML shell assembly: token substitution over the vendored frontend.
//!
//! Ports `cognee_network_visualization()` from
//! [`cognee/modules/visualization/cognee_network_visualization.py:122-169`](https://github.com/topoteretes/cognee/blob/main/cognee/modules/visualization/cognee_network_visualization.py#L122-L169).
//!
//! The HTML shell (`assets/template.html`) and the six view chunks
//! (`assets/views/*.js`) are vendored byte-for-byte from the Python repo — see
//! `assets/README.md` for the resync rule. This module is the only Rust code
//! that knows their filenames.
//!
//! # Substitution order is load-bearing
//!
//! Python replaces the **eight JS-chunk tokens first**, then the twelve data
//! tokens. That order is not cosmetic: `__SCHEMA_GRAPH_DATA__` and
//! `__SCHEMA_DATA__` live *inside* `views/schema_view.js`, `__MEMORY_DATA__` /
//! `__SEARCH_EVENTS__` inside `views/memory_map.js`, the six graph payload
//! tokens inside `views/story_view.js`, and `__SEMANTIC_POSITIONS__` inside the
//! one line emitted for `__SEMANTIC_LAYOUT_JS__`. Substituting data first would
//! leave every one of those tokens unresolved in the output.
//!
//! Every data token is substituted **unconditionally with a JSON fallback**
//! (`null`, `{}`, `[]`). A leaked `__TOKEN__` is a hard failure — it renders as
//! literal garbage inside a `<script>` block and breaks the page — which is why
//! Python does it this way (`cognee_network_visualization.py:156-167`) and why
//! `tests/html_test.rs` scans the whole output for surviving tokens.
//!
//! # Deferred payloads (three tokens are structurally `null`/`[]`)
//!
//! Three payloads are deliberately out of scope for this crate today. All three
//! are substituted with valid JSON so no token leaks, and the vendored JS
//! renders a friendly empty state for each — the same state Python reaches
//! whenever its own best-effort computation fails (Python wraps
//! `_semantic_payload` in a blanket `except Exception`,
//! `cognee_network_visualization.py:63-65`), so this is a legal,
//! upstream-test-passing configuration rather than a broken one.
//!
//! * `__SEMANTIC_POSITIONS__` → `null`. Filling it needs the stored embedding
//!   vector of every node plus a 2-D projection. Rust's
//!   `cognee_vector::SearchResult` has no `vector` field, so stored vectors
//!   cannot be read back out of the vector store at all, and the workspace has
//!   no SVD/PCA implementation.
//! * `__SEMANTIC_CLUSTERS__` → `null`. Needs the same embeddings plus k-means
//!   seeded by NumPy's PCG64 to match Python's cluster ids; neither the
//!   clustering nor a PCG64 generator exists in the workspace.
//! * `__SEARCH_EVENTS__` → `[]`. Needs the session/operation event log
//!   (Python's `visualization/session_events.py`) that drives the Memory tab's
//!   timeline rail and the Semantic tab's recall overlay.
//!
//! With the first two `null`, `views/semantic_map.js` shows its empty state and
//! the Semantic tab is inert; the Graph, Schema and Memory tabs are fully
//! functional.

use serde_json::json;

use crate::VisualizationError;
use crate::preprocessor::PreprocessedGraph;

/// The vendored HTML shell.
///
/// Byte-for-byte copy of `cognee/modules/visualization/template.html`. Carries
/// the eight JS-chunk token slots (`:488-501`) plus the four tab buttons.
pub const HTML_TEMPLATE: &str = include_str!("../assets/template.html");

/// Vendored `views/ui_chrome.js` — theme toggle + tab switching.
const UI_CHROME_JS: &str = include_str!("../assets/views/ui_chrome.js");
/// Vendored `views/schema_view.js` — the Schema tab's ontology diagram.
const SCHEMA_VIEW_JS: &str = include_str!("../assets/views/schema_view.js");
/// Vendored `views/story_view.js` — the main canvas graph renderer.
const STORY_VIEW_JS: &str = include_str!("../assets/views/story_view.js");
/// Vendored `views/inspector.js` — the schema type/instance side panel.
const INSPECTOR_JS: &str = include_str!("../assets/views/inspector.js");
/// Vendored `views/memory_map.js` — the Memory tab's deterministic column map.
const MEMORY_VIEW_JS: &str = include_str!("../assets/views/memory_map.js");
/// Vendored `views/semantic_map.js` — the Semantic tab's meaning-space scatter.
const SEMANTIC_VIEW_JS: &str = include_str!("../assets/views/semantic_map.js");

/// `layouts/pipeline_layout.emit_js()` returns `""` upstream
/// ([`pipeline_layout.py:16-17`](https://github.com/topoteretes/cognee/blob/main/cognee/modules/visualization/layouts/pipeline_layout.py#L16-L17)).
/// The slot is kept so a future L→R Story layout drops in without touching the
/// vendored template; the legacy `computeRankedLayout` inside `story_view.js`
/// provides the Flow layout in the meantime.
const PIPELINE_LAYOUT_JS: &str = "";

/// `layouts/semantic_layout.emit_js()` returns exactly this one line upstream
/// ([`semantic_layout.py:200-206`](https://github.com/topoteretes/cognee/blob/main/cognee/modules/visualization/layouts/semantic_layout.py#L200-L206)).
/// It embeds `__SEMANTIC_POSITIONS__`, which is why this chunk must be
/// substituted before the data tokens.
const SEMANTIC_LAYOUT_JS: &str = "window._semanticPositions = __SEMANTIC_POSITIONS__;";

/// Serialize a value to JSON and neutralize `</` so the result is safe to embed
/// inside a `<script>` element.
///
/// Ports `_safe_json_embed()` (`cognee_network_visualization.py:70-73`). This is
/// the only XSS guard on the embedded payload: without it a node whose `name`
/// contains `</script>` would terminate the script block early and let the
/// remainder be parsed as markup.
fn safe_json_embed<T: serde::Serialize>(value: &T) -> Result<String, VisualizationError> {
    let raw = serde_json::to_string(value)?;
    Ok(raw.replace("</", "<\\/"))
}

/// Assemble the final HTML from the vendored shell + the preprocessed graph.
///
/// Substitutes all 20 `__TOKEN__` placeholders — the eight JS chunks first, then
/// the twelve data payloads — mirroring
/// `cognee_network_visualization.py:136-168` step for step. See the module docs
/// for why the order matters and which three payloads are deferred.
pub(crate) fn build_html(pre: &PreprocessedGraph) -> Result<String, VisualizationError> {
    let mut html = HTML_TEMPLATE.to_string();

    // 1) JS chunks: ordered so the first script block (ui_chrome + schema) runs
    //    before the main story-view IIFE in the second block. Substituted first
    //    so the data tokens *inside* these chunks get resolved below.
    html = html.replace("__UI_CHROME_JS__", UI_CHROME_JS);
    html = html.replace("__SCHEMA_VIEW_JS__", SCHEMA_VIEW_JS);
    html = html.replace("__STORY_VIEW_JS__", STORY_VIEW_JS);
    html = html.replace("__PIPELINE_LAYOUT_JS__", PIPELINE_LAYOUT_JS);
    html = html.replace("__INSPECTOR_JS__", INSPECTOR_JS);
    html = html.replace("__MEMORY_VIEW_JS__", MEMORY_VIEW_JS);
    html = html.replace("__SEMANTIC_LAYOUT_JS__", SEMANTIC_LAYOUT_JS);
    html = html.replace("__SEMANTIC_VIEW_JS__", SEMANTIC_VIEW_JS);

    // 2) Data tokens: substituted last so JSON-embedded `__SCHEMA_GRAPH_DATA__`
    //    and friends inside the JS chunks above are reached.
    html = html.replace("__NODES_DATA__", &safe_json_embed(&pre.nodes)?);
    html = html.replace("__LINKS_DATA__", &safe_json_embed(&pre.links)?);
    html = html.replace("__TASK_COLORS__", &safe_json_embed(&pre.color_maps.task)?);
    html = html.replace(
        "__PIPELINE_COLORS__",
        &safe_json_embed(&pre.color_maps.pipeline)?,
    );
    html = html.replace(
        "__NODESET_COLORS__",
        &safe_json_embed(&pre.color_maps.node_set)?,
    );
    html = html.replace("__USER_COLORS__", &safe_json_embed(&pre.color_maps.user)?);

    // Caller-supplied opaque passthrough; the literal `null` when absent —
    // `schema_view.js:22` reads it as `const schemaData = null`.
    html = html.replace(
        "__SCHEMA_DATA__",
        &match pre.schema_data.as_ref() {
            Some(v) => safe_json_embed(v)?,
            None => "null".to_string(),
        },
    );

    // Preprocessor-derived type graph; `{"nodes": [], "links": []}` when the
    // preprocessor produced nothing usable (mirrors Python's
    // `pre.schema_graph or {"nodes": [], "links": []}` truthiness fallback).
    let schema_graph_is_empty = match &pre.schema_graph {
        serde_json::Value::Object(map) => map.is_empty(),
        _ => true,
    };
    html = html.replace(
        "__SCHEMA_GRAPH_DATA__",
        &if schema_graph_is_empty {
            safe_json_embed(&json!({"nodes": [], "links": []}))?
        } else {
            safe_json_embed(&pre.schema_graph)?
        },
    );

    // Unconditional JSON-fallback substitutions: a leaked `__MEMORY_DATA__` /
    // `__SEARCH_EVENTS__` token would break the page and fail the
    // no-placeholder assembly test.
    let memory_is_empty = match &pre.memory_map {
        serde_json::Value::Object(map) => map.is_empty(),
        _ => true,
    };
    html = html.replace(
        "__MEMORY_DATA__",
        &if memory_is_empty {
            safe_json_embed(&json!({}))?
        } else {
            safe_json_embed(&pre.memory_map)?
        },
    );
    // Deferred: no session-event source in Rust yet (see module docs).
    html = html.replace("__SEARCH_EVENTS__", "[]");

    // Deferred: no readable stored embeddings, hence no projection or
    // clustering. `null` is the same state Python reaches when its best-effort
    // `_semantic_payload` raises; `semantic_map.js` renders an empty state.
    html = html.replace("__SEMANTIC_POSITIONS__", "null");
    html = html.replace("__SEMANTIC_CLUSTERS__", "null");

    Ok(html)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;

    /// Every token slot declared by the vendored template must be present
    /// before substitution — a missing one means a bad or partial asset resync.
    #[test]
    fn template_declares_all_js_chunk_slots() {
        for p in [
            "__UI_CHROME_JS__",
            "__SCHEMA_VIEW_JS__",
            "__STORY_VIEW_JS__",
            "__PIPELINE_LAYOUT_JS__",
            "__INSPECTOR_JS__",
            "__MEMORY_VIEW_JS__",
            "__SEMANTIC_LAYOUT_JS__",
            "__SEMANTIC_VIEW_JS__",
        ] {
            assert!(
                HTML_TEMPLATE.contains(p),
                "vendored template.html is missing JS chunk slot {p}"
            );
        }
    }

    /// The twelve data tokens live inside the vendored JS chunks, not in the
    /// shell — assert each chunk still declares the ones it owns.
    #[test]
    fn vendored_chunks_declare_all_data_slots() {
        for (chunk, name, tokens) in [
            (
                STORY_VIEW_JS,
                "story_view.js",
                &[
                    "__NODES_DATA__",
                    "__LINKS_DATA__",
                    "__TASK_COLORS__",
                    "__PIPELINE_COLORS__",
                    "__NODESET_COLORS__",
                    "__USER_COLORS__",
                ][..],
            ),
            (
                SCHEMA_VIEW_JS,
                "schema_view.js",
                &["__SCHEMA_DATA__", "__SCHEMA_GRAPH_DATA__"][..],
            ),
            (
                MEMORY_VIEW_JS,
                "memory_map.js",
                &["__MEMORY_DATA__", "__SEARCH_EVENTS__"][..],
            ),
            (
                SEMANTIC_VIEW_JS,
                "semantic_map.js",
                &["__SEMANTIC_CLUSTERS__"][..],
            ),
            (
                SEMANTIC_LAYOUT_JS,
                "semantic_layout chunk",
                &["__SEMANTIC_POSITIONS__"][..],
            ),
        ] {
            for t in tokens {
                assert!(
                    chunk.contains(t),
                    "vendored {name} is missing data slot {t}"
                );
            }
        }
    }

    #[test]
    fn safe_json_embed_escapes_closing_script() {
        let v = serde_json::json!({"x": "</script>"});
        let out = safe_json_embed(&v).expect("json encode");
        assert!(out.contains("<\\/script>"));
        assert!(!out.contains("</script>"));
    }

    #[test]
    fn pipeline_layout_chunk_is_the_upstream_empty_stub() {
        assert_eq!(PIPELINE_LAYOUT_JS, "");
    }

    #[test]
    fn semantic_layout_chunk_matches_upstream_one_liner() {
        assert_eq!(
            SEMANTIC_LAYOUT_JS,
            "window._semanticPositions = __SEMANTIC_POSITIONS__;"
        );
    }
}
