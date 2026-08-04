# Vendored visualization frontend

Everything in this directory is **vendored verbatim** from the upstream Python
[`cognee`](https://github.com/topoteretes/cognee) repository. Not a port, not an
adaptation — a byte-for-byte copy.

- **Upstream commit:** `38eece5b` (`38eece5bbb0cb9f5706fed908abd16dba0f5505e`)

| Vendored file            | Upstream source path                                       |
| ------------------------ | ---------------------------------------------------------- |
| `template.html`          | `cognee/modules/visualization/template.html`               |
| `views/ui_chrome.js`     | `cognee/modules/visualization/views/ui_chrome.js`          |
| `views/schema_view.js`   | `cognee/modules/visualization/views/schema_view.js`        |
| `views/story_view.js`    | `cognee/modules/visualization/views/story_view.js`         |
| `views/inspector.js`     | `cognee/modules/visualization/views/inspector.js`          |
| `views/memory_map.js`    | `cognee/modules/visualization/views/memory_map.js`         |
| `views/semantic_map.js`  | `cognee/modules/visualization/views/semantic_map.js`       |

## Rule: re-sync wholesale, never hand-edit

**Do not hand-edit any file in this directory.** If the frontend needs to
change, change it upstream in the Python repo first, then re-copy the whole set
here and bump the commit recorded above.

This rule exists because it was already broken once. Rust used to carry
`assets/graph_template.html`, a 1680-line hand-forked copy of Python's
pre-refactor monolith. It was edited in place, drifted silently, and ended up
several features behind: two tabs instead of four, no Memory or Semantic view,
no Story/Flow/Force layout switch, no schema inspector, and a Schema tab that
could never render because `schema_data` was hardcoded to `None` on the Rust
side. Hand-editing is exactly how that happened, so the resync rule is the fix.

Re-sync procedure:

```bash
git clone --depth 1 https://github.com/topoteretes/cognee.git /tmp/cognee-python
SRC=/tmp/cognee-python/cognee/modules/visualization
DST=crates/visualization/assets
cp "$SRC/template.html" "$DST/template.html"
for f in ui_chrome schema_view story_view inspector memory_map semantic_map; do
  cp "$SRC/views/$f.js" "$DST/views/$f.js"
done
# then update the commit hash in this file and re-run:
#   cargo test -p cognee-visualization
```

`cargo test -p cognee-visualization` is enough on its own: the crate declares
every feature its output ordering depends on (notably `serde_json/preserve_order`)
in its own `Cargo.toml`, so the `-p` build renders the same bytes as the
workspace binary. It used to inherit `preserve_order` from `cognee-database` by
feature unification, which made the `-p` build emit a different Schema tab than
the shipped one.

The orchestrator that stitches these together — the `__TOKEN__` substitutions —
lives in [`../src/html.rs`](../src/html.rs) and is the *only* Rust code that
should know about these filenames. `crates/visualization/tests/html_test.rs`
asserts that every token is resolved and that each chunk's distinctive
identifiers survived the copy, so a bad or partial resync fails the test suite
rather than silently shipping a broken page.

## Token slots

`template.html` and the `views/*.js` chunks between them declare 20
`__TOKEN__` placeholders. The full list, the substitution order (JS chunks
first, data payloads second) and the rationale are documented in
[`../src/html.rs`](../src/html.rs) and
[`../README.md`](../README.md).

Two of the tokens are *not* filled from a vendored file:

- `__PIPELINE_LAYOUT_JS__` → the empty string. Upstream
  `layouts/pipeline_layout.py:16-17` is a deliberate stub whose `emit_js()`
  returns `""`. The slot is kept so a future Phase 1d layout drops in without a
  template change.
- `__SEMANTIC_LAYOUT_JS__` → the single line
  `window._semanticPositions = __SEMANTIC_POSITIONS__;`, matching upstream
  `layouts/semantic_layout.py:200-206`. This is why the layout chunk must be
  substituted *before* the data tokens.

## Known limitation: CDN fetches

The vendored `template.html` loads d3 v7 from `https://d3js.org` and the Inter
webfont from `https://fonts.googleapis.com` / `https://fonts.gstatic.com`.
That is exactly what Python does, so it is **not a parity gap** — but it does
mean the generated HTML is not fully self-contained and degrades on offline or
air-gapped edge devices (no d3 ⇒ no graph, only the page chrome). Inlining d3
would be a deliberate divergence from upstream and is intentionally not done
here.
