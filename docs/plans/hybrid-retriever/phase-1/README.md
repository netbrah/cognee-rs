# Phase 1 — HybridRetriever (HYBRID_COMPLETION) core port

Ship the default-config core `HybridRetriever` (`SearchType::HybridCompletion` /
wire `HYBRID_COMPLETION`) at behavioral parity with the Python defaults. The
truth-subspace weighting lane (`use_truth_weight`) and the global-context-index
lane (`include_global_context_index`) are wired on the request shape but stay
functionally inert in Phase 1 — their execution is Phase 2
(`../phase-2/README.md`).

## Goals

- Port the real Okapi BM25 lexical lane (k1=1.5, b=0.75, per-query corpus
  rebuild, `DEFAULT_STOP_WORDS`) alongside the existing vector and graph lanes,
  without touching or reusing the existing Jaccard `LexicalRetriever`.
- Merge chunk, entity, and fact candidates through the same RRF ranking,
  pairing, and gating logic as Python's `hybrid/{results,pairs,ranking,chunks,
  entities,facts}.py`, validated at **structural** parity (same algorithm +
  exact constants + overlapping top-k against the e2e-cross-sdk harness,
  ~50% tolerance like cognify) — not bit-identical output.
- Add the supporting primitives the retriever needs that don't exist yet:
  `VectorDB::retrieve` (fetch-by-id), `GraphDBTrait::get_neighborhood`
  (batched, direction-correct neighborhood fetch — this also fixes ladybug's
  undirected edge-direction bug), `importance_weight` on `DataPoint`, and the
  shared `EdgeType::retrieval_text` / `TextSummary.source_chunk_id` /
  `DEFAULT_HYBRID_USER_PROMPT_TEMPLATE` helpers.
- Register `SearchType::HybridCompletion` end-to-end: registry, HTTP DTO,
  C/Java/Python/TS bindings, docs, and `SearchParams`, exposing **all** hybrid
  knobs on the wire in this PR (including the two Phase-2-reserved keys,
  which stay inert) rather than adding them piecemeal later.
- Extend session/orchestrator plumbing (`build_used_graph_element_ids`) so
  hybrid search results feed the used-graph-element cache correctly, and cover
  the whole surface with unit, mock-integration, and cross-SDK e2e tests.
- Leave existing datasets to degrade gracefully (documented, no backfill
  migration for `importance_weight` / `source_chunk_id`).

## Checkpoints

Phase-level milestones an outside observer can verify once all twelve tasks
land:

1. `SearchType::HybridCompletion` round-trips as `"HYBRID_COMPLETION"` through
   serde and every binding surface (C, Java, Python, TS, HTTP `WireSearchType`),
   and `SearchBuilder`/`SearchTypeRegistry` resolves it to a working
   `HybridRetriever` (no longer `SearchError::UnsupportedSearchType`).
2. A hybrid search against a real dataset returns chunk, entity, and fact
   `SearchItem`s ranked by RRF + importance-weight, backed by real Okapi BM25
   scores, real vector similarity, and true-direction graph neighborhoods —
   all four new primitives (`VectorDB::retrieve`, `GraphDBTrait::
   get_neighborhood`, BM25 scorer, importance weight) are exercised in the
   same request path.
3. `get_context_batch`/`get_completion_batch` reject cleanly with
   `SearchError::InvalidInput` (no silent per-query fallback), and
   `build_used_graph_element_ids` populates `node_ids` (never `edge_ids`) from
   hybrid chunk/entity/edge-endpoint ids while excluding facts.
4. The e2e-cross-sdk harness runs `HYBRID_COMPLETION` through the Phase-1
   wire/status check plus a dedicated context-only structural parity test
   (entity-name Jaccard >= 0.3, passage/fact count tolerance), gated into the
   Phase-2 OPENAI_KEY CI lane, with no golden-fixture regeneration needed.
5. `docs/architecture.md`, `docs/operations.md`, `docs/configuration.md`,
   `docs/tools/cli.md`, and `docs/http-server/routers/search.md` all reflect
   the new search mode/wire-variant counts and the 8 active + 2 reserved
   hybrid knobs, with no stale mode/variant counts left behind.
6. `cargo check --all-targets`, `cargo test`, and `scripts/check_all.sh` pass
   at every task boundary; no `unwrap()` introduced in non-test code.

## Task index

| Task | Title | Depends on | Link |
|---|---|---|---|
| P1-01 | SearchType HYBRID_COMPLETION plumbing | — | [./P1-01.md](./P1-01.md) |
| P1-02 | Vector retrieve-by-id | — | [./P1-02.md](./P1-02.md) |
| P1-03 | Per-query Okapi BM25 chunk scorer | — | [./P1-03.md](./P1-03.md) |
| P1-04 | importance_weight on DataPoint | — | [./P1-04.md](./P1-04.md) |
| P1-05 | Shared helpers + payload fixes | — | [./P1-05.md](./P1-05.md) |
| P1-06 | get_neighborhood on GraphDBTrait | — | [./P1-06.md](./P1-06.md) |
| P1-07 | Hybrid module — chunk + BM25 lane | P1-02, P1-03, P1-05 | [./P1-07.md](./P1-07.md) |
| P1-08 | Hybrid module — entity + facts lane | P1-05, P1-06, P1-07 | [./P1-08.md](./P1-08.md) |
| P1-09 | Hybrid module — context, retriever impl, registration | P1-05, P1-07, P1-08 | [./P1-09.md](./P1-09.md) |
| P1-10 | Orchestrator session-cache wiring + tests | P1-09 | [./P1-10.md](./P1-10.md) |
| P1-11 | Cross-SDK harness coverage for HYBRID_COMPLETION | P1-04, P1-09 | [./P1-11.md](./P1-11.md) |
| P1-12 | Documentation updates for HYBRID_COMPLETION | P1-01, P1-09 | [./P1-12.md](./P1-12.md) |

## Sequencing notes

- **Foundation layer (no dependencies, parallelizable):** P1-01, P1-02, P1-03,
  P1-04, P1-05, P1-06 have no dependencies on each other and can all run
  concurrently — they touch disjoint files (`SearchType`/wire DTOs; the
  `VectorDB` trait + 4 adapters; a new standalone `bm25.rs`; `DataPoint` +
  cognify propagation; small shared helpers in `cognee-models`/`cognee-cognify`;
  the `GraphDBTrait` + 3 backend overrides). This is the widest parallelism
  window in Phase 1 — assign these six first if running multiple workers.
- **Critical path:** P1-02/P1-03/P1-05 → **P1-07** (chunk+BM25 lane) →
  P1-06/P1-05 → **P1-08** (entity+facts lane, also needs P1-07) → **P1-09**
  (context assembly, `HybridRetriever` impl, registration — also needs P1-05)
  → **P1-10** (orchestrator wiring) and, separately, **P1-11**/**P1-12** (both
  only need P1-09, plus P1-04 for P1-11 and P1-01 for P1-12). This
  P1-07 → P1-08 → P1-09 → P1-10 spine is the longest chain and gates
  everything downstream of it.
- **Second wave (after P1-09 lands):** P1-10, P1-11, and P1-12 can run in
  parallel with each other — P1-10 touches the orchestrator + new test files,
  P1-11 touches only `e2e-cross-sdk/`, and P1-12 touches only `docs/*.md`; none
  of the three share edited files.
- **Fan-in risk:** P1-07 depends on all three of P1-02/P1-03/P1-05, and P1-09
  depends on all three of P1-05/P1-07/P1-08 — these are the two points where
  parallel foundation work must fully land before the next module can compile.
  Land P1-05 early since it feeds both P1-07 and P1-08 directly.
- **Documentation and cross-SDK tasks (P1-11, P1-12) are safe to defer** to
  the very end of the phase without blocking any other task, but both name
  P1-09's not-yet-landed field names/wire shape as their highest-risk
  assumption — re-verify against the actual P1-09 diff before finalizing
  either.
