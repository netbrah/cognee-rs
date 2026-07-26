# HybridRetriever Rust Port — Plan

## Overview

`HybridRetriever` (Python `SearchType.HYBRID_COMPLETION`) is cognee's most
elaborate retrieval mode: a multi-lane RAG retriever that fuses several
independently-scored candidate sets into one ranked context before handing it
to the completion LLM. Concretely:

- **Chunk lane** — three independent scorers over document chunks and text
  summaries — Okapi BM25 (real IDF/TF lexical scoring, not the existing
  Jaccard `LexicalRetriever`), dense vector similarity, and summary vector
  similarity — merged via Reciprocal Rank Fusion (RRF), then re-weighted by
  each `DataPoint`'s `importance_weight`.
- **Entity + facts lane** — graph-neighborhood expansion around entities
  mentioned in the top chunks, producing deduplicated "facts" (edge
  descriptions) gated by a minimum word count and formatting rules.
- **Optional truth-subspace lane** (Phase 2, default-off) — reranks the chunk
  lane by projecting the query and candidate chunks into a learned
  "truth-subspace" derived from distilled session learnings, applied as a
  multiplicative factor on top of the RRF + importance score.
- **Optional global-context lane** (Phase 2, default-off) — injects a
  dataset-level global-context index alongside the per-query lanes.

All lanes are merged into a single context payload (chunks, entities, facts,
formatted separately) and passed to the LLM as `HYBRID_COMPLETION`'s output.

**Feasibility verdict: portable.** The default-config core (chunk lane +
entity/facts lane, RRF fusion, importance weighting) requires no new
architectural concepts beyond what cognee-rust already has — it needs four new
primitives (`VectorDB::retrieve`-by-id, batched direction-correct
`GraphDBTrait::get_neighborhood`, a real BM25 scorer, `importance_weight` on
`DataPoint`) plus a new retriever module. This is **Phase 1**. The
truth-subspace reranking lane is materially larger (a new alignment/centroid
math crate, new graph/vector persistence for truth-state, a session-learnings
distillation pipeline, and an orchestration stage to rebuild the subspace) and
is functionally independent of the Phase 1 core — it is deferred as **Phase
2**, wired on the wire format from day one but inert until Phase 2 lands.

## Locked decisions

These eight decisions are binding on every task doc in this tree; do not
relitigate them at the task level.

1. **Real Okapi BM25**, not Jaccard: `k1=1.5`, `b=0.75`,
   `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`, summed over unique query terms
   after removing `DEFAULT_STOP_WORDS`. The existing Jaccard-based
   `LexicalRetriever` is untouched and not reused.
2. **Acceptance bar is structural parity**, not bit-identical output: same
   algorithm + exact constants + overlapping top-k, validated against the
   `e2e-cross-sdk` harness at the same ~50% tolerance already used for
   `cognify` structural comparisons.
3. **`importance_weight` lives on `DataPoint`** (the base model), matching
   Python's `DataPoint.importance_weight`, not bolted onto a narrower type.
4. **Phase 2 is fully spec'd but deferred at execution**: the truth-subspace
   lane (`use_truth_weight`) and the truth-subspace-rebuild orchestration
   (`build_truth_subspace`) are both default-off flags, documented and wired
   on the request/DTO shape in Phase 1, implemented in Phase 2.
5. **BM25 rebuilds its corpus per-query.** It must NOT reuse the existing
   lexical retriever's process-lifetime `OnceCell` corpus cache — every hybrid
   search call re-derives corpus statistics from the current candidate scope.
6. **Add batched `get_neighborhood(ids, depth) -> (nodes, edges)` to
   `GraphDBTrait`** (~4 backend implementations) returning the *true stored
   edge direction*. This is a real bug fix: it corrects ladybug's existing
   undirected edge-direction flip, not just a hybrid-retriever convenience.
7. **No backfill migration** for existing datasets missing `importance_weight`
   / `source_chunk_id` / truth-state properties — degradation is documented,
   not migrated.
8. **Expose all hybrid knobs on the wire in PR1**, including the two
   Phase-2-reserved keys (`use_truth_weight`, `build_truth_subspace` /
   `include_global_context_index`), which stay inert until Phase 2 lands
   rather than being added piecemeal later.

## Structure

This document tree has three layers:

- **This root (`README.md`)** — the plan-wide entry point: overview, locked
  decisions, phase index, full task index, and the fable-validation record.
- **[`phase-1/README.md`](./phase-1/README.md)** — Phase 1 scope, goals,
  phase-level checkpoints, its own task index, and sequencing/parallelism
  notes for the twelve Phase 1 tasks.
- **[`phase-2/README.md`](./phase-2/README.md)** — Phase 2 scope, goals,
  phase-level checkpoints, its own task index, and sequencing/parallelism
  notes for the seven Phase 2 tasks.
- **Per-task docs** (`phase-1/P1-NN.md`, `phase-2/P2-NN.md`) — one file per
  task, each self-contained with prerequisites, exact file/line references
  verified against the live repo, step-by-step implementation instructions,
  parity notes against the Python reference, checkpoints, and risks.

Task IDs encode phase and order (`P1-01`…`P1-12`, `P2-01`…`P2-07`) but do not
imply strict serialization — see each phase README's sequencing notes for
what can run in parallel.

## Phase index

| Phase | Goal | # tasks | Link |
|---|---|---|---|
| Phase 1 | Ship the default-config `HybridRetriever` core (`SearchType::HybridCompletion`) at behavioral parity with Python's defaults: BM25 + vector + summary chunk lane, entity + facts lane, RRF fusion, importance weighting, full wire/binding plumbing — with the two Phase-2 knobs present but inert. | 12 | [./phase-1/README.md](./phase-1/README.md) |
| Phase 2 | Add the optional, default-off truth-subspace reranking lane: pure alignment/centroid math crate, vector + graph persistence for centroids/truth-state, session-learnings distillation, `build_truth_subspace` orchestration, and the ranking glue in `HybridRetriever` — zero behavior change for Phase-1 users. | 7 | [./phase-2/README.md](./phase-2/README.md) |

## Full task index

| Task | Phase | Title | Link | Status |
|---|---|---|---|---|
| P1-01 | 1 | SearchType HYBRID_COMPLETION plumbing | [./phase-1/P1-01.md](./phase-1/P1-01.md) | completed |
| P1-02 | 1 | Vector retrieve-by-id | [./phase-1/P1-02.md](./phase-1/P1-02.md) | completed |
| P1-03 | 1 | Per-query Okapi BM25 chunk scorer | [./phase-1/P1-03.md](./phase-1/P1-03.md) | completed |
| P1-04 | 1 | importance_weight on DataPoint | [./phase-1/P1-04.md](./phase-1/P1-04.md) | completed |
| P1-05 | 1 | Shared helpers + payload fixes | [./phase-1/P1-05.md](./phase-1/P1-05.md) | completed |
| P1-06 | 1 | get_neighborhood on GraphDBTrait | [./phase-1/P1-06.md](./phase-1/P1-06.md) | completed |
| P1-07 | 1 | Hybrid module — chunk + BM25 lane | [./phase-1/P1-07.md](./phase-1/P1-07.md) | completed |
| P1-08 | 1 | Hybrid module — entity + facts lane | [./phase-1/P1-08.md](./phase-1/P1-08.md) | completed |
| P1-09 | 1 | Hybrid module — context, retriever impl, registration | [./phase-1/P1-09.md](./phase-1/P1-09.md) | completed |
| P1-10 | 1 | Orchestrator session-cache wiring + tests | [./phase-1/P1-10.md](./phase-1/P1-10.md) | completed |
| P1-11 | 1 | Cross-SDK harness coverage for HYBRID_COMPLETION | [./phase-1/P1-11.md](./phase-1/P1-11.md) | completed |
| P1-12 | 1 | Documentation updates for HYBRID_COMPLETION | [./phase-1/P1-12.md](./phase-1/P1-12.md) | completed |
| P2-01 | 2 | Truth-subspace: pure alignment math | [./phase-2/P2-01.md](./phase-2/P2-01.md) | completed |
| P2-02 | 2 | Truth-subspace: centroid slot logic | [./phase-2/P2-02.md](./phase-2/P2-02.md) | pending |
| P2-03 | 2 | Truth-subspace: vector persistence | [./phase-2/P2-03.md](./phase-2/P2-03.md) | pending |
| P2-04 | 2 | Truth-subspace: graph truth-state methods | [./phase-2/P2-04.md](./phase-2/P2-04.md) | pending |
| P2-05 | 2 | Session-learnings distillation gate | [./phase-2/P2-05.md](./phase-2/P2-05.md) | pending |
| P2-06 | 2 | build_truth_subspace orchestration | [./phase-2/P2-06.md](./phase-2/P2-06.md) | pending |
| P2-07 | 2 | Hybrid retriever truth-weight glue | [./phase-2/P2-07.md](./phase-2/P2-07.md) | pending |

## Fable validation

Three task docs were selected for adversarial ("fable") validation — a
deliberate stress test of the highest-risk, most cross-referencing docs in
the tree (the two Phase 1 hybrid-module tasks with the deepest Python-parity
surface, plus the one Phase 2 task with the widest divergence from Rust's
existing session/improve machinery):

| Task | Verdict | Outcome |
|---|---|---|
| P1-08 (Hybrid module — entity + facts lane) | patched | Fixed a wrong claim about `belongs_to_set` propagation (the real gap is chunk NodeSet names not being inherited by entities, not "never set"), redirected an `is_a` citation off a test-only mock onto the real production emitter, rewrote `connection_edge_type_id`'s proposed signature to restore the top-level `edge_text`-first check it had silently dropped, corrected a prerequisite's description of `mock.rs` from "production backend" to testing-gated, and completed the `TestGraphDb`-implementor risk list (search_execution_builder.rs and http-server's `FailingGraphDB` were missing). |
| P1-09 (Hybrid module — context, retriever impl, registration) | patched | Fixed a top-k default-resolution bug that would have made Rust ignore request-level `top_k` where Python honors it (patched to the correct three-layer `config[key] -> params.top_k -> constructor default` chain), moved `node_name`/`node_name_filter_operator` from the wrong config location to the existing dedicated `SearchParams` fields, harmonized an inconsistent batch-rejection error message, corrected a wrong claim about `dataset_scope.rs`'s field reads, fixed two Python line-ref off-by-ones, and added a missing note (blank-name entity skip rule) plus a warning about `prepare_search_result` not surfacing hybrid graphs today. |
| P2-05 (Session-learnings distillation gate) | patched | Fixed a vector-metadata filter that, as originally specified, would have matched zero rows (NodeSet entries are JSON objects, not bare strings), corrected an off-by-one `improve.rs` insertion anchor, fixed several stale line-range citations (`SessionStore`, `models.py` length, `BatchedGraphs`, `_extract_agent_context`), and softened an overstated Prerequisites claim about `cognee-session`'s type surface. |

All three docs were revised in place after validation; no findings were left
unresolved. The remaining task docs (P1-01 through P1-07, P1-10 through P1-12,
P2-01 through P2-04, P2-06, P2-07) were not put through adversarial
validation as part of this pass.
