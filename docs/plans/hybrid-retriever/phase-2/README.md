# Phase 2 — Truth-subspace reranking lane

Phase 2 adds the optional truth-subspace reranking lane to the HybridRetriever
(`SearchType::HybridCompletion`, shipped in Phase 1). It ports Python's
`truth_subspace` package and `session_distillation`/`build_truth_subspace`
orchestration into a new `cognee-truth-subspace` crate plus targeted glue in
`cognee-graph`, `cognee-vector`, `cognee-cognify`, and `cognee-search`. Every
piece lands **default-off**: with `use_truth_weight=false` and
`build_truth_subspace=false` (the defaults), behavior is bit-for-bit identical
to Phase 1.

## Goals

- Port the pure alignment math (`align.py`) and centroid slot logic
  (`centroids.py`/`models.py`) into a standalone, dependency-light
  `cognee-truth-subspace` crate.
- Persist per-dataset centroid payloads (vector store) and per-node
  truth-state (`truth_epoch`, alignment) (graph store), with deterministic
  round-tripping.
- Port the session-learnings distillation gate (curate → propose → evaluate →
  publish) as `improve()` Stage 2c, tagging published lessons with
  `session_learnings`/`session_learnings:{session_id}` NodeSets.
- Port `build_truth_subspace` orchestration as an opt-in, fail-open `improve()`
  Stage 2d that rebuilds centroids from session learnings and writes updated
  truth-state back to the graph.
- Wire `use_truth_weight` into `HybridRetriever`'s chunk ranking: project the
  query into truth-subspace coordinates, fetch candidate chunks' truth state,
  and apply `truth_factor` as a ranking multiplier — failing open to the
  Phase‑1 baseline ranking on any error, missing data, or multi-dataset scope.
- Keep every new surface reachable only behind explicit, default-off flags so
  Phase 1 users see zero behavior change.

## Checkpoints

- `cognee-truth-subspace` crate exists, is a workspace member, and
  `cargo test -p cognee-truth-subspace` passes covering the full ported
  Python `align.py`/`centroids.py`/`models.py` test suites (cosine,
  node/query_coords, truth_score, truth_factor, stable_signature, centroid
  id/learning id derivation, normalize/pad/weighted_centroid, nearest-slot
  assignment, `centroids_changed`).
- `VectorDB::upsert_raw_vectors` exists on all four adapters and
  `load_centroids`/`upsert_centroids` round-trip per-dataset
  `TruthCentroidPayload`s deterministically (sort by slot, dataset filter,
  partial slots, empty-upsert no-op).
- `GraphDBTrait::get_node_truth_state`/`set_node_truth_state` exist with a
  default impl plus Ladybug/Postgres/Mock overrides; `-1` is the consistent
  sentinel for missing/invalid `truth_epoch` across all impls.
- `distill_sessions` runs as `improve()` Stage 2c: curates session Q&A
  batches, proposes/evaluates/publishes lessons, and tags published documents
  with both NodeSets in the same `add()` call; stage 2b2
  (gated-guidance/agent-context extraction) is explicitly absent since
  `cognee-session` has no equivalent concept.
- `build_truth_subspace` runs as `improve()` Stage 2d only when
  `has_sessions && build_truth_subspace`, reproduces all 6 Python return
  shapes (including the signature-dropping quirk on empty
  `rebuilt_centroids`), and never aborts `improve()` on internal failure;
  `ImproveParams.build_truth_subspace` defaults to `false` end-to-end (lib,
  HTTP DTO, bindings-common, CLI).
- `HybridRetriever` ranking applies `truth_factor` only when
  `use_truth_weight=true`, `q_coords` is non-empty, and the candidate's stored
  `truth_epoch` matches the current epoch, multiplicatively after the
  importance factor; `build_truth_context` fails open to `(None, None, None)`
  on any missing precondition or store error, and `SearchParams.dataset_id`
  is `Some` only for an exactly-singleton `dataset_ids` resolution.
- `cargo check --all-targets`, the full `cargo test` matrix for
  `cognee-truth-subspace`/`cognee-graph`/`cognee-vector`/`cognee-cognify`/
  `cognee-search`, and `scripts/check_all.sh` all pass with the Phase‑2 flags
  left at their default-off values producing no observable change from
  Phase 1.

## Task index

| Task | Title | Depends on | Link |
|---|---|---|---|
| P2-01 | Truth-subspace: pure alignment math | — | [./P2-01.md](./P2-01.md) |
| P2-02 | Truth-subspace: centroid slot logic | P2-01 | [./P2-02.md](./P2-02.md) |
| P2-03 | Truth-subspace: vector persistence | P1-02, P2-02 | [./P2-03.md](./P2-03.md) |
| P2-04 | Truth-subspace: graph truth-state methods | — | [./P2-04.md](./P2-04.md) |
| P2-05 | Session-learnings distillation gate | — | [./P2-05.md](./P2-05.md) |
| P2-06 | build_truth_subspace orchestration | P2-01, P2-02, P2-03, P2-04, P2-05 | [./P2-06.md](./P2-06.md) |
| P2-07 | Hybrid retriever truth-weight glue | P2-01, P2-02, P2-03, P2-04, P2-06, P1-01, P1-06, P1-07, P1-09 | [./P2-07.md](./P2-07.md) |

## Sequencing notes

- **Critical path**: P2-01 → P2-02 → P2-03 → P2-06 → P2-07. This is the
  longest dependency chain (5 deep) and gates the final retriever-glue task;
  P2-04 and P2-05 must also land before P2-06 but are not on this chain's
  critical timing since both can proceed in parallel with P2-01→P2-03.
- **Parallelizable now (no Phase-2 prerequisites)**: P2-01, P2-04, and P2-05
  can all start immediately and independently — P2-01 is a pure-math
  standalone crate, P2-04 only touches `cognee-graph`, and P2-05 only touches
  `cognee-cognify`/`improve()`. None of the three depends on the others.
- **P2-02** depends only on P2-01 (needs the crate scaffold and `align.py`
  port to build centroid logic on top of) and can start as soon as P2-01
  lands, in parallel with P2-04/P2-05.
- **P2-03** depends on P2-02 (centroid payload shape) and the Phase-1 task
  P1-02 (`VectorDB::retrieve`/by-id fetch), which is an external
  cross-phase dependency — confirm P1-02 has landed before starting P2-03.
- **P2-06** is the join point: it cannot start until all of P2-01, P2-02,
  P2-03, P2-04, and P2-05 are done, since it orchestrates centroid rebuild
  (P2-01/02/03), truth-state writes (P2-04), and consumes distilled session
  learnings (P2-05).
- **P2-07** is the final task and the widest fan-in: beyond the full
  Phase-2 chain (P2-01/02/03/04/06) it also needs Phase-1 tasks P1-01
  (`SearchType::HybridCompletion`), P1-06 (`get_neighborhood`), P1-07
  (hybrid chunk/BM25 lane), and P1-09 (HybridRetriever context/registration)
  — it is the only Phase-2 task with a hard cross-phase dependency on nearly
  all of Phase 1's retriever plumbing, not just Phase 2's math/persistence
  layer. Do not start P2-07 until both chains are complete.
- Net effect: with three engineers, P2-01/P2-04/P2-05 can run concurrently
  from day one; P2-02 joins after P2-01; P2-03 joins after P2-02 and Phase-1's
  P1-02; P2-06 is a hard sync point for the whole team; P2-07 closes the
  phase and additionally requires Phase-1's retriever-plumbing tasks to be
  merged first.
