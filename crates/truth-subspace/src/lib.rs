//! Pure truth-subspace alignment math — no I/O, no database access, no LLM
//! calls. Ported from Python `cognee/modules/truth_subspace/align.py`.
//!
//! Everything here is NEUTRAL when inputs are missing/empty/zero:
//! [`align::truth_score`] returns `0.5` and [`align::truth_factor`] returns
//! `1.0`, so callers that pass nothing leave baseline scoring untouched. This
//! keeps the Phase-2 truth-subspace re-ranking knobs (`use_truth_weight` /
//! `build_truth_subspace`, both default-off) safe by construction.
#![forbid(unsafe_code)]

pub mod align;
