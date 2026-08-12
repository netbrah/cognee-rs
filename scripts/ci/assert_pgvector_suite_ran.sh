#!/usr/bin/env bash
#
# Assert that the pgvector suites actually executed against a live Postgres,
# instead of skipping — or compiling — themselves into a green no-op.
#
# Two independent ways this lane can pass without testing anything, both of which
# have happened:
#
#   1. **The feature is missing.** Both files are `#![cfg(feature = "pgvector")]`
#      and `crates/vector` declares no default features, so
#      `cargo test -p cognee-vector --test pgvector_close` compiles the file down
#      to zero cases: libtest prints "running 0 tests", exits 0, and emits no skip
#      marker. Indistinguishable from success. This is how the lane shipped, for
#      the span tests as well as the teardown ones.
#   2. **The URL is missing.** Every case returns early — still reporting `ok` —
#      when its Postgres URL is absent, so libtest's exit status proves nothing.
#
# The sibling of scripts/ci/assert_pggraph_suite_ran.sh, kept as its own file
# because the two lanes assert different case names; the shared reasoning is
# written down once, in that script's header.
#
# Usage: assert_pgvector_suite_ran.sh <spans-log> <close-log>

set -uo pipefail

SPANS_LOG="${1:?usage: $0 <spans-log> <close-log>}"
CLOSE_LOG="${2:?usage: $0 <spans-log> <close-log>}"

# Floors rather than exact counts, so adding cases does not break the guard.
MIN_SPAN_TESTS="${MIN_SPAN_TESTS:-3}"
MIN_CLOSE_TESTS="${MIN_CLOSE_TESTS:-2}"

# Cases that must have run by name. A count floor is not enough for these: they
# are what separates "the pool was closed" from "the process happened to exit",
# so one disappearing has to fail the lane rather than lower a total.
CLOSE_TESTS=(
  close_releases_the_pool_while_the_adapter_is_still_held
  close_does_not_touch_a_caller_owned_connection
)

fail() {
  echo "::error::$*"
  exit 1
}

# ── 0. The logs exist ───────────────────────────────────────────────────────
# `if grep A B` treats grep's exit 2 ("cannot open B") the same as exit 1 ("no
# match"), i.e. as proof of success. An unreadable log is a failure, not a pass.
for log in "$SPANS_LOG" "$CLOSE_LOG"; do
  [[ -s "$log" ]] ||
    fail "$log is missing or empty — the test step produced no output, so nothing can be asserted about this run."
done

# ── 1. Nothing announced a skip ─────────────────────────────────────────────
# Globbed variable name: renaming PGVECTOR_TEST_URL rewrites the Rust-side
# message too, and a hard-coded pattern would stop matching in exactly the
# scenario this guard exists for. Filtered for `eprintln!` because
# `cargo test 2>&1` also captures rustc diagnostics, and rustc echoes the
# offending source line verbatim — a future warning pointing at a skip message
# would otherwise fail a run in which every test passed.
matches="$(grep -hE '[A-Z_]+ not set .* skipping' "$SPANS_LOG" "$CLOSE_LOG")"
grep_status=$?
case "$grep_status" in
  0) ;;
  1) matches="" ;;
  *) fail "grep exited $grep_status while scanning the test logs; refusing to report success on unverified output." ;;
esac
skips="$(printf '%s\n' "$matches" | grep -v 'eprintln!' | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$skips" ]]; then
  printf '%s\n' "$skips"
  fail "pgvector tests skipped — the suite never reached a live Postgres. The URL is unset or empty in the test process."
fi

# ── 2. A plausible number of cases actually ran ─────────────────────────────
# This is the check that catches a missing `--features pgvector`: zero cases.
assert_count() {
  local log="$1" floor="$2" what="$3"
  local passed
  passed="$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$log" | head -1)"
  [[ -n "$passed" ]] ||
    fail "no libtest summary line in $log — the $what binary did not run to completion."
  ((passed >= floor)) ||
    fail "only $passed $what tests ran, expected at least $floor. The pgvector/testing features have probably stopped applying, which compiles the suite down to nothing — a step that passes while asserting nothing."
  echo "$what: $passed tests ran."
}

assert_count "$SPANS_LOG" "$MIN_SPAN_TESTS" "pgvector span"
assert_count "$CLOSE_LOG" "$MIN_CLOSE_TESTS" "pgvector pool-teardown"

# ── 3. The pool-teardown cases ran by name ─────────────────────────────────
for test_name in "${CLOSE_TESTS[@]}"; do
  grep -qF -- "$test_name ... ok" "$CLOSE_LOG" ||
    fail "pool-teardown test $test_name did not run (or did not pass) — see $CLOSE_LOG. This is the suite that proves the pgvector pool is closed rather than leaked (topoteretes/cognee-rs#132); a green lane without it means nothing."
done

echo "pgvector suites verified: both ran against a live Postgres, ${#CLOSE_TESTS[@]} pool-teardown cases by name, no skip markers."
