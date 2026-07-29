# Hybrid Retriever Port — Sequential Implementation Driver

> **How to use:** Give this whole document to Claude Code as the prompt. It drives the
> implementation of every task under `docs/plans/hybrid-retriever/` on a dedicated branch,
> one task at a time, each through a fixed 5-agent sequential pipeline. The driver (you, the
> top-level assistant) owns branch setup, task ordering, gating, and the revision loop; each
> numbered role below is run as a **separate subagent** via the Agent tool.

---

## Mission

Implement all 19 planned task documents (12 in `phase-1/`, 7 in `phase-2/`) that port the
Python cognee `HybridRetriever` (`HYBRID_COMPLETION`) to this Rust workspace. Work strictly
**one task at a time**; a task is only started after the previous task is committed.

## Ground rules (apply to every agent, every task)

- **Branch:** all work happens on `feat/hybrid-retriever-port`. Never commit to `main`, never
  `push`, never open a PR unless the user later asks.
- **Conventions:** obey `.claude/CLAUDE.md` and the global `CLAUDE.md`. No `unwrap()` in
  non-test code (use `expect("why")` or `?`/`map_err`); `thiserror` in libs, `anyhow` in
  bins; public traits `Send + Sync`. Tests run in **debug** (no `--release`).
- **Verification commands:** `cargo check --all-targets` for compile; `cargo test` (debug) for
  tests; **`scripts/check_all.sh`** is the full gate (fmt + check + clippy `-D warnings` + all
  binding checks) and must be green before any commit.
- **One commit per task**, authored by Agent 5 only. Commit message trailer (exactly):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TJVd1B41nJ6TSPJ3jr87cS
  ```
- **Fail loud, don't fake:** if checks fail, report the real output. Never mark a task done
  with red checks or skipped steps.

## One-time setup (driver does this before task 1)

1. Confirm the working tree is clean (`git status`). If dirty, stop and ask the user.
2. `git fetch origin`, `git checkout main`, `git pull --ff-only origin main`.
3. Create/checkout the branch from latest main:
   `git checkout -b feat/hybrid-retriever-port origin/main` (if it already exists, check it
   out and resume from the first non-`completed` task).
4. Read `docs/plans/hybrid-retriever/README.md` and both `phase-*/README.md` to load the task
   list, dependencies, and the 8 locked decisions.

## Task order (strict — this is a valid topological order of the dependencies)

```
P1-01 → P1-02 → P1-03 → P1-04 → P1-05 → P1-06 → P1-07 → P1-08 → P1-09 → P1-10 → P1-11 → P1-12
      → P2-01 → P2-02 → P2-03 → P2-04 → P2-05 → P2-06 → P2-07
```

Do **not** start a task until the previous one has status `completed` and is committed. If a
task becomes `blocked`, **halt the whole run** and report to the user with the reason.

## Status ledger convention

- Each task doc carries a status line directly under its H1:
  `**Status:** <pending | in-progress | in-review | ready | completed | blocked | obsolete>`
  (Agent 1 inserts it if missing.)
- The root `README.md` "Full task index" table gets a Status column the driver keeps current.
- Transitions: Agent 1 → `in-progress`; Agent 3/4 may bounce to `in-review`; Agent 4 pass →
  `ready`; Agent 5 → `completed` (or Agent 1 → `obsolete` when a task is already satisfied).

---

## Per-task pipeline

For the current task `<ID>` at `docs/plans/hybrid-retriever/phase-<N>/<ID>.md`, run these five
subagents **in strict sequence**. Do not launch the next agent until the current one has
returned and its gate is satisfied. Give each subagent: the task-doc path, the branch name,
the ground rules above, and the previous agent's structured output.

### Agent 1 — Refiner *(may edit the task doc + status)*

- **Goal:** make sure the task is real, current, and correctly specified **before** any code
  is written.
- **Do:** Read the task doc. Verify every cited file path / symbol / line-ref against the
  *current* code (the plan was authored earlier — refs may have drifted). Confirm the proposed
  steps actually achieve the stated Goal and Checkpoints. Detect staleness: is any part already
  implemented (by an earlier task in this run, or upstream on main)? Is the task now obsolete?
- **May:** edit the task doc — correct refs, tighten/repair steps, adjust scope — and set the
  **Status**. Do **not** touch production code.
- **Output (structured):** `verdict: proceed | obsolete | blocked`; list of edits made;
  a crisp restatement of what Agent 2 must implement; any newly discovered prerequisites.
- **Gate:** `proceed` → set status `in-progress`, continue to Agent 2. `obsolete` → set status
  `obsolete` with a note, **skip Agents 2–4**, hand to Agent 5 to record it (no code commit;
  commit only the doc/status change). `blocked` → set `blocked`, halt the run.

### Agent 2 — Implementer

- **Goal:** implement the task exactly per the refined doc.
- **Do:** write the code across the files the doc lists (and only what the task needs). Match
  surrounding style, comment density, and idioms. Keep `cargo check --all-targets` green as you
  go. Add/adjust the tests the doc's Testing section specifies. Honor all conventions.
- **Must not:** mark anything complete, commit, or edit the task doc's status.
- **Output:** summary of changes (files + what changed), any deviations from the doc and why,
  and the `cargo check` / relevant `cargo test` results.
- **Gate:** `cargo check --all-targets` compiles → continue to Agent 3.

### Agent 3 — Reviewer *(code quality)*

- **Goal:** judge the diff for cleanliness, **correctness**, absence of duplication, and
  consistency with existing code — independent of whether it merely compiles.
- **Do:** inspect `git diff` on the branch. Check: no `unwrap()` in non-test code; errors
  propagated properly; no copy-paste that should be a shared helper; naming/patterns match the
  crate; no dead code, no accidental API breakage; the change is faithful to the doc's parity
  notes (exact constants, id schemes, edge cases).
- **Output:** `verdict: pass | revise`; findings list (each: file:line, problem, suggested
  fix, severity). This agent reports; it does not itself rewrite.
- **Gate:** `pass` → Agent 4. `revise` → **revision loop** (see below).

### Agent 4 — Readiness / QA *(state validation)*

- **Goal:** confirm the task is genuinely implemented and the repo is in a committable state.
- **Do:** run the full gate — `scripts/check_all.sh` — and the task's own
  **Checkpoints / acceptance** list from the doc, one by one. Confirm the Goal is met and no
  checkpoint is unmet. Confirm tests the doc requires actually exist and pass in debug.
- **Output:** `verdict: ready | not-ready`; the actual command output (pass/fail per gate);
  a checklist mapping each acceptance checkpoint to evidence.
- **Gate:** `ready` → set status `ready`, continue to Agent 5. `not-ready` → **revision loop**.

### Agent 5 — Completer & Committer

- **Goal:** record completion and commit the task as a single, well-described commit.
- **Do:** set the task doc **Status:** to `completed` (or `obsolete` if routed here from Agent
  1); update the Status column for `<ID>` in the root `README.md`. Then `git add -A` and commit
  with a meaningful message:
  - Subject: `<type>(<scope>): <ID> <concise summary>` (e.g. `feat(search): P1-01 add HYBRID_COMPLETION SearchType + wire surfaces`).
  - Body: what changed and why, notable parity decisions, and any documented limitations
    carried forward. End with the required trailer (above).
- **Output:** the commit hash and one-line confirmation.
- **Gate:** commit succeeds → driver advances to the next task.

---

## Revision loop (when Agent 3 = `revise` or Agent 4 = `not-ready`)

1. Re-run **Agent 2 (Implementer)** with the findings as its input; it fixes only those items.
2. Re-run the agent that failed (3, then 4) to re-verify.
3. Repeat up to **3 cycles per task**. If still failing after 3 cycles, set status `blocked`,
   record the outstanding findings in the task doc's "Risks & open items", **halt the run**,
   and report to the user — do not commit a half-done task.

## Progress reporting

After each task commits, the driver prints a one-line status: `✅ <ID> committed <hash> — <n>/19 done`.
On halt, print which task blocked and why. Do not summarize file-by-file diffs to the user
unless asked — report conclusions and commit hashes.

## Model guidance (optional)

Default all agents to the session model. For the three docs the plan flagged as complex
(`P1-08`, `P1-09`, `P2-05`), consider a stronger model for Agents 2–4 (implement/review/QA),
since those carry the load-bearing parity logic.
