# Development Workflow

## Authority

`WORKFLOW.md` and `CLAUDE.md` are authoritative. This page is the operational guide for readers.

## Default Cycle

Every non-trivial code change follows:

```text
SPEC-READ → TEST-READ → FACT-CHECK → SPEC-CORRECT → [DESIGN] → TEST-WRITE → IMPLEMENT → TEST-RUN → REVIEW → [PERF-CONFIRM] → SPEC-UPDATE
```

## Mode Classification

Before editing, classify the work:

| Mode | Meaning | First Durable Action |
|------|---------|----------------------|
| New behavior | Behavior is not currently promised | Update/add spec, then write failing test. |
| Bug fix | Existing behavior violates spec/reference/test | Add or preserve reproducer. |
| Refactor | Behavior unchanged | Existing tests are the spec; add characterization only if weak. |
| Performance-sensitive | Equivalent behavior, faster/slower path | Establish equivalence and baseline before claiming speedup. |
| Spec drift | Spec, tests, and code disagree | Identify authority before coding. |

## Implementation Rules

- Read the relevant spec before code.
- Read existing tests before writing tests.
- Make the smallest change that satisfies the behavior.
- Do not broaden scope opportunistically.
- Do not weaken tests.
- Do not hide failures as future work without a visible TODO/task.
- For math/numerical changes, cite evidence in code only when the reason is non-obvious.

## Review Rules

After code changes, review is automatic and required before the work is considered complete:

- +1 self-review with fresh context.
- +2 independent review.
- Both reviews must pass, and blockers must be fixed before reporting done.

If review flags non-blocking issues, record whether they were fixed, deferred visibly, or accepted with rationale.

## Commit Rules

Before commit:

- Check `git status --short`.
- Inspect the staged diff.
- Exclude generated raw data, temporary artifacts, credentials, and unrelated files.
- Use a concise commit message describing purpose and impact.
- Do not amend unless explicitly requested.

## Documentation Rules

Update docs according to the kind of change:

| Change | Docs to Check |
|--------|---------------|
| Behavior change | `spec/README.md`, tests, wiki if reader-facing. |
| New test category | `harmonyos_3dgs/tests/TEST_PLAN.md`, `dev_notes/session_state.md`. |
| New workflow rule | `WORKFLOW.md`, `CLAUDE.md`, wiki. |
| New gotcha | Auto-memory T1 first, then synced repo `memory/gotchas.md` or focused memory file. |
| Milestone result | `dev_notes/session_state.md`, `dev_notes/captains_log.md`, wiki if durable. |

## Phase Transitions

Pause and re-check direction when:

- Moving from exploration to implementation.
- Changing the planned approach.
- A fix fails and a second hypothesis is needed.
- A destructive or externally visible action is needed.
- A milestone is complete and knowledge should be consolidated.

## Definition of Done

A change is done when:

- Behavior is specified or covered by existing spec/tests.
- Tests relevant to the change pass.
- Full suite status is known or explicitly not run with reason.
- Review blockers are resolved.
- Durable docs/state are updated.
- Remaining risks are visible.
