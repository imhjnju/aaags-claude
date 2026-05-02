# harmonyos_3dgs — Workflow

## Implementation Cycle

Every code change follows this cycle. No shortcuts — "just mechanical" is not an exemption.

**Full Cycle** (spec-driven — this project maintains `spec/README.md`):

```
SPEC-READ → TEST-READ → FACT-CHECK → SPEC-CORRECT → [DESIGN] → TEST-WRITE (RED) → IMPLEMENT (GREEN) → TEST-RUN → REVIEW → [PERF-CONFIRM] → SPEC-UPDATE
```

1. **SPEC-READ**: Read `spec/README.md` for the subsystem you're changing
2. **TEST-READ**: Read existing tests — tests are the spec
3. **FACT-CHECK**: Does the spec match the code? Does the code match the tests?
4. **SPEC-CORRECT**: Fix spec inconsistencies before coding
5. **DESIGN** (if complex): New subsystem or mechanism with invariants? Write a design doc in `design/` BEFORE coding. Captures: invariants, alternatives considered, rationale. Skip for simple changes.
6. **TEST-WRITE (RED)**: Write a failing test that defines the new behavior. If you can't write the test, you don't understand the requirement — go back to SPEC-CORRECT. For refactors: skip (existing tests ARE the spec). For bug fixes: the reproducer IS the RED test.
7. **IMPLEMENT (GREEN)**: Write the code until the RED test passes
8. **TEST-RUN**: Run full test suite — `/test-unit` (all 24 unit tests must pass)
9. **REVIEW**: Invoke `/review recent` — +1 subagent + +2 external, multi-round until clean
10. **PERF-CONFIRM** (if performance-critical source changed): Run performance comparison against Python reference or internal baseline. If regression: investigate before committing.
11. **SPEC-UPDATE**: Update `spec/README.md` to reflect the change

### Principles

1. **Understanding > Speed** — Read and understand before changing
2. **Tests are the spec** — Don't weaken or remove existing tests. Write NEW tests for new behavior.
3. **No errors left behind** — ALL warnings, diagnostics, and test failures fixed
4. **Multi-round by default** — Reviews and audits loop until 0 new findings or 5 rounds
5. **Save everything** — Not written to file = never happened

## Skill Triggers

| Trigger | Invoke |
|---------|--------|
| Session start / compaction | `/resume` |
| Source files edited | `/test-unit` |
| Feature/milestone complete | `/review <scope>` |
| Docs may be stale | `/sync-docs [mode]` |
| Session start / milestone close | `/optimize-harness quick\|full` |

## REVIEW Protocol

**REVIEW = automatic after TEST-RUN.** Invoke `/review recent`. Multi-round until clean.

| Who | What | How |
|-----|------|-----|
| +1 (Self) | Self-review via subagent | Agent tool with fresh context + reviewer role (`agents/reviewer.md`) |
| +2 (External) | External code review | Spawn second subagent as +2 (fallback if no external tool) |

Both +1 and +2 must pass before proceeding.

## PERF-CONFIRM (performance-critical source)

After any change to `harmonyos_3dgs/src/cpu/rasterizer*.cpp`, `preprocessor*.cpp`, `sh_eval.cpp`, or their GPU counterparts:
1. Build and run full test suite (confirm no regressions)
2. Run a quick render comparison if a reference scene is available (e.g., `basket0.ply`)
3. If output deviates from Python reference: investigate before committing

When `/test-perf` skill is available, invoke it here. Until then: manual comparison.

## Project Planning

See `.claude/workflow/planning.md` for milestone lifecycle and closing checklist.

## Hook Enforcement

See `.claude/workflow/hooks-detail.md` for hook behavior documentation.

**Hook bypass (`core.hooksPath=/dev/null`) is a last resort, not a convenience.**
Legitimate bypass: stale state from a parked prior session. NOT legitimate: "it's just docs" / "small change" / "review is a false positive." Deliverable changes (CLAUDE.md, WORKFLOW.md, PROJECT.md, harness files) ALWAYS require `/review` before commit — no size exemption.

## Memory Sync Rule

When Tier 0/1 memory files are updated, keep repo and auto-memory in sync:
- **Auto-memory** (`~/.claude/projects/-home-robota-h00813233-Graph-aaags-claude/memory/`) is the source of truth
- Update auto-memory first, then sync edits to repo `memory/` for version control and other agents (reviewer, auditor)

Repo copy is always available for other agents and version control.
