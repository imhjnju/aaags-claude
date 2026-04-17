# harmonyos_3dgs — Primary Dev Agent

**Soul: match reference exactly, then optimize.** Know which mode you're in.

- **Designing**: Explore freely, reason about trade-offs, sketch approaches. Question the design at transition points, not mid-exploration.
- **Implementing**: Question every assumption — especially your own. Report what you actually know vs what you're guessing. When uncertain, say so. "It's probably fine" is not engineering.

See `PROJECT.md` for facts. Full workflow: `WORKFLOW.md`.

**Permissions**: All repo-scoped commands pre-authorized. Pause for phase transitions only.

---

## STOP Gates — Read the gate file when triggered

| # | Trigger | Action | Gate File |
|---|---------|--------|-----------|
| 1 | Edit rasterizer / preprocessor / sh_eval / backward code | Research first | [research](.claude/gates/research.md) |
| 2 | Fix didn't work | Stop guessing, re-observe | [research](.claude/gates/research.md) |
| 3 | "Reference also fails" | Exhaust OUR bugs first | (inline) |
| 4 | Writing math/numerical fix | Cite evidence in comment | (inline) |
| 5 | Unfamiliar subsystem this session | Read spec section first | [proactive-review](.claude/gates/proactive-review.md) |
| 6 | Test code modified / GPU tests ran | Dual +1/+2 audit | [test-audit](.claude/gates/test-audit.md) |
| 7 | New external library | Spy API, audit, gap doc | [integration](.claude/gates/integration.md) |
| 8 | `git commit` | Pre-commit checklist | [commit](.claude/gates/commit.md) |

## Standing Directives

1. **Understanding > Speed** — Research before implementing. Don't guess at 3DGS math.
2. **Spec-First** — Read `spec/README.md` for the subsystem before coding. WHAT/WHY, never HOW.
3. **Save Everything** — Compaction erases conversation — unsaved = lost.
4. **Update = Rewrite** — Read, integrate, improve — never tack onto the end.
5. **Context Budget** — Analysis in main context. Large mechanical edits via subagent.
6. **Evidence, Not Reasoning** — Verify (run checks), observe before theorizing. One variable per experiment.
7. **Hypothesis Discipline** — Each retry must bring new data. After 3 failures: STOP, save state, fresh-eyes escalation.
8. **No Silent Deferrals** — Anything observed but not fixed leaves a visible TODO. No mark = forgotten.
9. **Tests Are the Specification** — Never modify a test to pass. Test infra is product infra.
10. **Review Everything** — +1 self-review + +2 external. The author is blind to own assumptions.

## Build

`-ffp-contract=off` (floating-point determinism — critical for cross-platform numerical matching). Build from `harmonyos_3dgs/`.

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build
```

## Workflow

**Full Cycle** (spec-driven): SPEC-READ → TEST-READ → FACT-CHECK → SPEC-CORRECT → [DESIGN] → TEST-WRITE (RED) → IMPLEMENT (GREEN) → TEST-RUN → REVIEW → [PERF-CONFIRM] → SPEC-UPDATE

- **No shortcuts**: Every code change follows the cycle. "Just mechanical" is not an exemption.
- **Design exploration** is exempt. Sketch freely, prototype without tests. When you pick an approach — enter the cycle.
- Tests are the spec. Don't weaken existing tests — write NEW tests.
- No errors left behind. ALL warnings and test failures fixed.
- Full workflow detail: `WORKFLOW.md`

## Skills — when to invoke

| Trigger | Invoke |
|---------|--------|
| Session start / compaction | `/resume` |
| After code changes | `/test-unit` |
| Feature complete | `/review <scope>` |
| Docs may be stale | `/sync-docs` |
| Harness feedback exists | `/optimize-harness quick\|full` |
| Milestone close | `/lint-knowledge full` |
| Sync methodology to builder | `/sync-harness` |

Hooks enforce debts. Commit gate blocks if unpaid.
