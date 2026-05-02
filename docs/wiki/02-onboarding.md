# Onboarding

## Goal

A new human or agent should be able to understand the project, recover current state, build, run tests, and choose the next safe action without relying on chat history.

## First 30 Minutes

1. Read [Project Overview](00-project-overview.md).
2. Read `PROJECT.md` for project facts and taxonomy.
3. Read `CLAUDE.md` for mandatory operating rules.
4. Read `dev_notes/session_state.md` first 80-120 lines for current phase and test baseline.
5. Read `memory/MEMORY_INDEX.md` to find topic-specific context.
6. Build and run tests only after understanding current branch state.

## Environment Check

Use the commands from `CLAUDE.md` / `WORKFLOW.md` as authority.

Typical native build:

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build
```

Typical full test run:

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure
```

Before interpreting results, compare expected test count with `harmonyos_3dgs/tests/TEST_PLAN.md` and `dev_notes/session_state.md`.

## Onboarding Checklist for Humans

- [ ] I understand the project goal: match reference first, optimize second.
- [ ] I know where behavior is specified: `spec/README.md`.
- [ ] I know where current status lives: `dev_notes/session_state.md`.
- [ ] I know where session history lives: `dev_notes/captains_log.md`.
- [ ] I know how to run the build and tests.
- [ ] I know not to weaken tests to pass.
- [ ] I know how to update docs/memory when knowledge changes.

## Onboarding Checklist for Agents

- [ ] Read `CLAUDE.md`, `PROJECT.md`, and relevant wiki page.
- [ ] If starting or continuing a session, invoke `/resume`; manually read `session_state.md` and `captains_log.md` only as supplemental context or fallback if the skill is unavailable.
- [ ] Load only relevant memory via `memory/MEMORY_INDEX.md`.
- [ ] For code changes, classify the work and follow spec-first workflow.
- [ ] Use subagents for broad exploration and independent review.
- [ ] Save durable knowledge in the correct layer before context is lost.

## Common First Tasks

| Task | Start Here |
|------|------------|
| Fix a bug | [Debugging Playbook](07-debugging-playbook.md) |
| Add/modify behavior | [Development Workflow](03-development-workflow.md) |
| Run parity comparison | [Vulkan/CUDA Parity](05-vulkan-cuda-parity.md) |
| Validate tests | [Testing and Validation](04-testing-and-validation.md) |
| Update knowledge | [Knowledge System](09-knowledge-system.md) |

## What Not to Do

- Do not treat old chat summaries as authoritative over files.
- Do not skip spec/test reading before editing unfamiliar code.
- Do not delete or overwrite unfamiliar artifacts without checking ownership.
- Do not commit generated raw data or large temporary artifacts.
- Do not bury important results only in terminal output.

## Handoff Standard

A handoff should include:

- Current branch and commit.
- Modified/untracked files.
- Commands run and results.
- What was changed and why.
- Open risks or non-blocking findings.
- Next recommended action.

Use `dev_notes/session_state.md` for durable project-level handoff and commit messages for code-level rationale. For milestone handoffs, use [Milestone Closeout Template](templates/milestone-closeout.md); for wiki changes, use [Wiki Update Checklist](templates/wiki-update-checklist.md).
