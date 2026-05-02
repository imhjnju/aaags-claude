# Repo Wiki — harmonyos_3dgs Team Knowledge Base

> Internal team wiki for humans and agents working on `harmonyos_3dgs`.
>
> Purpose: make project knowledge discoverable, keep operational practice durable, and provide a lightweight self-evolution loop so the wiki improves as the project changes.

## Audience

This wiki is for a mixed human + agent team:

- New team members who need to understand the project quickly.
- Active developers working on Vulkan/CUDA parity, training, testing, and performance.
- Reviewers who need the workflow, validation gates, and known pitfalls.
- Agents that need a stable navigation layer before reading deeper specs, logs, and source files.

## How to Use This Wiki

Start here, then follow the page that matches your task:

| Need | Read |
|------|------|
| Understand the project and current direction | [Project Overview](00-project-overview.md) |
| Find the right file or knowledge source | [Repo Map](01-repo-map.md) |
| Onboard a new human or agent | [Onboarding](02-onboarding.md) |
| Make a code change safely | [Development Workflow](03-development-workflow.md) |
| Validate behavior, parity, and performance | [Testing and Validation](04-testing-and-validation.md) |
| Work on Vulkan/CUDA parity | [Vulkan/CUDA Parity](05-vulkan-cuda-parity.md) |
| Understand training end-to-end | [Training Pipeline](06-training-pipeline.md) |
| Debug a regression or parity gap | [Debugging Playbook](07-debugging-playbook.md) |
| Run team operations and handoffs | [Operational Rhythm](08-operational-rhythm.md) |
| Understand the knowledge system | [Knowledge System](09-knowledge-system.md) |
| Improve the wiki itself | [Self-Evolution](10-self-evolution.md) |

## Source-of-Truth Policy

This wiki is an index and operating guide, not the only source of truth.

| Topic | Source / Entry Point |
|-------|----------------------|
| Agent directives and STOP gates | `CLAUDE.md` is authoritative. |
| Project facts and taxonomy | `PROJECT.md` is authoritative. |
| Implementation workflow | `WORKFLOW.md` is authoritative. |
| Behavioral contracts | `spec/README.md` is authoritative. |
| Test inventory and acceptance criteria | `harmonyos_3dgs/tests/TEST_PLAN.md` is authoritative. |
| Current project state | `dev_notes/session_state.md` is authoritative for latest recorded baseline. |
| Session narrative and experiment history | `dev_notes/captains_log.md` is the durable narrative log. |
| Retrieval index and memory layers | Auto-memory is authoritative for T0/T1; repo `memory/MEMORY_INDEX.md` is the versioned discovery copy. |
| Known bug patterns | Auto-memory is authoritative for T1; repo `memory/gotchas.md` is the versioned synced copy. |
| Python/CUDA reference | `AAA-Gaussians/` is the behavioral reference implementation. |

When this wiki conflicts with an authoritative source, trust the authoritative source and update the wiki. For T0/T1 memory conflicts, follow [Knowledge System](09-knowledge-system.md): update auto-memory first, then sync the repo copy.

## Reader Test FAQ

These are the questions a fresh reader should be able to answer quickly:

| Question | Answer |
|----------|--------|
| I am new to the repo. What should I read first? | Start with this page, then [Onboarding](02-onboarding.md), `PROJECT.md`, `CLAUDE.md`, and `dev_notes/session_state.md`. |
| I need to fix a Vulkan/CUDA parity issue. Where do I start? | Read [Vulkan/CUDA Parity](05-vulkan-cuda-parity.md), [Debugging Playbook](07-debugging-playbook.md), relevant spec/tests, then the Python/CUDA reference. |
| I changed DSSIM/loss or hot-path training code. What must I validate? | Run targeted tests, full CTest, equivalence checks, and performance comparison; see [Testing and Validation](04-testing-and-validation.md) and [Training Pipeline](06-training-pipeline.md). |
| Where do I record a non-obvious bug pattern? | Update auto-memory T1 first, then sync repo `memory/gotchas.md` or a focused T1 memory file and add retrieval pointers if needed. |
| What is the source of truth for current test counts? | `harmonyos_3dgs/tests/TEST_PLAN.md` for inventory and `dev_notes/session_state.md` for latest verified baseline. |
| How does the wiki evolve over time? | Use [Self-Evolution](10-self-evolution.md): per-change hygiene, milestone consolidation, periodic reader audit. |
| Is this compatible with GitHub Wiki? | Repo pages are GitHub-renderable by design; a GitHub Wiki mirror should flatten/rewrite links as described below. |

## GitHub Wiki Mirror Compatibility

The files in this directory are intentionally plain Markdown and render correctly in the repo. A GitHub Wiki mirror is supported, but mirroring is a separate transformation step because GitHub Wiki commonly uses a flatter page model than this repo directory.

Mirror policy:

- `README.md` becomes the GitHub Wiki `Home` page.
- Top-level pages keep stable names and become individual wiki pages.
- Links such as `02-onboarding.md` should be rewritten to the corresponding wiki page slug if the mirror flattens files.
- Template files under `templates/` should either remain in a supported subdirectory mirror or be flattened into pages named `Template - <name>`.
- Avoid repo-only rendering features, generated tables, or links that require local tooling.
- The repo copy is authoritative for edits; the GitHub Wiki copy, if used, is a published mirror.

## Maintenance Model

This wiki evolves at three levels:

1. **Per change**: update affected wiki pages only when behavior, workflow, or validation practice changes.
2. **Per milestone**: run a wiki closeout using [Self-Evolution](10-self-evolution.md) and [Milestone Closeout Template](templates/milestone-closeout.md).
3. **Periodic audit**: reader-test the wiki with a fresh agent and remove stale or duplicated knowledge.

Use [Wiki Update Checklist](templates/wiki-update-checklist.md) whenever changing this directory.

## Status

This is the first scaffold. It should be treated as a durable navigation layer that points to existing project knowledge rather than duplicating every detail.
