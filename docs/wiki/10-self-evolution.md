# Self-Evolution

## Goal

The wiki should improve as the project learns, without becoming a stale duplicate of specs, logs, or memory. Self-evolution means every significant change leaves the knowledge system easier for the next reader.

## Three-Layer Evolution Model

### Layer 1: Per-Change Hygiene

Triggered by code, workflow, or validation changes.

Checklist:

- [ ] Did behavior change? Update spec/tests and link from wiki if reader-facing.
- [ ] Did validation practice change? Update testing wiki and `TEST_PLAN.md` if needed.
- [ ] Did workflow change? Update `WORKFLOW.md` / `CLAUDE.md` and relevant wiki page.
- [ ] Did a reusable gotcha emerge? Update memory and optionally link from wiki.
- [ ] Did current baseline change? Update `session_state.md` and possibly T0 memory.

### Layer 2: Milestone Consolidation

Triggered when a milestone or major feature set closes.

Checklist:

- [ ] Summarize what changed and what is now stable.
- [ ] Move reusable lessons from logs into wiki/memory/spec as appropriate.
- [ ] Remove or mark stale guidance.
- [ ] Update current-status pointers.
- [ ] Run reader-test on changed wiki pages.
- [ ] If the user explicitly requests tag/commit/push: complete review, check status/diff, exclude artifacts, then perform the requested external-visible action.

### Layer 3: Periodic Knowledge Audit

Triggered periodically or when onboarding/recovery feels slow.

Checklist:

- [ ] Fresh reader can find project status.
- [ ] Fresh reader can identify authoritative source for behavior.
- [ ] Fresh reader can run build/tests.
- [ ] Fresh reader can explain the current parity strategy.
- [ ] Wiki links resolve.
- [ ] No page duplicates volatile data unnecessarily.
- [ ] Old docs are marked historical or linked with caution.

## Reader Testing Protocol

Use fresh-context agents or humans to test the wiki.

Example reader questions:

1. I am new to the repo. What should I read first?
2. I need to fix a Vulkan/CUDA parity issue. Where do I start?
3. I changed DSSIM loss code. What must I validate?
4. Where do I record a non-obvious bug pattern?
5. What is the source of truth for current test counts?
6. How do I know whether an old doc is still authoritative?
7. What should be updated at milestone close?

If the reader gives wrong or incomplete answers, update the wiki page that should have made the answer obvious.

## Wiki Quality Bar

A good wiki page:

- Routes readers to the right source quickly.
- States authority and staleness rules.
- Avoids copying long implementation details.
- Provides checklists for repeated workflows.
- Is useful to both humans and agents.
- Can be tested by asking a fresh reader realistic questions.

A bad wiki page:

- Duplicates volatile state.
- Hides uncertainty.
- Mixes chronological logs with stable guidance.
- Has links but no decision guidance.
- Becomes a dumping ground for everything.

## Update Flow

1. Identify changed knowledge.
2. Choose the right layer using [Knowledge System](09-knowledge-system.md).
3. Edit the smallest necessary wiki pages.
4. Run link/readability check.
5. Reader-test if substantial.
6. Record the update in commit or session notes.

Use [Wiki Update Checklist](templates/wiki-update-checklist.md).
