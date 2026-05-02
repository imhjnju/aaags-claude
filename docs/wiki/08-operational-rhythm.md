# Operational Rhythm

## Purpose

This page describes how the team keeps work understandable across humans, agents, sessions, commits, and milestones.

## Operating Cadence

| Moment | Action |
|--------|--------|
| Session start | Invoke `/resume`; use `dev_notes/session_state.md`, `captains_log.md`, and `memory/MEMORY_INDEX.md` as supplemental context, or as fallback only if `/resume` is unavailable. |
| Before implementation | Read spec/tests/reference and create or confirm the plan. |
| During work | Save durable findings, commands, metrics, and blockers. |
| After tests | Record validation results if they change the baseline. |
| Before commit | Review diff, exclude artifacts, ensure docs/state are updated. |
| Milestone close | Consolidate wiki, memory, session state, and captain's log. |

## Human + Agent Collaboration

Humans own judgment and priorities. Agents help by:

- Locating knowledge and code.
- Drafting tests and implementation.
- Running validation.
- Performing fresh-context review.
- Updating durable docs.
- Reader-testing wiki/doc changes.

Agents should not silently decide broad direction changes, destructive actions, or external-visible actions without appropriate confirmation.

## Session Records

Use:

- `dev_notes/session_state.md` for current status, counts, milestone state, next action.
- `dev_notes/captains_log.md` for narrative: what happened and why it matters.
- `memory/MEMORY.md` for future-session identity and important current baseline.
- Topic memory files for reusable non-obvious lessons.

## Review Rhythm

Review is expected after code changes and milestone-level documentation changes.

Minimum review questions:

- Does this match the relevant spec/reference?
- Are tests strong enough and not weakened?
- Did the change broaden scope unnecessarily?
- Are performance claims backed by data?
- Are durable docs updated?
- Are generated artifacts excluded?

## Milestone Closeout

At milestone close:

1. Confirm build/test baseline.
2. Summarize metrics and gates.
3. Update `session_state.md` and `captains_log.md`.
4. Update memory only for future-relevant non-obvious facts.
5. Update wiki if workflows, maps, or durable lessons changed.
6. Run reader-test on changed wiki pages if substantial.
7. Tag/commit/push only when requested and after checking status.

Use [Milestone Closeout Template](templates/milestone-closeout.md).

## Communication Style

Durable records should be:

- Evidence-backed.
- Specific enough to reproduce.
- Clear about authority and uncertainty.
- Short enough to stay maintainable.

Avoid:

- Duplicating large code explanations.
- Hiding uncertainty.
- Writing chronological noise into stable docs.
- Storing ephemeral task details in memory.
