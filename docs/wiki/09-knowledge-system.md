# Knowledge System

## Purpose

The project uses multiple knowledge layers because different information decays at different speeds. This page explains where knowledge should live so it remains useful without becoming stale.

## Five-Tier Model

| Tier | Path | Role | Update When |
|------|------|------|-------------|
| T0 | auto-memory, synced to `memory/MEMORY.md` | Always-loaded identity, latest baseline, pointers. | Current baseline or next action changes in a future-relevant way. |
| T1 | auto-memory, synced to `memory/*.md` | Reusable gotchas and operational rules. | A non-obvious bug pattern or validated practice is discovered. |
| T2 | `spec/`, `design/` | Semantic WHAT/WHY contracts and design invariants. | Behavior or architecture changes. |
| T3 | `dev_notes/`, `investigations/` | Episodic history, experiments, session records. | During sessions, investigations, and milestone closeout. |
| T4 | `AAA-Gaussians/` | External reference implementation. | Read-only reference for parity. |
| Wiki | `docs/wiki/` | Navigation and team operating handbook. | When workflows, maps, or durable reader guidance changes. |

## What Belongs in Wiki

Put content in this wiki when it answers:

- Where should I start?
- Which source is authoritative?
- What process should I follow?
- What durable lesson should future readers know?
- How do humans and agents coordinate around this repo?

Do not put raw experiment dumps, temporary task notes, or code-level details here unless they are stable operational knowledge.

## Auto-Memory vs Repo Memory

For T0/T1 memory, the auto-memory directory under the Claude project state is the operational source of truth. The repo `memory/` directory is the versioned, reviewable copy used by humans, subagents, and future checkouts.

When memory changes are durable and project-relevant:

1. Update auto-memory first.
2. Sync the corresponding repo `memory/` file.
3. Update `memory/MEMORY_INDEX.md` if retrieval cues changed.
4. Link from wiki only when the lesson changes team navigation or process.

## Memory vs Wiki

| Question | Use Memory | Use Wiki |
|----------|------------|----------|
| Should future agents remember this automatically? | Yes | Maybe link it. |
| Is this a team-facing process or handbook page? | Maybe | Yes |
| Is this a current baseline that changes often? | T0 memory + session state | Link only |
| Is this a reusable gotcha? | T1 memory | Summarize if broadly important |
| Is this a how-to navigation page? | No | Yes |

## Dev Notes vs Wiki

Use `dev_notes/` for chronological or milestone-specific facts. Promote to wiki only when the lesson is durable and broadly useful.

Promotion examples:

- A one-off command result stays in `captains_log.md`.
- A reusable parity comparison pattern belongs in wiki.
- A detailed incident belongs in `dev_notes/` or `investigations/`; its prevention rule belongs in `memory/gotchas.md` and may be linked from wiki.

## Index Maintenance

When adding a new durable knowledge file:

- Add it to `memory/MEMORY_INDEX.md` if it should be discoverable by topic.
- Add or update a wiki link if it changes team navigation.
- Avoid duplicating full content across layers.

## Staleness Management

Any page that names specific metrics, test counts, paths, or commands must either:

- Point to the living source of truth, or
- State the date/context and be updated during milestone closeout.

Stable wiki pages should prefer structure and process over volatile numbers.

## Reader Rule

If a reader cannot answer “where should this knowledge live?” after reading this page, update the examples table rather than relying on tribal knowledge.
