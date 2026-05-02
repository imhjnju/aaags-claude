# Repo Map

## Core Project Files

| Path | Role |
|------|------|
| `CLAUDE.md` | Primary agent directives, STOP gates, and standing rules. |
| `PROJECT.md` | Shared project facts, document taxonomy, key paths. |
| `WORKFLOW.md` | Required implementation cycle and review/test discipline. |
| `spec/README.md` | Behavioral specification for subsystem correctness. |
| `harmonyos_3dgs/` | Main C++17/CMake codebase. |
| `AAA-Gaussians/` | Python/CUDA reference implementation. Read-only for parity work unless explicitly maintaining reference tooling. |
| `memory/` | Versioned memory and retrieval index for project knowledge. |
| `dev_notes/` | Current state, session history, master plans, investigations. |
| `docs/` | Documentation archive and this wiki. |

## Code Layout

| Path | Use |
|------|-----|
| `harmonyos_3dgs/src/` | Core implementation. |
| `harmonyos_3dgs/include/` | Public/internal headers. |
| `harmonyos_3dgs/src/vulkan/` | Vulkan host-side implementation and CLI tools. |
| `harmonyos_3dgs/src/vulkan/shaders/` | Vulkan compute shaders. |
| `harmonyos_3dgs/tests/` | CTest/GTest tests and golden fixtures. |
| `harmonyos_3dgs/tools/` | Comparison harnesses, dumpers, diagnostics, scripts. |

## Knowledge Layers

| Layer | Path | Purpose |
|-------|------|---------|
| T0 identity/status | `memory/MEMORY.md` | Always-loaded project identity and latest baseline. |
| T1 operational gotchas | `memory/gotchas.md`, topic files | Durable prevention rules and non-obvious traps. |
| T2 semantic specs | `spec/`, design docs | WHAT/WHY behavioral knowledge. |
| T3 episodic history | `dev_notes/`, `investigations/` | What happened, what was tried, experiment records. |
| T4 external reference | `AAA-Gaussians/` | Ground-truth Python/CUDA behavior. |
| Wiki navigation | `docs/wiki/` | Human/agent readable map and operating handbook. |

## Where to Put New Knowledge

| Knowledge Type | Put It Here |
|----------------|-------------|
| Behavioral contract | `spec/README.md` or a subsystem spec. |
| Complex design decision | Create or update a design doc in `design/` before coding; use dev notes only for supporting investigation history. |
| Current project status | `dev_notes/session_state.md` and `memory/MEMORY.md` if future sessions need it. |
| Session narrative | `dev_notes/captains_log.md`. |
| Reusable gotcha | `memory/gotchas.md` or a focused T1 memory file. |
| Debug evidence | `investigations/` or `dev_notes/` with artifact paths. |
| Team operating rule | `WORKFLOW.md`, `CLAUDE.md`, and this wiki if reader-facing. |
| Wiki navigation or process | `docs/wiki/`. |

## Known Structural Gaps

These are intentionally visible so the team can close them deliberately:

| Gap | Status | Reader Action |
|-----|--------|---------------|
| `design/` may not yet contain a complete design set. | Harmless unless starting a complex new mechanism. | If a change has meaningful alternatives or invariants, create/update a design note before coding. |
| `harmonyos_3dgs/ARCHITECTURE.md` is referenced as a future stable architecture doc. | Not a blocker for current work. | Use this wiki, `PROJECT.md`, source layout, and specs until architecture stabilizes. |
| Older docs in `docs/` and `harmonyos_3dgs/docs/` are historical snapshots. | Potentially stale. | Verify against current source, tests, `session_state.md`, and recent commits before acting. |

## Rule for Readers

If you are unsure whether a document is current, prefer this order:

1. Tests and source code.
2. `spec/README.md` for intended behavior.
3. `dev_notes/session_state.md` for current status.
4. `memory/MEMORY_INDEX.md` to find relevant memory.
5. Historical docs only after checking dates and current code.
