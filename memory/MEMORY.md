# harmonyos_3dgs — Agent Memory

> Tier 0: Identity + pointers. Always loaded at session start.

## Project Identity
C++ port of AAA-Gaussians (Anti-Aliased 3DGS) for HarmonyOS. C++17 + CMake + OpenCL. M0 in progress.

## Current Status
- **Milestone**: M0 Foundation — IN PROGRESS
- **Tests**: 24 unit test files (see `harmonyos_3dgs/tests/TEST_PLAN.md`). Run: `ctest --test-dir harmonyos_3dgs/build`

## Memory Architecture (5-Tier)

| Tier | Role | Location | Loading |
|------|------|----------|---------|
| **T0: Identity** | Who am I, what's the status | `memory/` | Every session start |
| **T1: Operational** | Bug patterns, validated rules | `memory/` | On demand via retrieval cues |
| **T2: Semantic** | Behavioral knowledge — WHAT/WHY | `spec/` | When working on a subsystem |
| **T3: Episodic** | What happened, what was tried | `dev_notes/`, `investigations/` | Debugging, investigation, resume |
| **T4: Reference** | How the Python original does it | `AAA-Gaussians/` | Via research |

### Fixed Files (created at init)

| File | Tier | Location | Purpose |
|------|------|----------|---------|
| `MEMORY.md` | T0 | `memory/` | This file — identity, status, pointers |
| `MEMORY_INDEX.md` | T0 | `memory/` | Topic-organized retrieval index into all tiers |
| `gotchas.md` | T1 | `memory/` | Bug prevention patterns |
| `captains_log.md` | T3 | `dev_notes/` | Session narrative — what happened, key decisions |
| `session_state.md` | T3 | `dev_notes/` | Milestone tracking — current phase, test counts |

### Emergent Files
T1 topic files (e.g., `floating_point.md`, `opencl_patterns.md`) are NOT pre-created.
The agent creates them when it discovers knowledge that is (1) needed repeatedly,
(2) not obvious from the code, (3) hard to rediscover.

**How to create**: Choose a descriptive name, write to `memory/`, add to §Tier 1 Files table,
add retrieval cues to MEMORY_INDEX.md. Write to auto-memory first, then sync to repo.

## Tier 1 Files

| File | Contains | When to Load |
|------|----------|-------------|
| `gotchas.md` | Bug prevention patterns | Before any code change |
<!-- Add rows as T1 files emerge -->

## Retrieval
**All retrieval cues are in MEMORY_INDEX.md** — topic-organized, cross-tier.
Load it when you need to find what to read before a task.

## Skills & Tools

| Skill | When | What |
|-------|------|------|
| `/resume` | Session start, compaction | Load state, confirm build, set direction |
| `/test-unit` | After code changes | Build + run 24 unit tests |
| `/review` | Feature complete | Code review (+1 self, +2 external) |
| `/sync-docs` | Docs may be stale | Code → docs consistency |
| `/lint-knowledge` | Milestone close | Knowledge consistency audit |
| `/optimize-harness` | Lessons exist or milestone close | Harness self-optimization |
| `/sync-harness` | Push improvements to builder | Bidirectional harness sync |
<!-- Add rows as skills are created -->

**Rule**: If a skill exists for your task, USE IT. Don't reconstruct steps from memory.

## User Preferences
<!-- Populated as preferences are discovered during sessions -->
