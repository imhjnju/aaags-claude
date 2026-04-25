# Memory Retrieval Index

> Topic-organized pointers into all 5 tiers. Load this when you need to find
> what to read before a task. Each entry points to a file — don't duplicate content here.

## How to Use

1. Find your topic below
2. Load the files listed under that topic
3. Tier markers: **(T0)** identity, **(T1)** operational, **(T2)** semantic, **(T3)** episodic, **(T4)** reference

## Maintaining This Index

When you create or update a memory file:
1. Add/update its entry under the relevant topic(s) — one file can appear under multiple topics
2. For large files (200+ lines), add grep terms: `search: "keyword1", "keyword2"`
3. Keep entries under 150 chars — this is an index, not content
4. When retiring a memory file, remove its entries here too

---

## Project Status
- [MEMORY.md](MEMORY.md) **(T0)** — identity, current status, skill table
- [session_state.md](../dev_notes/session_state.md) **(T3)** — milestone tracking, latest sessions
- [captains_log.md](../dev_notes/captains_log.md) **(T3)** — session narrative (search, don't load fully)

## Bug Patterns & Gotchas
- [gotchas.md](gotchas.md) **(T1)** — known bug patterns, prevention rules
- [image_layout_chw.md](image_layout_chw.md) **(T1)** — unified CHW layout convention, component map

## Methodology & Process
- [lessons_learned.md](../dev_notes/lessons_learned.md) **(T3)** — methodology insights, pending promotion

## 3DGS Implementation — T1 operational + T2 semantic
- [spec/README.md](../spec/README.md) **(T2)** — behavioral spec per subsystem (preprocessor, rasterizer, sh_eval, etc.)
<!-- Add T1 operational files here as they emerge:
- [floating_point.md](floating_point.md) **(T1)** — fp-contract rules, cross-platform numerical matching
- [opencl_patterns.md](opencl_patterns.md) **(T1)** — OpenCL backend patterns -->

## Architecture & Design — T2 semantic
<!-- Uncomment when ARCHITECTURE.md is created:
- [ARCHITECTURE.md](../harmonyos_3dgs/ARCHITECTURE.md) **(T2)** — subsystem boundaries, module map -->
- [design/](../design/) **(T2)** — design docs (invariants, mechanisms, interfaces)

## Testing
- [TEST_PLAN.md](../harmonyos_3dgs/tests/TEST_PLAN.md) **(T2)** — test inventory, commands, acceptance criteria

## External Reference — T4 reference
- [AAA-Gaussians/](../AAA-Gaussians/) **(T4)** — Python+CUDA reference implementation (read-only)
<!-- Add investigation files here as they emerge:
- [investigations/](../investigations/) **(T3)** — debugging evidence files -->
