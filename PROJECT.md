# harmonyos_3dgs — Project Context

Shared context for all agents. Facts only — no behavioral directives.

## Project Identity
- **What**: C++ port of AAA-Gaussians (Anti-Aliased 3D Gaussian Splatting) for HarmonyOS — same visual quality, high FPS on target platform
- **Language**: C++17, CMake, optional OpenCL backend; Python reference in `AAA-Gaussians/`
- **Success**: Visual quality matches Python reference (PSNR/SSIM parity); high FPS on HarmonyOS target
- **Reference**: `AAA-Gaussians/` — Python+CUDA original implementation (Steiner et al., ICCV 2025)

## Project Paths

### Harness (reusable)
| Role | Path | Purpose |
|------|------|---------|
| Dev directives | `CLAUDE.md` | Standing directives + STOP gates |
| Workflow | `WORKFLOW.md` | Full implementation cycle |
| Shared context | `PROJECT.md` | This file |
| Skills | `.claude/commands/` | Executable workflows |
| Memory (Tier 0) | `memory/MEMORY.md` | Identity, status, pointers |
| Harness builder | `harness_builder/` | Portable package |

### Document Taxonomy

| Document | Path | Write when |
|----------|------|-----------|
| Behavioral spec | `spec/README.md` | Before implementing a subsystem |
| Design docs | `design/` | Before implementing complex mechanism |
| Architecture | `harmonyos_3dgs/ARCHITECTURE.md` | When module structure stabilizes |
| Test plan | `harmonyos_3dgs/tests/TEST_PLAN.md` | Before writing tests for each category |
| Captain's log | `dev_notes/captains_log.md` | Each session |
| Session state | `dev_notes/session_state.md` | Each session |
| Master plan | `dev_notes/master_plan/README.md` | Milestone planning |
| Gotchas | `memory/gotchas.md` | When bug patterns are root-caused |
| Investigations | `investigations/` | During debugging (evidence files) |
| Reference analysis | `AAA-Gaussians/` | Read-only reference; don't modify |

## Key Metrics
- **Tests**: 24 unit test files (see `harmonyos_3dgs/tests/TEST_PLAN.md`)
- **Milestones**: M0 (foundation — harness, build, test baseline)

## Code Conventions
- Build flag: `-ffp-contract=off` on ALL targets (floating-point determinism for cross-platform numerical matching)
- Cross-compile: `cmake -DCMAKE_TOOLCHAIN_FILE=<ohos_toolchain> -DCMAKE_CROSSCOMPILING=ON` disables tests, enables `-ffp-contract=off` on ARM
- Tests are the spec. Don't weaken existing tests — write NEW tests.
- Build: `cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build`
- Run tests: `cd harmonyos_3dgs && ctest --test-dir build --output-on-failure`
