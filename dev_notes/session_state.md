# Session State

## Current Phase
SP-5: GPU Optimizer + Training Hyperparameters

## Test Counts
- Total passing: 216 / 216 (as of SP-4 completion, 2026-04-18)
- GPU tests: skipped when no device (expected)

## Milestones
| Milestone | Status | Sessions | Summary |
|-----------|--------|----------|---------|
| SP-0: CUDA golden infra | DONE | — | CPU reference + FD test harness |
| SP-1: Vulkan infra | DONE | S2 | VulkanContext/Buffer/Shader/Pipeline; TDD gate |
| SP-2: Vulkan forward pipeline | DONE | S2 | preprocess.comp + sort + rasterize.comp |
| SP-3: Vulkan backward pipeline | DONE | S3 | rasterize_backward.comp + preprocess_backward.comp |
| SP-4: Training integration | DONE | S4 | ForwardCache caching, GPU Adam skeleton (CpuAdam), VulkanTrainer, 216 tests |
| SP-5: GPU optimizer + hyperparams | IN PROGRESS | S5 | GPU Adam, LR schedule, DSSIM, MCMC densification, 2000-step validation |
| M0: Foundation | IN PROGRESS | — | Interleaved with Vulkan migration |

## Latest Sessions

### S5 — 2026-04-18 (current)
- Writing SP-5 plan
- Next: execute SP-5 tasks via subagent-driven development

### S4 — 2026-04-18
- SP-4 all 8 tasks complete + extra cov2D/Part C bug fix
- Added: count-based T_MIN guard, d_raw_opacities (sigmoid backward), quaternion normalization Jacobian, means2D_cache backward, cov2D/det caching to eliminate Part A recompute
- Fixed: missing SH→position gradient chain in Part C (was primary cause of ~0.82 abs error)
- CpuAdam + VulkanTrainer: full forward→backward→Adam training loop working
- 216/216 tests passing

### S3 — 2026-04-17 to 2026-04-18
- SP-3 complete: rasterize_backward.comp + preprocess_backward.comp
- All 208 SP-3 tests passing before SP-4

### S2 — 2026-04-17
- SP-1 complete (Vulkan infra) + SP-2 complete (forward pipeline)
- Preprocess + sort + rasterize working on GPU, CPU-matched

### S1 — 2026-04-16
- Installed dev harness (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
