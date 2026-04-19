# Session State

## Current Phase
SP-6: Command buffer chaining + convergence validation (next)

## Test Counts
- Total passing: 229 / 229 (as of SP-5 completion, 2026-04-19)
- GPU tests: skipped when no device (expected)

## Milestones
| Milestone | Status | Sessions | Summary |
|-----------|--------|----------|---------|
| SP-0: CUDA golden infra | DONE | — | CPU reference + FD test harness |
| SP-1: Vulkan infra | DONE | S2 | VulkanContext/Buffer/Shader/Pipeline; TDD gate |
| SP-2: Vulkan forward pipeline | DONE | S2 | preprocess.comp + sort + rasterize.comp |
| SP-3: Vulkan backward pipeline | DONE | S3 | rasterize_backward.comp + preprocess_backward.comp |
| SP-4: Training integration | DONE | S4 | ForwardCache caching, GPU Adam skeleton (CpuAdam), VulkanTrainer, 216 tests |
| SP-5: GPU optimizer + hyperparams | DONE | S5 | GPU Adam kernel, LR+SH schedules, DSSIM, MCMC densification, basketball E2E smoke test, 229 tests |
| SP-6: CB chaining + convergence | PENDING | — | Eliminate sync-per-dispatch, enable 2000-step PSNR>10dB test |
| M0: Foundation | IN PROGRESS | — | Interleaved with Vulkan migration |

## Latest Sessions

### S6 — 2026-04-19 (current)
- SP-5 Task 6 (basketball E2E) completed and committed
- 229/229 tests pass
- Next: SP-6 command buffer chaining

### S5 — 2026-04-18 to 2026-04-19
- SP-5 Tasks 1-6 complete via subagent-driven development
- GPU Adam (adam_step.comp + VulkanAdam), LR schedule, SH warmup, DSSIM analytical gradient, MCMC densification, basketball smoke test
- Key perf finding: ~2.5 s/step on Tegra due to sync-per-dispatch; SP-6 will chain CBs

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
