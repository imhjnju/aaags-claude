# Session State

## Current Phase
SP-6: Training gaps closed; basketball 100-step PSNR validation in progress

## Test Counts
- Total passing: 243 / 243 (non-Basketball, as of SP-6 T4 completion, 2026-04-19)
- GPU tests: skipped when no device (expected)
- Basketball test: 100-step run in progress (SP-6 T5; ~250s expected)

## Milestones
| Milestone | Status | Sessions | Summary |
|-----------|--------|----------|---------|
| SP-0: CUDA golden infra | DONE | — | CPU reference + FD test harness |
| SP-1: Vulkan infra | DONE | S2 | VulkanContext/Buffer/Shader/Pipeline; TDD gate |
| SP-2: Vulkan forward pipeline | DONE | S2 | preprocess.comp + sort + rasterize.comp |
| SP-3: Vulkan backward pipeline | DONE | S3 | rasterize_backward.comp + preprocess_backward.comp |
| SP-4: Training integration | DONE | S4 | ForwardCache caching, GPU Adam skeleton (CpuAdam), VulkanTrainer, 216 tests |
| SP-5: GPU optimizer + hyperparams | DONE | S5 | GPU Adam kernel, LR+SH schedules, DSSIM, MCMC densification, basketball E2E smoke test, 229 tests |
| SP-6: Training gaps closed | DONE | S7 | T1-T5 done; 243/243 + basketball loss-decrease pass |
| M0: Foundation | IN PROGRESS | — | Interleaved with Vulkan migration |

## SP-6 Task Status
| Task | Status | Commit | Tests |
|------|--------|--------|-------|
| T1: VkTrainingConfig + train_utils | DONE | 78ab0d1 | 241/241 |
| T2: Regularization gradients | DONE | 63e54c9 | 241/241 |
| T3: Position noise injection | DONE | c66afc0 | 243/243 |
| T4: Spatial LR scale | DONE | 57ad86b | 243/243 |
| T5: Basketball 100-step loss-decrease | DONE | 4b036e7 | PASSED 250s |

## Python → C++ Gaps Closed (SP-6)
1. **Opacity reg**: `dL/d_raw_opacity += (0.01/N)*sig*(1-sig)` — after backward, before Adam upload
2. **Scale reg**: `dL/d_raw_scale += (0.01/N)*exp(raw_sc)` — per-component
3. **Position noise**: `Sigma @ N(0,1) * op_sigmoid(1-opacity) * noise_lr * pos_lr` — after Adam download
4. **Spatial LR**: `pos_lr = spatial_lr_scale * lr_schedule(...)` — default scale=1.0

## Latest Sessions

### S7 — 2026-04-19 (current)
- SP-6 T1-T4 complete via subagent-driven development
- 243 tests pass (non-basketball)
- Basketball 100-step PSNR > 10dB test running in background

### S6 — 2026-04-19
- SP-5 Task 6 (basketball E2E) completed and committed
- 229/229 tests pass
- Basketball 100-step background validation PASSED (248.7s, loss decreased)

### S5 — 2026-04-18 to 2026-04-19
- SP-5 Tasks 1-6 complete via subagent-driven development
- GPU Adam (adam_step.comp + VulkanAdam), LR schedule, SH warmup, DSSIM analytical gradient, MCMC densification, basketball smoke test
- Key perf finding: ~2.5 s/step on Tegra due to sync-per-dispatch; SP-7 will chain CBs

### S4 — 2026-04-18
- SP-4 all 8 tasks complete + extra cov2D/Part C bug fix
- 216/216 tests passing

### S3 — 2026-04-17 to 2026-04-18
- SP-3 complete: rasterize_backward.comp + preprocess_backward.comp
- All 208 SP-3 tests passing before SP-4

### S2 — 2026-04-17
- SP-1 complete (Vulkan infra) + SP-2 complete (forward pipeline)

### S1 — 2026-04-16
- Installed dev harness (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
