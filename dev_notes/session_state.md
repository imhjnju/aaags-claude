# Session State

## Current Phase
VK eval_3D CUDA parity: PSNR 42.8 dB (target ≥60 dB). 17.2 dB gap remains.

## Test Counts
- Total passing: 243 / 243 + 1 VK-vs-CUDA basketball test (PSNR=42.8 dB, baseline=42.1 dB)
- As of 2026-04-21 (Session 8)

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
| VK-CUDA parity (PSNR≥60dB) | IN PROGRESS | S8 | 25.3→42.8 dB; proper_ewa, tile_culling, sub-tile sort done. Need persistent TAIL buffer |
| M0: Foundation | IN PROGRESS | — | Interleaved with Vulkan migration |

## VK-CUDA Parity Task Status (Session 8)
| Task | Status | PSNR Impact | Commit |
|------|--------|-------------|--------|
| Harness (render_single.py + gtest) | DONE | baseline=25.3 | a4dc240, fa6adc9 |
| proper_ewa_scaling (eval_3D + 2D) | DONE | +11.6 dB | ed303bd |
| rect_bounding + tight_opacity_bounding | DONE | 0 dB | 6ff257b |
| tile_based_culling (INVALID sentinel) | DONE | +3.4 dB | 83ca5b1 |
| Hierarchical sub-tile TAIL re-sort | DONE | +2.3 dB | b4cd239 |
| HEAD_W=8, subtile_cx+2.0, z/w key | DONE | +0.2 dB | 72bd4bf |
| Persistent cross-batch TAIL buffer | TODO | ? | — |
| Verify ≥60 dB + lock baseline | TODO | — | — |

## Python → C++ Gaps Closed (SP-6)
1. **Opacity reg**: `dL/d_raw_opacity += (0.01/N)*sig*(1-sig)` — after backward, before Adam upload
2. **Scale reg**: `dL/d_raw_scale += (0.01/N)*exp(raw_sc)` — per-component
3. **Position noise**: `Sigma @ N(0,1) * op_sigmoid(1-opacity) * noise_lr * pos_lr` — after Adam download
4. **Spatial LR**: `pos_lr = spatial_lr_scale * lr_schedule(...)` — default scale=1.0

## Latest Sessions

### S8 — 2026-04-20 (current)
- SP-7 T1-T5 complete (CB chaining, persistent buffers): 248 tests pass
- Basketball test: changed to 100 steps (no densification) PSNR=5.61 dB, finite+positive assertion
- VK vs Python gradient comparison: 3 new tests implemented (task 54-56)
  - `VkVsPyReference.Step1GradientAndLoss`: loss rel_diff=3.6e-7, all grad norms 0.0000 diff
  - `VkVsPyReference.ConvergenceTable100Steps`: both converge 0.042→0.002 (100 steps)
- **BUG FIXED**: rasterize.comp writes CHW but loss/backward expect HWC — fixed in vulkan_trainer.cpp
  - Gradient norms were 45-60% below Python reference before fix
  - See gotchas.md for full details
- Python reference dump: tools/dump_tiny_reference.py + tests/golden/tiny/py_ref/ (1000 files)
- VulkanTrainer gradient capture: enable_gradient_capture() + captured_grad_*() accessors

### S7 — 2026-04-19
- SP-6 T1-T4 complete via subagent-driven development
- 243 tests pass (non-basketball)
- Basketball 100-step loss-decrease validated

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
