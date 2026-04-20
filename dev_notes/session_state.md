# Session State

## Current Phase
Vulkan Migration — SP-4 under review (branch: sp4-training-integration)

## Test Counts
- Total ctest entries: 187 — verified 2026-04-18 (pre-SP2/SP3/SP4)
- SP-4 branch adds: test_forward_cache_vk, test_cpu_adam, test_training_step_vk + expanded backward tests
- Estimated total on sp4 branch: ~220+ (not yet verified post-merge)

## Milestones
| Milestone | Status | Sessions | Summary |
|-----------|--------|----------|---------|
| M0: Foundation | IN PROGRESS | S1-S2 | Harness installed; build verified; 187 tests passing |
| SP-1: Vulkan Infra | COMPLETE | S2 | VulkanContext, Buffer, Shader, ComputePipeline; TDD gate 2 tests pass |
| SP-2: Forward Pipeline | COMPLETE | - | preprocess.comp + PreprocessorVulkan + TileBinner + Sorter + Rasterizer; merged to master |
| SP-3: Backward Pipeline | COMPLETE | - | preprocess_backward.comp + rasterize_backward.comp + integration test T24; merged to master |
| SP-4: Training Integration | IN REVIEW | S3 | VulkanTrainer + CpuAdam + cov2D cache; REWORK on CHW/HWC bug before merge |
| SP-5: GPU-native backward | PLANNED | - | Layer-2 cache buffer getters; GPU-resident full training loop |

## Pending Before SP-4 Merge
- [ ] **BLOCKING** Fix CHW vs HWC layout in `vulkan_trainer.cpp` L1 loss loop
- [ ] Fix `vk_buffer.cpp` null-guard for `vkDeviceWaitIdle`
- [ ] Fix 5 items from Tasks 1–4 review (see captains_log.md S3)
- [ ] Add `d_raw_opacities` integration assertion to `test_backward_pipeline_vk.cpp`
- [ ] Fix `ConvergesToMinimum` test to call `CpuAdam::step()`
- [ ] Add position gradient assertion to `test_training_step_vk.cpp`

## Latest Sessions

### S4 — 2026-04-20
- Implemented `tools/render_single.py`: AAA-GS render for a single camera pose
- Loads basket-aaa.ply (400k Gaussians, sh_degree=3, filter_3D) → renders camera ID 0 from cameras.json
- All AAA features enabled via configs/aaa.json; runtime: `conda run -n aaa-gs`
- Merged worktree-render → master

### S3 — 2026-04-20
- Dual-reviewer code review of SP-4 (Tasks 1–4 and Tasks 5–8)
- Found 1 blocking bug: CHW vs HWC image layout in VulkanTrainer
- Found 5 fix-now items in Tasks 1–4 (stale comments, test gaps, UB in test)
- Verified: Adam formula, cov2D cache, gradient zeroing, ForwardCache wiring, SH backward math
- See captains_log.md for full finding list

### S2 — 2026-04-18
- Verified build baseline: OK (all targets compile clean)
- Verified test baseline: 187 tests, 100% pass
- Hooks verified: all 3 OK
- SP-1 already merged; SP-2 plan committed

### S1 — 2026-04-16
- Installed dev harness (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
