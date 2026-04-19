# Captain's Log

## Session 6 — 2026-04-19

SP-5 complete. 229/229 tests pass. Basketball E2E smoke test runs in ~7.7 s (3 steps).

Key perf finding: ~2.5 s/step on NVIDIA Tegra Thor at 720×960 due to sync-per-dispatch
architecture (~25 vkDeviceWaitIdle calls per step). This means:
- 50-step test: ~125 s
- 2000-step test: ~83 min (infeasible for CI)

SP-6 primary goal: eliminate per-dispatch syncs via command buffer chaining.
After SP-6, the 2000-step PSNR>10dB basketball milestone can be re-enabled.

SP-5 technical decisions:
- VulkanAdam: 4 SSBOs (params, grads, m, v) + UBO (AdamStepUBO) per group; 6 groups in VulkanTrainer
- DSSIM: correct analytical sliding-window gradient (center-window FD approximation was O(1/121))
- MCMC densification: separate output OwnedRawParams prevents pointer invalidation during realloc
- reallocate_for_n uses max(sz, 4) guard for all buffer sizes (prevents VUID-VkBufferCreateInfo-size-00912)
- densify_from_step=0 is the disable sentinel (documented in VkTrainingConfig)
- lambda_dssim=0.0f added to VkTrainingConfig for L1-only mode

---

## Session 5 — 2026-04-18

Starting SP-5: GPU Optimizer + Training Hyperparameters.

SP-4 landed with 216/216 tests passing. CpuAdam is the current optimizer — temporary stepping stone.
User directive: ultimate goal is ALL Vulkan/GPU execution; CpuAdam must be replaced.

SP-5 plan being written. Priority order:
1. GPU Adam (adam_step.comp + VulkanAdam) — top explicit user priority
2. LR schedule + SH degree schedule — needed for 2000-step convergence
3. DSSIM loss — match Python reference loss function
4. MCMC densification — core AAA-Gaussians training strategy
5. 2000-step basketball dataset validation — milestone completion

---

## Session 4 — 2026-04-18

SP-4: Training Integration. All 8 tasks + extra cov2D fix completed.

Key decisions:
- count-based n_contrib guard in rasterize_backward.comp (not position-based)
- Quaternion normalization Jacobian: divide by |q_raw| after optimizer steps
- ndc2Pix inverse: `ndc = (2*pixel+1)/S - 1` (not the plan's wrong formula)
- cov2D + cov2D_det cached from forward preprocess.comp (bindings 17,18)
  → eliminates ~0.82 abs error in Part A from recompute
- Missing SH→position gradient chain in Part C was primary 0.82 error source
- vkDeviceWaitIdle before VulkanBuffer destruction + VulkanContext::release()
- ForwardCache: pre-allocate T_final/n_contrib before rasterize() call
- CpuAdam: set_grad() per step because FrameAllocator resets grad pointers

216/216 tests passing. VulkanTrainer integrated, loss decreases over steps.

---

## Session 3 — 2026-04-17 to 2026-04-18

SP-3: Vulkan backward pipeline.

rasterize_backward.comp: per-tile back-to-front alpha-blend gradient pass.
preprocess_backward.comp: gradient chains for cov3D, scales/rotations, SH, positions, opacities.

All tests to 208 before SP-4.

---

## Session 2 — 2026-04-17

SP-1: Vulkan infrastructure (VulkanContext, Buffer, Shader, Pipeline, TDD gate).
SP-2: Vulkan forward pipeline (preprocess.comp, tile binner, radix sort, rasterize.comp).

Forward pass GPU-matched to CPU reference.

---

## Session 1 — 2026-04-16

Project harness initialized for harmonyos_3dgs.

Goals:
- [x] M0 kickoff: dev harness installed (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
- [x] M0: Build verified
- [x] M0: Test baseline verified
