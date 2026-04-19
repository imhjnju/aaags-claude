# Captain's Log

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
