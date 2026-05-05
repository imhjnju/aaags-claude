# harmonyos_3dgs — Agent Memory

> Tier 0: Identity + pointers. Always loaded at session start.

## Project Identity
C++ port of AAA-Gaussians (Anti-Aliased 3DGS) for HarmonyOS. C++17 + CMake + **Vulkan compute** (OpenCL is legacy — being replaced). Target: Maleoon 920 GPU on HarmonyOS NEXT, Vulkan 1.3 Roadmap 2022. Dev machine: NVIDIA Tegra Thor, Vulkan 1.4. See `dev_notes/master_plan/vulkan_migration_plan.md`.

## Current Status
- **Active work (2026-05-01, worktree-training)**: Vulkan end-to-end training gap closure now includes AAA-Gaussians MCMC densification, `filter_3D` propagation, saved `filter_3D` PLY output, CLI densification controls, CUDA-matched terminal no-update training, saved-PLY render-config parity, full-dataset/multi-view parity control, 1000-step feature-matrix coverage for DSSIM/position LR decay/SH warmup, optimized DSSIM loss/gradient runtime, and first-loss Gate_P2 eval_3D opacity parity.
- **Latest verified baseline**: Build OK; full CTest **315/315 PASS** after final-step no-update mode, train/render `--proper_ewa`, render `--sh_degree`, MCMC densification, VulkanAdam state-preserving resize/zeroing, opacity reset, `filter_3D` propagation, PLY filter save, densification CLI exposure, full-dataset view schedules, feature-matrix CLI knobs, DSSIM slow-reference equivalence coverage, and Gate_P2 eval_3D opacity parity. Full-dataset feature matrix 1000-step runs all PASSed; `VkVsCudaFirstLoss` is PASS through P2 with P3-P7/L1 still skipped.
- **MCMC densification status**: `VkTrainingConfig::cap_max > 0` enables MCMC relocate/add densification; `cap_max <= 0` keeps legacy clone/split/prune. Untouched Adam slots preserve moments; MCMC-modified source/destination slots are zeroed across all six parameter groups. Opacity reset writes raw `logit(0.01)` and zeros opacity Adam moments.
- **CLI densification/status**: `gs3d_vk_train` defaults remain no densification. `--densify 1` enables a schedule and defaults to MCMC growth if `--cap_max` is omitted; `--cap_max N` with `N > 0` also enables MCMC; `--densify 1 --cap_max 0` selects legacy densification. CLI parsing rejects unknown flags, missing values, non-finite floats, invalid booleans, invalid dimensions/FOV, and unsafe densification ranges.
- **Full-dataset training status**: `gs3d_vk_train --view_schedule <path>` consumes an explicit 0-based per-iteration view sequence so CUDA and Vulkan can train on the exact same cameras. Use `--require_all_gt 1` for parity harnesses to fail on missing/empty GT instead of silently shifting the view set. Mixed-resolution multi-view training is rejected for now because trainer image buffers are sized from the first view.
- **eval_3D status**: Vulkan eval_3D training is enabled only for `parity_mode=true`; non-parity training fails explicitly until default HEAD/sub-tile replay is implemented. `eval_3D + MCMC` is supported by propagating `filter_3D` through `OwnedRawParams`, MCMC relocate/add, and VulkanTrainer state reallocation; saved PLYs include `filter_3D` when the input has it or `--eval_3d 1` is used.
- **Saved-PLY render status**: For 2D training parity, standalone `gs3d_vk_render` must pass `--proper_ewa 0` to match `VulkanTrainer`'s current `proper_ewa=false` default. The previous basketball white-block/distortion artifact was a render-mode mismatch, not PLY serialization: same saved PLY re-rendered with `--proper_ewa 0` matches VK training-final render at **74.68 dB**.
- **eval_3D/proper_ewa MCMC status**: Basketball cam0 5000-step eval_3D+proper_ewa MCMC with `cap_max=30000` matched final counts **24680/24680**; metrics: CUDA vs GT **27.13 dB**, VK saved vs GT **27.25 dB**, VK saved vs CUDA **29.37 dB**, VK saved vs train **59.04 dB**. Full-dataset all-76-view 1000-step eval_3D+proper_ewa MCMC with shared schedule matched final counts **3513/3513**; metrics: CUDA vs GT **17.22 dB**, VK saved vs GT **16.92 dB**, VK saved vs CUDA **27.99 dB**, VK saved vs train-final **59.24 dB**, GT gap **0.30 dB**, gate PASS. Full runner: `harmonyos_3dgs/tools/compare_basketball_eval3d_proper_mcmc_full.py`.
- **Feature-matrix status**: Full-dataset 1000-step basketball feature matrix with `cap_max=100000`, eval_3D/proper_ewa/MCMC, and no opacity reset PASSed baseline (**28.71 dB VK saved vs CUDA**), DSSIM 0.2 originally (**29.30 dB**), LR decay (**30.71 dB**), and SH warmup 1000 (**30.91 dB**, render SH degree 0). All matched final counts **3513/3513**. DSSIM VK training bottleneck was host-side direct 11x11 sliding-window SSIM/gradient work; optimized separable/parallel DSSIM reduced the 1000-step VK DSSIM train time **2632.7s → 167.6s** while PASSing (`VK saved vs CUDA 28.25 dB`, counts **3513/3513**).
- **scan1 2D parity milestone**: VK↔CUDA final-render PSNR improved to 107.14 dB at 10 steps, 49.82 dB at 200 steps, and 27.77 dB at 1000 steps after the `n_contrib` replay-boundary fix.
- **First-loss gate status**: `Gate_P2_ConicOpacity` is open and PASSing for eval_3D opacity-lane parity on overlap-active Gaussians. CUDA's eval_3D golden stores opacity as flat `((float*)conic_opacity)[gid]`; row-wise `{a,b,c,opacity}` interpretation is invalid for this fixture. Active-set mismatches remain visible for Gate_P4 radii/AABB.
- **Next action**: Continue the CUDA intermediate-state ladder with Gate_P3 RGB colors or Gate_P4 radii/AABB, preserving the P2 flat-opacity semantic split.

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
| `image_layout_chw.md` | Unified CHW layout convention, component map | When writing shaders or image buffer code |
| `vk_cuda_parity.md` | VK vs CUDA training parity analysis | When comparing Vulkan/CUDA results |
| `vk_initial_loss_mismatch.md` | Root cause of VK vs CUDA initial loss gap (3 bugs) | When investigating training loss mismatches |
| `feedback_l2_strict_gradient_parity.md` | L2 grad parity quality bar — treat >1e-3 rel_diff as bug | When evaluating VkVsPyReference gradient tests |
| `sh_adam_group_layout.md` | SH DC/REST Adam-group split layout — gather/scatter required | Before touching SH upload/download in vulkan_trainer.cpp; when post-Adam SH diverges deterministically |
| `ill_conditioned_init_rotation_grad.md` | Identity-quat + isotropic-scale → analytical rotation grad is 0; observed drift is cancellation noise, not bug | When rotation parity test shows large rel_diff with tiny abs_diff |
| `scan1_n_contrib_parity.md` | scan1 n_contrib replay-boundary bug and fixed PSNR milestones | When touching 2D rasterizer/backward replay or scan1 VK↔CUDA parity |
<!-- Add rows as T1 files emerge -->

## Retrieval
**All retrieval cues are in MEMORY_INDEX.md** — topic-organized, cross-tier.
Load it when you need to find what to read before a task.

## Skills & Tools

| Skill | When | What |
|-------|------|------|
| `/resume` | Session start, compaction | Load state, confirm build, set direction |
| `/test-unit` | After code changes | Build + run current unit/CTest inventory |
| `/review` | Feature complete | Code review (+1 self, +2 external) |
| `/sync-docs` | Docs may be stale | Code → docs consistency |
| `/lint-knowledge` | Milestone close | Knowledge consistency audit |
| `/optimize-harness` | Lessons exist or milestone close | Harness self-optimization |
| `/sync-harness` | Push improvements to builder | Bidirectional harness sync |
<!-- Add rows as skills are created -->

**Rule**: If a skill exists for your task, USE IT. Don't reconstruct steps from memory.

## User Preferences
- [Parity investigation discipline](feedback_parity_investigation_discipline.md) — read latest 10 commits, measure errors, and backtrack thoroughly before VK/CUDA parity fixes.
- [Workspace artifacts](feedback_workspace_artifacts.md) — write generated artifacts in the workspace, not `/tmp`.
