# Session State

## Current Phase (merge-training-master — 2026-05-08)
Training/profiling sync into the master cascade/config line is integrated in isolated branch `worktree-merge-training-master`. Training-priority non-parity `eval_3D` replay-order support has now been ported and validated while preserving master `SplattingSettings`, cascade trace/config validation, rasterize specialization IDs, and the non-trace HEAD_W=8 fallback. Current test inventory is **327 tests / 73 .cpp files**.

## Training Replay-Order Sync (2026-05-08)
- Ported training's exact non-parity `eval_3D` replay sideband: `ForwardCache` owns replay offsets/GIDs, `RasterizerVulkan` materializes replay order with a second forward sideband dispatch, and eval_3D backward uploads/consumes replay order when present.
- Binding policy changed after user decision: replay-order owns forward rasterize bindings **14/15**; cascade trace shifted to **16..32** instead of blocking training. Rasterize specialization IDs remain `EVAL_3D=0`, `TRACE_ENABLED=1`, `SORT_MODE=2`, `EVAL3D_RAW_REPLAY=3`.
- Removed the non-parity `eval_3D` trainer rejection and replaced the guard test with `VulkanTrainer.Eval3DNonParityRecordsReplayOrder`, which verifies a non-parity eval_3D step succeeds and replay count matches captured blended contributions.
- Validation: shader build PASS; full build PASS; focused eval_3D replay tests **6/6 PASS**; trainer/backward focus **33/33 PASS**; trace focus **4/4 PASS** with existing skips; `VkVsPyReference` **8/8 PASS**; `VkVsCudaFirstLoss` **12/12 PASS**; basketball/VK-CUDA focus **7/7 PASS** with existing skips; full CTest **327/327 PASS** in **2674.18s**.
- Review/audit: +2 subagent review found no binding/replay blockers; stale parity-guard comments were corrected. Known limitation: the new smoke asserts replay count, not replay GID order content.

## Training Follow-up Sync (2026-05-08)
- Added training-side diagnostic tools `tools/diagnose_l1b_mid_replay.py` and `tools/compare_basketball_eval3d_proper_mcmc.py`; adapted the cam0 benchmark script path base to the merge worktree layout.
- Synced 2D saved-PLY render flags in `tools/compare_vk_cuda_2000step.py` and `tools/compare_vk_cuda_fair.py` so they pass `--eval_3d 0 --proper_ewa 0` explicitly.
- Narrowly ported the `VulkanTrainer::run_forward_and_loss` CPU-arena sizing improvement, then bounded first-frame tile-fanout headroom to avoid unbounded multi-GB allocation before actual `R` is observed.
- Validation: `python3 -m py_compile` for updated/new Python tools PASS; `cmake --build build` PASS; `VkVsPyReference` focused tests **8/8 PASS**; `VulkanTrainer`/`McmcDensify`/`VulkanAdam`/`Densification` focused tests **27/27 PASS**; `ctest -N` still reports **327 tests**.
- Review: independent diff review found the initial unbounded first-frame fanout allocation blocker; bounded follow-up review cleared it. Remaining warning is only that first-frame actual `R/N > 512` can still fail boundedly before `last_bin_R_` is learned.
- Deferred: training replay-order forward/backward shader sideband remains non-mechanical because forward bindings 14/15 conflict with merge trace bindings and specialization IDs; CMake/test/default changes that remove master cascade/config/Fuchsia surfaces remain intentional non-ports.

## Training/Profiling Sync Validation (2026-05-06)
- Build: `cmake -B build -DBUILD_TESTS=ON && cmake --build build` PASS after final training/profiling reconciliation.
- Full CTest: **327/327 PASS**, total **2538.11s**. Expected fixture-dependent skips: `BasketballDs.Trajectory10Steps`, `VkVsCudaBasketball.Cam0_TableSubset`, `CascadeEquivalence.SelfCompare_Cuda_vs_Cuda`, `CascadeEquivalence.Vk_vs_Cuda`.
- Focused first-loss gates: **12/12 PASS**, including newly active `Gate_P7_RenderedImage` and `Gate_L1_L1Loss`; long gates remained bounded (`Gate_P5` **1195.72s**, `Gate_P6` **1197.67s** in full CTest).
- Post-test external audit: PASS, no blockers; confirmed `tests/TEST_PLAN.md` count matches CTest and `.cpp` inventory.
- Apples-to-apples benchmark uses the isolated CUDA reference extension at `diff-gaussian-rasterization` commit `3373529` in `harmonyos_3dgs/build/cuda_ext/dgr-3373529`; the shared AAA-Gaussians checkout remains untouched.
- Final benchmark artifact: `build/compare_runs/matrix_baseline_1000_merge_after_full_integration`; 1000 steps / 76 views / eval_3D / proper_ewa / MCMC / L1 baseline PASSed with final counts **3513/3513**, CUDA final loss **0.082237**, VK final loss **0.079905**, CUDA events `[(600,2892,3036,144,9),(700,3036,3187,151,8),(800,3187,3346,159,13),(900,3346,3513,167,12)]`, VK saved vs CUDA **28.58 dB**, CUDA vs GT **17.44 dB**, VK saved vs GT **17.38 dB**, gate PASS.
- Final benchmark timing with CUDA `3373529`: CUDA train **196.27s**, VK train CLI **363.55s**, CUDA all-view render **5.22s**, VK all-view render **50.88s**, total CUDA **201.50s** vs VK **414.43s** (**0.49x** CUDA/VK ratio).
- Key final syncs: `gs3d_vk_train` exposes parity/proper-EWA/loss/LR/SH/view-schedule controls with fail-fast validation; `VulkanTrainer` scale regularization now uses the `[N,3]` mean denominator; `dump_cuda_training_step.py` is synced with training diagnostic artifacts; first-loss P7/L1 gates are active.
- Intentional remaining deferrals: training replay-order forward/backward shader binding changes conflict with merge trace binding IDs 14/15 and specialization contract; profiling GPU-resident/Fuchsia paths remain opt-in and are not default-enabled.

## Baseline Before Merge
- Master build: OK.
- Master CTest: **262/262 PASS** in `harmonyos_3dgs/build`.
- Hooks: workflow hook syntax OK.
- Worktree: merge performed only in isolated `.claude/worktrees/merge-training-master`; original master worktree remains dirty and untouched.

## Merge Status
- `git merge worktree-training` produced expected conflicts in CMake, rasterizer API/shader, `train_pytorch_reference.py`, `test_e2e_basketball.cpp`, this session state file, and `tests/golden/tiny/py_ref/*.npy`.
- Resolution policy:
  - Keep master `SplattingSettings` validation and cascade/trace plumbing.
  - Keep specialization constants `TRACE_ENABLED=1` and `SORT_MODE=2`.
  - Add `EVAL3D_RAW_REPLAY=3` for training parity raw replay instead of reusing constant id 1.
  - Regenerate `tiny/py_ref` 1000-step goldens with the merged Py reference and sync training-side tiny/basketball CUDA fixtures needed by the merged tests.

## Validation Result After Conflict Resolution
- Build: `cmake -B build -DBUILD_TESTS=ON && cmake --build build` PASS.
- Targeted validation: **93/93 PASS** for config/cascade/VkVsPyReference/VkVsCudaBasketball/eval_3D/MCMC/densification/VulkanAdam coverage; latest parity-sync focused tests **11/11 PASS**.
- Full CTest: **319/319 PASS** in `harmonyos_3dgs/build`.
- Key merge fix: restored 2D rasterizer `n_contrib` to CUDA-style last-candidate-position semantics so backward replay boundaries match `rasterize_backward.comp`.
- Latest parity sync: added final-step `apply_update=false` support plus train/render `--proper_ewa` CLI controls.
- Latest full-dataset sync: added `gs3d_vk_train --view_schedule`, `--require_all_gt`, CLI LR/loss/SH controls, mixed-resolution guard, and `tools/compare_basketball_eval3d_proper_mcmc_full.py`.
- Post-sync validation: build PASS; focused training parity **23/23 PASS**; full CTest **319/319 PASS**.
- Gate 6 isolated test audit: initial gap found in `StepCanSkipAdamUpdate`, fixed with Adam moment invariance assertions; re-audit PASS; final full-CTest log audit PASS.

## Baseline Test/Profile Snapshot (2026-05-01)
- Build: `cmake --build build` PASS.
- Full CTest: **319/319 PASS**, 11 explicit skips, total **82.22s**; Gate 6 log audit PASS.
- Slowest tests: `VkVsCpuRender.FullFramePSNR` **24.07s**, `VkVsPyReference.OraclePerStepComparison` **13.47s**, `VkVsCudaBasketball100Step.PerStepParity` **4.99s**, `Basketball.Training2000Steps` **4.39s**.
- Tiny render profiling (`build/profile_runs/profile_summary.json`, 5 runs each): 2D no-EWA mean **9.88ms**, eval_3D proper mean **9.64ms**, eval_3D parity/proper mean **9.66ms**; CLI wall is ~300ms, so tiny runs are process/setup dominated.
- Tiny training profiling: 100-step 2D **89.6 it/s**, 100-step eval_3D **86.9 it/s**; 1000-step 2D **88.9 it/s**, 1000-step eval_3D **92.9 it/s**.
- Synthetic `gs3d_train_compare --iters 500 --N 10 --sh 1 --res 32`: wall **0.269s**; final loss around **3.8e-4**.

## Full-Dataset Smoke/Profile Snapshot
- Dependency/build gate: basketball init PLY, selected camera JSON/images, AAA-Gaussians tree, `gs3d_vk_train`, `gs3d_vk_render`, conda, and python all present; `cmake --build build` PASS before harness runs.
- Smoke harness (`build/profile_runs/smoke_eval3d_mcmc_s3_v2`): 3 steps / 2 views, eval_3D + proper_ewa, densification disabled by out-of-range gates; CUDA final N **2892**, VK final N **2892**, VK saved vs CUDA **103.41 dB**, VK saved vs train-final **61.74 dB**, gate PASS; timing CUDA train **0.32s**, VK train CLI **0.65s**, total train+render CUDA **0.54s** vs VK **1.49s**.
- Extended harness (`build/profile_runs/extended_eval3d_mcmc_s900_v8`): 900 steps / 8 views, MCMC at steps 600/700/800; CUDA final N **3346**, VK final N **3346**, CUDA events `[(600,2892,3036,144,63),(700,3036,3187,151,13),(800,3187,3346,159,11)]`, VK saved vs CUDA **31.09 dB**, VK saved vs train-final **58.99 dB**, CUDA vs GT **19.47 dB**, VK saved vs GT **19.50 dB**, gate PASS.
- Extended performance: VK training CLI **95.42s** / **9.5 it/s** vs CUDA train loop **33.64s**; VK all-view render **3.68s** vs CUDA all-view render **1.15s**; total train+render CUDA **34.79s** vs VK **99.10s** (**0.35x** CUDA/VK ratio as reported by harness).

## Latest Training Work Being Merged
- MCMC densification: `VkTrainingConfig::cap_max > 0` selects MCMC relocate/add; `cap_max <= 0` keeps legacy clone/split/prune.
- Adam state resize: untouched slots preserve moments; MCMC-modified source/destination slots are zeroed across parameter groups.
- Opacity reset: raw `logit(0.01)` plus opacity Adam moment reset.
- `filter_3D`: propagated through owned raw params, legacy densification, MCMC relocation/add, VulkanTrainer reallocation, and saved PLY output.
- eval_3D training: supported for parity-mode raw replay; non-parity eval_3D training remains guarded until default HEAD/sub-tile backward replay is implemented.
- CLI: `gs3d_vk_train` defaults remain no densification; `--densify 1` defaults to MCMC when `--cap_max` is omitted; `--densify 1 --cap_max 0` selects legacy densification.

## Latest Cascade/Config Work Preserved From Master
- `RasterizerVulkan(ctx, SplattingSettings)` validates CUDA-shared `configs/aaa.json` behavior via `splatting::validate_vk_supported`.
- `rasterize.comp` preserves 3-stage cascade TAIL/MID/HEAD trace path and the non-trace HEAD_W=8 fallback.
- Config tests, cascade fixture dump, VK cascade trace dump, and cascade equivalence skeleton remain in `gs3d_vk_tests`.
