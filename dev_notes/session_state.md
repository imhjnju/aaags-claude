# Session State

## Current Phase (AAA defaults performance optimization — 2026-05-15)
After the AAA-default 5000/10000 benchmarks, the first two host-bound performance regressions are optimized. `RegularizationPass` keeps nonzero `opacity_reg`/`scale_reg` on the GPU before VulkanAdam, so regularization no longer forces gradient download/upload when `GS3D_TRAIN_GPU_GRAD_ADAM=1`. The noise path now also allows `GS3D_TRAIN_GPU_RAW_ACTIVATE=1` with nonzero `noise_lr`: GPU activation remains authoritative for forward/backward, pre-Adam active scale/rotation/opacity values are downloaded narrowly for CPU noise semantics, and post-Adam CPU sync downloads only raw positions unless densification or opacity reset requires full raw materialization.

Validation is green: build PASS; focused GPU raw activation/regularization/noise tests PASS; broad `VulkanTrainer|VkVsPyReference|VkVsCudaFirstLoss|Regularization|Dssim|DSSIM` **57/57 PASS**; full Release CTest **362/362 PASS** in **66.88s** with 5 expected skips. Gate 6 review/audit completed with no blockers after strengthening `VulkanTrainer.GpuRawActivationSupportsPositionNoise` to assert the GPU-noise path uses positions-only raw materialization before `raw_params()` performs its legitimate full sync. Final high-N 20-step smoke from the 200k-Gaussian AAA checkpoint at `harmonyos_3dgs/reports/aaa_defaults_noise_partial_smoke_20260515_170618` exited 0 with `raw_download_ms` mean **0.157 ms** (previous **39.658 ms**), `arena_activate_ms` mean **1.869 ms**, `backward_download_ms`/`grad_upload_ms`/`regularize_ms` all **0.000 ms**, and total mean **326.358 ms**. Remaining high-N costs are dominated by CPU `noise_ms` mean **28.747 ms** plus the `R`-scaled forward/backward GPU stages.

## Completed Phase (AAA training defaults sync — 2026-05-15)
`gs3d_vk_train` now defaults to `--training_preset aaa`, aligning the CLI/harness default hyperparameters with AAA-Gaussians CUDA-style training: `lambda_dssim=0.2`, `pos_lr_final=1.6e-6`, `sh_degree_warmup=1000`, `opacity_reg=0.01`, `scale_reg=0.01`, and `noise_lr=5e5`. The old L1/performance-baseline behavior is preserved behind explicit `--training_preset fast`; explicit CLI flags override either preset regardless of flag order. Densification remains off unless requested, and GPU DSSIM remains explicitly gated by `GS3D_TRAIN_GPU_DSSIM_LOSS=1` even when the AAA preset sets `lambda_dssim=0.2`.

Validation is green: Python `py_compile` for the updated harness scripts PASS; Release build for `gs3d_vk_train` and `gs3d_vk_tests` PASS; focused preset/config/trainer/DSSIM coverage **53/53 PASS** after test-audit fixes; broad training/parity coverage **62/62 PASS**; final full Release CTest **360/360 PASS** in **66.15s** with 5 expected skips. A 5-step basketball AAA-default smoke intentionally omitted `--lambda_dssim`, `--pos_lr_final`, `--sh_degree_warmup`, `--opacity_reg`, `--scale_reg`, and `--noise_lr`; with GPU DSSIM/Fuchsia/GPU-resident fast paths enabled it completed at `harmonyos_3dgs/reports/basketball_aaa_defaults_smoke_20260515`, final loss **0.586034**, loss ratio **0.9408**, and nonzero `regularize_ms`/`noise_ms`, confirming the AAA default regularization/noise paths were active.

## Completed Phase (GPU DSSIM productionization — 2026-05-15)
GPU DSSIM is now explicitly gated by `GS3D_TRAIN_GPU_DSSIM_LOSS=1`; `GS3D_TRAIN_GPU_L1_LOSS=1` no longer implicitly enables DSSIM when `lambda_dssim > 0`. The Vulkan DSSIM pass remains matched to the CPU/AAA-style contract: `(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)`, 11x11 Gaussian window, sigma 1.5, `C1=0.01^2`, `C2=0.03^2`. Added trainer coverage for apply-update GPU DSSIM parity and for the L1-env fallback guard. Validation: Release build PASS; targeted `Dssim|DSSIM|GpuDssim|VulkanTrainer` **32/32 PASS**; broader `VkVsCudaFirstLoss|VkVsPyReference|VulkanTrainer|Dssim|DSSIM` **52/52 PASS**; full Release CTest **353/353 PASS** in **66.05s** with 5 expected skips. A 5-step basketball multi-view smoke with `--lambda_dssim 0.2`, `GS3D_TRAIN_GPU_DSSIM_LOSS=1`, `GS3D_REUSE_FORWARD_OUTPUTS=1`, Fuchsia sort, GPU Grad Adam, and GPU raw activation completed at `harmonyos_3dgs/reports/basketball_gpu_dssim_smoke_20260515`, exit 0, final loss **0.584516**, and 5 data timing rows with `backward_download_ms`, `grad_upload_ms`, and `raw_download_ms` all 0.000 ms.

## Completed Phase (Raw/grad transfer elimination — 2026-05-15)
The `/goal` to improve `raw_download_ms` and `grad_upload_ms` by >100x is satisfied on the high-N 20-step smoke after adding the opt-in GPU raw-activation path (`GS3D_TRAIN_GPU_RAW_ACTIVATE=1`) on top of GPU Grad Adam (`GS3D_TRAIN_GPU_GRAD_ADAM=1`). Post-clean report: `harmonyos_3dgs/reports/gpu_raw_activate_highN_smoke_20260515_postclean/summary.json`, exit 0, 20 stage rows, `backward_download_ms`, `grad_upload_ms`, and `raw_download_ms` all **0.000 ms** (mean/p50/max). The fix also threads `PreprocessBackwardGpuInputs` through the non-eval3D preprocess-backward record path so GPU raw activation does not feed stale CPU activated/raw buffers into 2D backward. Validation after clean rebuild: `VulkanTrainer.GpuGradientAdamMatchesCpuGradientUploadStep` + `VulkanTrainer.GpuRawActivationMatchesCpuActivationAfterTwoSteps` **2/2 PASS**, `VkVsPyReference` **8/8 PASS**, and `VkVsCudaFirstLoss` **12/12 PASS**.

## Release+Fuchsia Multi-View Benchmark (2026-05-14)
- Confirmed `profiling_0512`'s 243.4s result came from `build_release/gs3d_vk_train` with Fuchsia default on a cam0-only 5000-step eval_3D/proper_ewa/MCMC workload; rejected replay-order/atomic/GPU-resident experiments were not ported.
- Created current-worktree `harmonyos_3dgs/build_release` with `CMAKE_BUILD_TYPE=Release` and cached FetchContent dependency sources after an initial GitHub SSL download failure.
- 5-step Release+Fuchsia smoke loaded 76 cameras, consumed the deterministic view schedule, emitted `[GS3D_STAGE_TIMING]` rows, wrote PLY/render output, and exited 0.
- Full report: `harmonyos_3dgs/reports/basketball_vk_5000_cap30000_release_fuchsia_20260514_191705/summary.md`; raw artifacts include `vk_train_release_fuchsia.log`, `vk_train_release_fuchsia.time.txt`, `stage_timing.csv`, and `stage_timing_summary.json`.
- Dominant all-step stages: `backward_gpu_ms` mean **40.906 ms** (**59.6%**), `forward_total_ms` **25.994 ms** (**37.9%**), `raster_ms` **11.095 ms** (**16.2%**), `preprocess_process_ms` **7.272 ms** (**10.6%**), `sort_ms` **3.583 ms** (**5.2%**).
- 10000-step / `cap_max=200000` stress sample reran with a report-local 10000-entry schedule after the first attempt correctly failed on the old 5000-entry schedule. Clean result: `harmonyos_3dgs/reports/basketball_vk_10000_cap200000_release_fuchsia_20260514_193728/summary.md`, **1049.00s** trainer / **1049.79s** elapsed, **104.900 ms/step**, final loss **0.024686**, final count **200000**. Tail grew sharply: all-step p50 **88.197 ms**, `step >= 9000` p50 **201.740 ms**, final tail p50 **211.811 ms**; late costs are led by backward, forward, raw download, and grad upload.

## High-Count Fuchsia Sort Performance (2026-05-13)
- Expanded `SorterVulkan`'s opt-in `GS3D_USE_FUCHSIA_SORT=1` route beyond the old project-side `1 << 22` cap, using the vendored Fuchsia library contract of `count < 1 << 30` and persistent buffers sized by Fuchsia memory requirements.
- Tightened `RadixSortFuchsia` constructor/count/key-bit validation and added high-count wrapper/e2e coverage, including a gated sorter route above the old cap and an explicit 7.5M-pair run.
- Validation completed: build PASS; targeted `Fuchsia|Sorter|TileRange` PASS with default large test skip; explicit `GS3D_RUN_LARGE_FUCHSIA_TEST=1 GS3D_LARGE_FUCHSIA_R=7500000` route PASS; external review PASS/no blockers; `ctest -N` reports **330 tests**.
- Performance A/B: Fuchsia high-count log `harmonyos_3dgs/reports/basketball_stage_timing_cam0_5200step_mcmc_fuchsia.log` produced 201 rows for `step 5000..5200`, `N 25914..27209`, `R 6483779..7152290`; total median **437.539 ms** and sort median **166.510 ms**. Report: `harmonyos_3dgs/reports/basketball_stage_timing_step5000plus_N10000plus_fuchsia.md`.
- Remaining caveat: the Fuchsia path still preserves host-materialized `values_sorted` contracts for rasterizer/backward, so `sort_ms` includes current sorted-keyval host materialization; zero-copy sorted-id consumption remains a future optimization, not required for the 4x goal.

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
