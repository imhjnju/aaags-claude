# Captain's Log

## Session 28 — 2026-05-21 — Latest backward sync and fused replay clear trim

Synced the remaining useful delta from `backward` HEAD `734c056d feat(vk-train): fuse eval3D replay backward` after a read-only diff audit showed the main fused eval_3D replay/tangent/subgroup implementation was already present. The code sync adds `PreprocessorBackwardVulkan::clear_geometry_grad_buffers()` and routes the fused eval_3D replay backward path through it, so the fused path clears only raw position/scale/rotation gradient buffers before raster backward writes them instead of also clearing SH and opacity buffers that are not written by that path.

The sync deliberately did not wholesale overwrite current files: kept `VK_BUFFER_USAGE_TRANSFER_DST_BIT` on fill-cleared preprocessor gradients, kept GPU `cache.dL_dpixels_gpu` precedence and CPU null guards, kept grouped CTest registration, kept current DSSIM/noise/raw-activation/position-noise/cache-skip trainer work, and kept tangent subgroup replay opt-in rather than adopting backward's default-on behavior. Imported `investigations/backward_eval3d_replay_perf.md` as the reproducibility/performance evidence from the backward worktree.

Validation is green: Release `gs3d_vk_train` and `gs3d_vk_tests` build PASS, targeted backward/training/CUDA-first-loss **47/47 PASS**, explicit fused replay env **3/3 PASS**, fused split-backward timing **1/1 PASS**, and full Release CTest **134/134 PASS** in **67.72s** with 5 expected skips. `tests/TEST_PLAN.md` now reports **134 CTest entries / 372 GoogleTest cases**.

## Session 27 — 2026-05-20 — Gate grouping and eval_3D backward perf sync

Synced the `remove_test` gate optimization into the current worktree: CPU GoogleTests now run as one grouped `gs3d_tests` CTest entry, and selected Vulkan infra/sort/config suites run through four manual group entries while excluded from discovery. The gate inventory is **133 CTest entries / 371 GoogleTest cases**, preserving the full GoogleTest case count with a much smaller CTest process count.

Synced the performance-oriented `backward` worktree eval_3D backward path with performance preferred in conflicts, but kept current correctness fixes. The imported pieces include fused replay tangent/subgroup shaders, hot-GID shader/reduction scaffolding currently disabled by `hot_gid_shard_bwd_enabled()`, C++ binding/pass plumbing, CMake shader generation, and Vulkan subgroup BASIC/BALLOT/SHUFFLE device minimums. Current fixes preserved include GPU `cache.dL_dpixels_gpu` precedence, the CPU `dL_dpixels` null guard, `PreprocessorBackwardVulkan` `TRANSFER_DST` gradient buffers, grouped CTest registration, and the existing DSSIM/noise/raw-activation trainer fast paths.

Review found two blockers after the initial import: `backward_record_fused_eval3d_replay_into()` had one unguarded CPU `dL_dpixels` upload, and the disabled hot fused shader treated all full-tile subgroup lanes as active instead of using an active-lane ballot. Both were fixed; re-review found no remaining blockers. Validation is green: Debug and Release `gs3d_vk_tests` builds PASS, focused Debug backward/parity **20/20 PASS**, explicit `GS3D_EVAL3D_FUSED_REPLAY_BWD=1 GS3D_EVAL3D_TANGENT_ROT_BWD=1 GS3D_EVAL3D_TANGENT_SUBGROUP_BWD=1` Debug **4/4 PASS**, focused Release backward **9/9 PASS**, and full grouped Debug CTest **133/133 PASS** in **49.41s** with 5 expected skips.

## Session 26 — 2026-05-15 — Noise raw materialization fast path

Optimized the second AAA-default host-bound regression by allowing GPU raw activation with nonzero `noise_lr`. The trainer now downloads only active scale/rotation/opacity values after GPU activation so CPU `inject_position_noise()` keeps the same pre-Adam active-input semantics, then downloads only raw positions after VulkanAdam before mutating and re-uploading positions. Full raw materialization is still forced for densification, opacity reset, and `raw_params()`/checkpoint boundaries; the positions-only sync intentionally leaves `raw_cpu_dirty_` set because SH and non-position raw CPU vectors remain stale.

Added `VulkanTrainer.GpuRawActivationSupportsPositionNoise`, which compares CPU activation/full materialization against GPU raw activation with noise, asserts GPU Grad Adam and GPU raw activation were used, verifies the GPU-noise path used positions-only raw materialization before `raw_params()` performs its legitimate full sync, and checks raw positions actually changed. Independent review found no code blockers; test audit initially requested materialization-mode instrumentation and cleared the blocker after the strengthened assertion. `tests/TEST_PLAN.md` now reports **362 tests**.

Validation is green: focused raw activation/regularization tests **3/3 PASS**; broad `VulkanTrainer|VkVsPyReference|VkVsCudaFirstLoss|Regularization|Dssim|DSSIM` **57/57 PASS**; full Release CTest **362/362 PASS** in **66.88s** with 5 expected skips. Final high-N smoke from the 200k-Gaussian AAA checkpoint at `harmonyos_3dgs/reports/aaa_defaults_noise_partial_smoke_20260515_170618` completed 20 rows with `raw_download_ms` mean **0.157 ms** vs the prior **39.658 ms**, `arena_activate_ms` mean **1.869 ms**, `backward_download_ms`/`grad_upload_ms`/`regularize_ms` all **0.000 ms**, and total mean **326.358 ms**. Remaining bottlenecks are CPU `noise_ms` mean **28.747 ms** and the `R`-scaled forward/backward GPU stages.

## Session 25 — 2026-05-15 — GPU regularization fast path

Optimized the first AAA-default performance regression by keeping nonzero opacity/scale regularization on the GPU. Added `RegularizationPass` and `regularization.comp` to inject the same gradient corrections as the CPU path directly into `PreprocessorBackwardVulkan`'s raw opacity/scale gradient buffers before VulkanAdam. `use_gpu_grad_adam` now permits nonzero `opacity_reg`/`scale_reg` when capture/debug paths are off; capture/debug still use the CPU fallback, and nonzero `noise_lr` still prevents GPU raw activation.

The new regression test `VulkanTrainer.GpuGradientAdamSupportsRegularization` compares CPU-upload Adam against GPU Grad Adam with strong opacity/scale regularization and captures Adam moments for opacity/scale groups, so the test observes regularization gradient magnitude rather than only first-step raw-param signs. Independent code review and test audit found no blockers after that strengthening. `tests/TEST_PLAN.md` now reports **361 tests**.

Validation is green: focused GPU Adam regularization tests PASS; broad `VulkanTrainer|VkVsPyReference|VkVsCudaFirstLoss|Regularization|Dssim|DSSIM` **56/56 PASS**; full Release CTest **361/361 PASS** in **66.09s** with 5 expected skips. High-N smoke from the 200k-Gaussian AAA checkpoint at `harmonyos_3dgs/reports/aaa_defaults_gpu_reg_smoke_20260515_160119` completed 20 rows with `backward_download_ms`, `grad_upload_ms`, and `regularize_ms` all **0.000 ms**; the remaining high-N bottlenecks are `raw_download_ms` mean **39.658 ms**, `noise_ms` mean **25.485 ms**, plus the `R`-scaled forward/backward GPU stages.

## Session 24 — 2026-05-15 — AAA training defaults and presets

Synchronized the standalone Vulkan training CLI and comparison harness defaults with AAA-Gaussians CUDA-style training. `gs3d_vk_train` now defaults to `--training_preset aaa`, which uses the reference-style defaults for DSSIM loss, position LR decay, SH warmup, opacity/scale regularization, and position noise. The old L1/performance baseline is preserved as explicit `--training_preset fast`, and parser tests lock that explicit overrides win regardless of whether the override appears before or after the preset flag.

Extracted the CLI parser into `vulkan/vk_train_args.{h,cpp}` so preset semantics are unit-testable without brittle executable-output matching. Updated legacy diagnostic scripts (`compare_basketball_eval3d_proper_mcmc.py`, `compare_vk_cuda_fair.py`, and `compare_vk_cuda_2000step.py`) to pass `--training_preset fast` so their historical benchmark intent is unchanged. Updated the full basketball parity harness to default to AAA settings, expose `--training_preset {aaa,fast}`, pass the preset through to Vulkan for traceability, and record it in `config.json` and the summary report.

Validation is green: Python `py_compile` for the updated harness scripts PASS; Release build for `gs3d_vk_train` and `gs3d_vk_tests` PASS; focused preset/config/trainer/DSSIM coverage **53/53 PASS** after isolated test-audit suggestions; broad training/parity coverage **62/62 PASS**; final full Release CTest **360/360 PASS** in **66.15s** with 5 expected skips. A 5-step basketball AAA-default smoke at `harmonyos_3dgs/reports/basketball_aaa_defaults_smoke_20260515` intentionally omitted the sensitive hyperparameter flags, ran with GPU DSSIM/Fuchsia/GPU-resident fast paths, and completed with final loss **0.586034**, loss ratio **0.9408**, plus nonzero regularization/noise timing. Remaining caveat: AAA defaults can be slower; performance benchmarks that want the old L1 baseline should now pass `--training_preset fast` explicitly, and GPU DSSIM still requires `GS3D_TRAIN_GPU_DSSIM_LOSS=1`.

## Session 23 — 2026-05-15 — GPU DSSIM production gate

Productionized the existing Vulkan GPU DSSIM path for AAA-style training. The DSSIM fast path is now explicitly gated by `GS3D_TRAIN_GPU_DSSIM_LOSS=1`; `GS3D_TRAIN_GPU_L1_LOSS=1` no longer implicitly selects DSSIM when `lambda_dssim > 0`. The shader/pass math remains aligned with the CPU/AAA contract: `(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)`, 11x11 Gaussian window, sigma 1.5, and `C1=0.01^2`, `C2=0.03^2`.

Added trainer-level coverage in `test_training_step_vk.cpp`: `GpuDssimFastPathMatchesCpuStepApplyUpdate` verifies two update steps with GPU DSSIM and GPU-resident forward outputs match CPU fallback, while `GpuL1EnvDoesNotEnableDssimFastPath` locks the new explicit-env behavior. `tests/TEST_PLAN.md` now reports **353 tests** and names the GPU DSSIM update coverage.

Validation is green: Release build PASS; targeted `Dssim|DSSIM|GpuDssim|VulkanTrainer` **32/32 PASS**; broader `VkVsCudaFirstLoss|VkVsPyReference|VulkanTrainer|Dssim|DSSIM` **52/52 PASS**; full Release CTest **353/353 PASS** in **66.05s** with 5 expected skips. A 5-step basketball multi-view smoke using `--lambda_dssim 0.2`, `GS3D_TRAIN_GPU_DSSIM_LOSS=1`, `GS3D_REUSE_FORWARD_OUTPUTS=1`, Fuchsia sort, GPU Grad Adam, and GPU raw activation completed at `harmonyos_3dgs/reports/basketball_gpu_dssim_smoke_20260515`, exit 0, final loss **0.584516**, and 5 data timing rows; `backward_download_ms`, `grad_upload_ms`, and `raw_download_ms` stayed **0.000 ms** on all rows. Remaining caveat: scalar loss still downloads the small DSSIM partial-reduction buffer for host summation.

## Session 22 — 2026-05-15 — Raw/grad transfer elimination fast path

Implemented the opt-in GPU raw-activation path for Vulkan training. With `GS3D_TRAIN_GPU_RAW_ACTIVATE=1` and the existing `GS3D_TRAIN_GPU_GRAD_ADAM=1`, raw parameters stay GPU-authoritative across non-densification/non-opacity-reset steps: `raw_activation.comp` activates scales, rotations, opacities, and SH into GPU buffers, `VulkanTrainer::run_forward_and_loss()` feeds those buffers into preprocess, and post-Adam raw CPU materialization is skipped unless a CPU-side boundary needs it.

Fixed the correctness bug found by the two-step regression: the non-eval3D `PreprocessorBackwardVulkan::backward_record_into()` path now honors `PreprocessBackwardGpuInputs` for positions/radii/SH/scales/rotations/opacities/raw rotations instead of uploading stale CPU `g_`/`raw_view_` buffers. Before that fix, the GPU raw-activation path could produce second-step loss drift and NaNs when `raw_params()` materialized the poisoned raw buffers.

Validation after a clean Release rebuild: `VulkanTrainer.GpuGradientAdamMatchesCpuGradientUploadStep` and `VulkanTrainer.GpuRawActivationMatchesCpuActivationAfterTwoSteps` **2/2 PASS**, `VkVsPyReference` **8/8 PASS**, and `VkVsCudaFirstLoss` **12/12 PASS**. The transient `Gate_I3_RawParams` double-free was diagnosed with Valgrind as stale-object ABI fallout after adding the `RawActivationPass` member; clean rebuild cleared it. High-N post-clean smoke at `harmonyos_3dgs/reports/gpu_raw_activate_highN_smoke_20260515_postclean` exited 0 with 20 stage rows and `backward_download_ms`, `grad_upload_ms`, `raw_download_ms` all **0.000 ms** mean/p50/max; total mean **117.513 ms** and p50 **116.998 ms**.

External review found no blocker in the raw-activation/backward path. Remaining caution: the existing CMake `spirv-opt -O` hook optimizes all shaders when available, not only the new raw activation shader, so it should remain tracked as a broader build-determinism/perf variable rather than being attributed solely to this fast path.

## Session 21 — 2026-05-14 — Release+Fuchsia multi-view benchmark hygiene

Confirmed the earlier **243.4s / 20.5 it/s** record from `profiling_0512` was real Vulkan training, not CTest, but it was a cam0-only Release+Fuchsia run. The accepted optimization there was benchmark/build hygiene: `build_release/gs3d_vk_train` with `CMAKE_BUILD_TYPE=Release` and `GS3D_USE_FUCHSIA_SORT=1`; nearby replay-order, workgroup-atomic, and GPU-resident replay experiments were recorded as rejected and were not ported.

Applied that accepted scope to the current 76-view basketball GT benchmark without copying old `profiling_0512` source. Created `harmonyos_3dgs/build_release` using cached FetchContent dependencies after the first configure hit a GitHub SSL download failure. A 5-step Release+Fuchsia multi-view smoke loaded all 76 cameras, used the deterministic schedule, emitted stage rows, wrote outputs, and exited cleanly.

The full 5000-step multi-view Release+Fuchsia run completed cleanly at `harmonyos_3dgs/reports/basketball_vk_5000_cap30000_release_fuchsia_20260514_191705`: **343.70s** trainer time / **344.41s** elapsed, **68.740 ms/step**, 5000 stage rows, final loss **0.032813**, final count **24680**. This is **1.29x** faster than the prior current-worktree 76-view trainer baseline (**444.20s**) and **1.30x** faster than its elapsed time (**448.58s**). The cam0-only 243.4s reference remains context only, not apples-to-apples.

Ran the requested 10000-step / `cap_max=200000` stress sample with the same Release+Fuchsia 76-view scope. The first attempt failed fast because the old schedule had only 5000 entries; the rerun generated a report-local 10000-entry seed-42 schedule and completed cleanly at `harmonyos_3dgs/reports/basketball_vk_10000_cap200000_release_fuchsia_20260514_193728`: **1049.00s** trainer / **1049.79s** elapsed, **104.900 ms/step**, 10000 stage rows, final loss **0.024686**, final count **200000**. Tail cost increased substantially after growth: all-step p50 **88.197 ms**, `step >= 9000` p50 **201.740 ms**, final-tail p50 **211.811 ms**.

## Session 20 — 2026-05-13 — Fuchsia high-count sort reaches 4x basketball target

Expanded the opt-in Fuchsia GPU radix-sort route for high-count basketball training. `SorterVulkan` no longer rejects counts above the old project cap of `1 << 22`; it grows the Fuchsia wrapper and persistent buffers by required capacity while preserving the existing host-materialized `values_sorted`/`tile_ranges` contract. `RadixSortFuchsia` now enforces constructor/count/key-bit limits against the vendored library's `count < 1 << 30` contract.

Added high-count Fuchsia coverage: wrapper memory requirements now include `(1 << 22) + 1` and 7.5M keyvals, invalid max/count/key-bit paths throw, and the e2e sorter route has an explicit large-test gate with configurable `GS3D_LARGE_FUCHSIA_R`. Validation completed for this change: build PASS; targeted `Fuchsia|Sorter|TileRange` PASS; explicit 7.5M large route PASS; external review PASS/no blockers; CTest discovery reports **330 tests**. Full CTest was not claimed because the long CUDA first-loss gates were not rerun to completion after the change.

The high-count basketball A/B with `GS3D_TRAIN_STAGE_TIMING=1 GS3D_USE_FUCHSIA_SORT=1` produced 201 rows for `step >= 5000 && N >= 10000` (`N 25914..27209`, `R 6483779..7152290`). Median total time improved from **1974.840 ms/step** to **437.539 ms/step** (**4.51x**), passing the requested 4x target; median sort time improved from **1610.384 ms** to **166.510 ms** (**9.67x**). The report is `harmonyos_3dgs/reports/basketball_stage_timing_step5000plus_N10000plus_fuchsia.md`.

## Session 19 — 2026-05-08 — Training-priority eval_3D replay-order sync

Ported the remaining training replay-order sideband into the merge worktree after the priority decision that training should win over trace binding preservation. Forward replay-order now owns rasterize bindings 14/15; cascade trace shifted to 16..32 and remained functional. Specialization IDs stayed compatible with the merge contract: `EVAL_3D=0`, `TRACE_ENABLED=1`, `SORT_MODE=2`, `EVAL3D_RAW_REPLAY=3`.

Implemented the exact non-parity eval_3D training replay path: `ForwardCache` owns replay offsets/GIDs, `RasterizerVulkan` does the normal render first then a second replay-capture dispatch, and eval_3D backward binds replay buffers and reverses the captured blend order when available. `VulkanTrainer` no longer rejects `eval_3D && !parity_mode`; the old throw test was replaced with `VulkanTrainer.Eval3DNonParityRecordsReplayOrder`, which verifies a non-parity eval_3D step succeeds and replay count matches captured blended contributions.

Validation is green: shader build PASS, full build PASS, eval_3D focused **6/6 PASS**, trainer/backward focus **33/33 PASS**, trace focus **4/4 PASS** with existing skips, `VkVsPyReference` **8/8 PASS**, `VkVsCudaFirstLoss` **12/12 PASS**, basketball/VK-CUDA focus **7/7 PASS** with existing skips, and full CTest **327/327 PASS** in **2674.18s**. +2 subagent review found no binding/replay blockers; stale parity-guard comments were corrected. Remaining limitation: the smoke checks replay count, not replay GID order content.

## Session 18 — 2026-05-08 — Training follow-up safe-delta sync

Re-ran the remaining training diff classification after the full reconciliation. Safe follow-up ports were limited to additive diagnostics and allocation/observability deltas: added `tools/diagnose_l1b_mid_replay.py`, added the cam0 `tools/compare_basketball_eval3d_proper_mcmc.py` harness with its path base adapted to this worktree, and pinned 2D saved-PLY render mode in `compare_vk_cuda_2000step.py` / `compare_vk_cuda_fair.py` with explicit `--eval_3d 0 --proper_ewa 0`.

Narrowly ported the `VulkanTrainer::run_forward_and_loss` arena estimate improvement, then fixed a review blocker before validation: the initial training hunk used full `tiles_x * tiles_y` as first-frame fanout, which could reserve multi-GB arenas for large PLYs before actual `R` is known. The merge version now keeps 4× observed-R headroom and uses bounded first-frame fanout `[50,512]`.

Validation: Python tool `py_compile` PASS, `cmake --build build` PASS, focused `VkVsPyReference` **8/8 PASS**, and `VulkanTrainer`/`McmcDensify`/`VulkanAdam`/`Densification` **27/27 PASS**. `ctest -N` still reports **327 tests**. Shader/replay-order deltas remain deferred because training forward bindings 14/15 collide with merge trace bindings and the preserved specialization contract.

## Session 17 — 2026-05-06 — Final training/profiling reconciliation validation

Finished the remaining safe training-delta reconciliation after the profiling merge. `tools/dump_cuda_training_step.py` is now byte-for-byte synced with training's diagnostic dump artifacts (`depths.npy`, optional `rects2D.npy`, `gauss2screen.npy`, `aabb_debug.npy`). `tests/test_vk_vs_cuda_first_loss.cpp` now has active P7 rendered-image and L1 loss gates instead of placeholders. `VulkanTrainer` scale regularization was corrected to the Python/CUDA `[N,3]` mean denominator, and `gs3d_vk_train` retains merge-safe defaults while exposing parity/proper-EWA/loss/LR/SH/view-schedule controls and fail-fast validation.

Validation is green: build PASS, focused `VkVsCudaFirstLoss` **12/12 PASS**, full CTest **327/327 PASS** in **2538.11s** with only the four expected fixture-dependent skips. Post-test external audit found no blockers and confirmed the `tests/TEST_PLAN.md` inventory matches both CTest and the `.cpp` file count.

Final 1000-step / 76-view basketball eval_3D+proper_ewa+MCMC benchmark with isolated CUDA `3373529` PASSed at `build/compare_runs/matrix_baseline_1000_merge_after_full_integration`: final counts **3513/3513**, CUDA final loss **0.082237**, VK final loss **0.079905**, VK saved vs CUDA **28.58 dB**, CUDA vs GT **17.44 dB**, VK saved vs GT **17.38 dB**, gate PASS. Timing: CUDA train **196.27s**, VK train CLI **363.55s**, CUDA render **5.22s**, VK render **50.88s**, total CUDA **201.50s** vs VK **414.43s**.

Remaining code differences were audited rather than blindly copied. Training replay-order forward/backward shader bindings remain deferred because they collide with merge trace binding IDs 14/15 and the preserved rasterize specialization contract. Profiling GPU-resident/Fuchsia fast paths remain opt-in and are not default-enabled.

## Session 16 — 2026-05-06 — 1000-step benchmark rerun with isolated CUDA 3373529

The initial apples-to-apples rerun failed on the CUDA side before Vulkan comparison: the shared dirty AAA-Gaussians `diff-gaussian-rasterization` checkout at HEAD `72635e8` produced non-finite gradients on step 1 and NaN loss from step 2 in both the merge and training worktrees. Built an isolated workspace-local extension from commit `3373529` under `harmonyos_3dgs/build/cuda_ext/dgr-3373529` and patched only its Python 3.13 dataclass wrapper compatibility (`field(default_factory=...)`) so imports use the 3373529 CUDA `.so` without modifying the shared dirty checkout.

With `PYTHONPATH` routed to the isolated extension, the 3-step CUDA-only reproducer no longer NaNs (`0.550707`, `0.615173`, `0.605667`). The full 1000-step / 76-view / eval_3D / proper_ewa / MCMC / L1 baseline then PASSed at `build/compare_runs/matrix_baseline_1000_merge_current_cuda3373529`: final counts **3513/3513**, CUDA final loss **0.082979**, VK saved vs CUDA **27.93 dB**, CUDA vs GT **17.28 dB**, VK saved vs GT **17.33 dB**, gate PASS.

Performance with CUDA `3373529`: CUDA train **46.77s**, VK train CLI **119.48s**, CUDA all-view render **1.59s**, VK all-view render **26.05s**, total CUDA **48.35s** vs VK **145.52s**. Compared to the training baseline (`48.23s`, `82.83s`, `1.60s`, `22.10s`, total `49.84s` vs `104.92s`), CUDA is comparable while current merge VK is slower, especially training.

## Session 15 — 2026-05-06 — Training/profiling sync into cascade master

Synced selected `worktree-training` and low-conflict `worktree-profiling` changes into the isolated merge worktree while preserving master-side `SplattingSettings`, cascade trace/config validation, rasterize specialization IDs, and the non-trace HEAD_W=8 fallback. The profiling sync brought in the Fuchsia radix sort vendor/wrapper/tests and packed-keyval compatibility without wholesale replacing the sorter/rasterizer GPU-resident path.

Resolved the final first-loss blocker without weakening tests. `Gate_P5_SortedIds` initially exceeded the boundary masking budget (`34 > 32`) after sync; comparison against the training worktree showed `preprocess.comp` had lost eval_3D parity numerics (`precise` intermediates and CUDA trunc-style `normalizeAngle`). Restoring those semantics made P5 and P6 pass with the existing assertions.

Validation: build PASS; focused `VkVsCudaFirstLoss.Gate_P5_SortedIds` PASS in 1227.88s; focused `Gate_P6_TFinalNContrib` PASS in 1226.99s; full CTest PASS **326/326** in 2533.93s. Gate 6 external test/log audit found no blockers and noted the remaining P7/L1 first-loss gates are intentional skips.

## Session 14 — 2026-04-29 — Vulkan MCMC densification port

Ported the AAA-Gaussians MCMC densification path from the `densification` worktree into the training worktree without replacing the existing eval_3D parity files wholesale. The new path is selected by `VkTrainingConfig::cap_max > 0`; `cap_max <= 0` preserves the legacy clone/split/prune densification path. `vk_train_main` still disables densification by default for parity-safe CLI behavior.

Added `mcmc_densification` and `relocation` modules plus CUDA-golden replay fixtures. Test coverage now includes relocation math, MCMC relocate/add/densify behavior, replay against CUDA-generated sample plans, VulkanTrainer MCMC integration, Adam moment preservation, opacity reset, and `eval_3D + MCMC` filter propagation.

Extended `VulkanAdam` with state-preserving group resize and selective moment zeroing. `VulkanTrainer` now reallocates Adam groups without resetting untouched MCMC slots, zeros modified source/replaced destination slots across all six parameter groups, and resets opacity to raw `logit(0.01)` on schedule while zeroing opacity Adam moments.

Independent review found substantive issues and they were fixed before final validation: `eval_3D + MCMC` would lose/misassign `filter_3D`, opacity reset initially left stale opacity moments, the opacity reset gate was too broad, and legacy split children initially failed to inherit `filter_3D`. Current behavior propagates filter values through MCMC relocate/add and legacy clone/split paths, zeros group-3 moments on reset, and gates reset to the densification window.

Validation: build passed; targeted MCMC/Adam/densification/training/config/filter tests passed **54/54**; targeted final-step/trainer/densification parity tests passed **62/62**; parity-sensitive regression passed **42/42** with expected skips; full CTest passed **314/314** after adding terminal no-update coverage. Independent subagent review of current uncommitted changes found no blockers after the filter propagation pass. The project test inventory remains 66 `.cpp` test files.

Final-step CUDA parity was fixed for `gs3d_vk_train`: the terminal iteration still runs forward/backward and updates `step_count_`, but skips Adam, raw-param download, noise injection, densification, and opacity reset. This matches the CUDA harness contract where the final loss/render is computed without a final optimizer step. A focused `VulkanTrainer.StepCanSkipAdamUpdate` regression verifies finite loss, populated nonzero gradients, unchanged raw params, and `step_count()==1` for the no-update mode.

The 5000-step basketball MCMC comparison with `cap_max=50000`, no opacity reset, and matched densification gates produced matched final counts (**CUDA 24680 / VK 24680**). After the final-step fix, the VK training-final render was aligned with CUDA (`VK train final vs CUDA 29.94 dB`, `VK train final vs GT 27.77 dB`, `CUDA vs GT 27.66 dB`), but the saved-PLY standalone render initially remained bad (`old standalone vs GT 22.59 dB`) with the basketball white-block distortion.

The saved-PLY artifact was isolated to render-config mismatch, not PLY serialization: training uses `VkTrainingConfig::proper_ewa=false`, while standalone `gs3d_vk_render` previously defaulted `PreprocessorVulkan` to `proper_ewa=true`. Added `--proper_ewa 0|1` to `gs3d_vk_render` and updated 2D comparison harness render calls to pass `--proper_ewa 0`. Re-rendering the same saved PLY with `--proper_ewa 0` matches the VK training-final render at **74.68 dB** (`mean_abs=0.000009`, max `0.007843`), while preserving the expected comparison against CUDA (`29.94 dB`) and GT (`27.77 dB`). Full CTest after the render CLI change passed **314/314** in 40.25s, and an independent subagent diff review found no blockers.

Added `--proper_ewa 0|1` to `gs3d_vk_train` so training-time preprocessing can explicitly match CUDA `ExtendedSettings.proper_ewa_scaling`. The requested basketball cam0 eval_3D/proper_ewa/MCMC experiment used `/tmp/basketball_init_3dgs_filter3d.ply`, `cap_max=30000`, `steps=5000`, `opacity_reset_interval=0`, CUDA gate `iter > 500 && iter < 4999 && iter % 100 == 0`, and VK gate `step >= 600 && step <= 4999 && step % 100 == 0`. Final counts matched (**CUDA 24680 / VK 24680**). Final metrics: `CUDA vs GT 27.13 dB`, `VK saved vs GT 27.25 dB`, `VK saved vs CUDA 29.37 dB`, `VK train vs CUDA 29.36 dB`, and `VK saved vs train 59.04 dB`. Artifacts live under `harmonyos_3dgs/build/compare_runs/basketball_eval3d_proper1_mcmc_5000_cap30000/`; the reusable runner is tracked at `harmonyos_3dgs/tools/compare_basketball_eval3d_proper_mcmc.py`. Full CTest after the train CLI change passed **314/314** in 40.63s.

Extended the comparison from cam0-only to true full-dataset/multi-view parity. `gs3d_vk_train` now accepts `--view_schedule <path>` so CUDA and Vulkan consume the exact same 0-based camera index sequence, and `--require_all_gt 1` now fails if any selected camera lacks a GT image instead of silently dropping views. The CLI also rejects mixed-resolution multi-view training for now because `VulkanTrainer` allocates image buffers from the first view dimensions. This makes the multi-view training contract explicit: same camera list, same GT list, same per-iteration schedule, same final-step no-update semantics.

Added `harmonyos_3dgs/tools/compare_basketball_eval3d_proper_mcmc_full.py`, a full-dataset basketball eval_3D/proper_ewa/MCMC harness. It loads all selected cameras and GT images, writes a shared schedule, trains CUDA inline, trains Vulkan via `gs3d_vk_train --view_schedule --require_all_gt 1`, renders every selected view from the saved PLYs, reports aggregate/per-view metrics, compares the saved PLY render to the Vulkan training-final render for the last scheduled view, and exits nonzero on Gaussian-count mismatch, low VK-vs-CUDA PSNR, or excessive CUDA/VK GT-quality gap.

Independent review found four harness/CLI issues and they were fixed before final validation: empty/missing `img_name` could bypass `--require_all_gt`, nested/suffixed image names needed parent-directory creation and `.ppm` lookup handling, CUDA's exclusive densification upper bound needed mapping to Vulkan's inclusive bound, and saved-PLY-vs-training-final consistency needed to be reported and gated. The fixed all-view 5-step smoke passed with `VK saved vs CUDA 96.30 dB` and `VK saved vs train final view 63.29 dB`.

Final full-dataset validation used all 76 basketball views, eval_3D=true, proper_ewa=true, `cap_max=30000`, 1000 steps, shared schedule, no opacity reset, CUDA gate `iter > 500 && iter < 999 && iter % 100 == 0`, and VK gate `step >= 600 && step <= 998 && step % 100 == 0`. CUDA/VK final counts matched (**3513 / 3513**). Metrics: `CUDA vs GT 17.22 dB`, `VK saved vs GT 16.92 dB`, `VK saved vs CUDA 27.99 dB`, `VK saved vs train final view 59.24 dB`, GT-quality gap `0.30 dB`; harness gate status PASS. A final rebuild plus full CTest after review fixes passed **314/314** in 40.18s.

## Session 13 — 2026-04-28 — scan1 n_contrib replay-boundary fix

Switched the parity target to `/home/robota/h00813233/Graph/datasets/scan1` camera 0 and measured VK↔CUDA independent-training drift at 10/200/1000 steps. Initial render was bit-identical/all-black; the first real split was step1 SH gradient drift. Forward buffers showed `T_final` matched, but `n_contrib` did not: CUDA records the 1-based candidate position of the last Gaussian that blended, whereas Vulkan/CPU were using blended-count semantics.

Fixed the 2D path by making `rasterize.comp` store the CUDA-style last-contributor position and making `rasterize_backward.comp` replay only candidates up to that position. Synchronized the CPU 2D rasterizer/backward reference and the Vulkan backward fixture comment so regression tests assert the same contract. Validation: affected test `RasterizerBackwardVulkan.MatchesCPU_TinyFixture` passed, then full CTest passed **273/273**.

Impact on scan1: VK↔CUDA final-render PSNR improved **90.91→107.14 dB at 10 steps**, **22.42→49.82 dB at 200 steps**, and **8.24→27.77 dB at 1000 steps**. The 1000-step final numbers after the fix were CUDA-vs-GT 19.753679 dB, VK-vs-GT 19.868585 dB, VK-vs-CUDA 27.773822 dB, render max_abs 0.6992977, mean_abs 0.0225525.

Remaining drift appears numerically amplified rather than a newly localized formula bug: step1 forward/position/opacity/scale/rotation gradients are exact, SH gradients differ only at atomic accumulation scale (`l2_rel≈6.2e-4`, max `≈1.9e-7`), and step2 render is still close (`l2_rel≈4.3e-5`, max `≈3.9e-6`). Adam with `eps=1e-15` turns near-zero step2 gradients into finite raw-parameter deltas, so the next controlled experiment should start both implementations from an identical post-step1 state or use a non-degenerate init before chasing more shader math.

Follow-up eval_3D smoke correction: added `--eval_3d 0|1` to `gs3d_vk_train`, then found the first smoke was misleading because the CLI set `RenderConfig::eval_3D` but not `VkTrainingConfig::eval_3D`, while `VulkanTrainer` specializes preprocessor/rasterizer from `VkTrainingConfig`. After appending AAA-style `filter_3D`, forward eval_3D rendering is not empty: black background appears black because the init RGB is zero, while `BG_WHITE=1` yields 1,450,525 non-white pixels. Propagating the flag into `VkTrainingConfig` made true eval_3D training hit the existing backward unsupported guard instead of silently running a non-eval_3D trainer.

Eval_3D backward port: added separate Vulkan eval_3D rasterizer and preprocessor backward passes instead of extending the fixed 2D shaders in-place. The new path propagates raster gradients into `d_gauss2screen`, then through the eval_3D preprocessor chain into raw position/scale/rotation/SH/opacity updates while preserving `filter_3D` storage in `VulkanTrainer`. Removed the temporary CLI fail-fast guard after `VulkanTrainer.Eval3DOneStepSmoke` and a one-step `gs3d_vk_train --eval_3d 1` CLI smoke passed. Added `VulkanTrainer.Eval3DStep1RawGradientParity`; the tiny-fixture first-step raw gradients now match CUDA within `2e-3` L2-relative / `2e-5` max-absolute tolerance after fixing parity-mode eval_3D forward to use CUDA-style direct blend/replay-boundary semantics and reading `d_gauss2screen` with the correct transposed storage interpretation in preprocessor backward. The same test now directly compares CUDA/VK eval_3D raster backward `d_rgb` (`l2_rel=1.76e-3`, max_abs `1.09e-5`), `d_opacity` (`l2_rel=1.52e-4`, max_abs `1.82e-6`), and `d_gauss2screen` (`l2_rel=5.63e-4`, max_abs `2.65e-5`). `d_gauss2screen` required a non-breaking CUDA extension binding, `rasterize_gaussians_backward_dump_g2s`, because the original Python API computed that tensor internally but discarded it. Validation now stands at full CTest **276/276 PASS** after adding `VulkanTrainer.Eval3DStepRequiresParityMode`, which rejects eval_3D training with `parity_mode=false` until default HEAD/sub-tile backward replay is implemented. Follow-up SH→position audit: CUDA eval_3D appears to call `computeColorFromSH`, but `computeGauss2Screen*` subsequently assigns `dL_dmean[idx] = ...` and overwrites that contribution; adding the 2D SH mean term to Vulkan worsened pos parity from `1.16e-3` to `9.52e-2`, so Vulkan intentionally preserves the CUDA overwrite behavior for now. Shared-memory audit found the unified forward shader's eval_3D path could exceed a 32 KiB budget when `sub_order[16][256]` used 32-bit slots; packing four 8-bit batch indices per word keeps the conservative footprint around 31.1 KiB, and the post-packing build plus full CTest still pass **276/276**. The CLI now wires `--eval_3d 1` training to the validated `parity_mode=true` path; scan1 camera0 10-step eval_3D smoke with `/tmp/scan1_init_3dgs_filter3d.ply` completed at `/tmp/scan1_vk_eval3d_filter_10_postbwd/`, with loss `0.665102→0.660058` and final render PSNR `2.664990 dB` against the JPEG-converted camera0 GT.

100-step VK↔CUDA numerical validation was rerun on basketball cam0. `VkVsCudaBasketball100Step.PerStepParity` passed and printed the full per-step loss/gradient/backward-chain/Adam m-v diagnostics; at step100, loss stayed closely aligned (`VK 0.383244`, `CUDA 0.383523`, abs diff `2.796e-4`), while scale/rotation gradients remain the known numerically amplified groups (`g_sca l2_rel=4.517e-1`, `g_rot l2_rel=8.811e-1`). The fair 100-step render comparison produced CUDA-vs-GT `7.96 dB`, VK-vs-GT `7.94 dB`, VK-vs-CUDA `54.51 dB`, gap `0.02 dB`, with artifacts under `/tmp/compare_fair_2000/`.

1000-step fair VK↔CUDA validation was rerun in a clean directory after an interrupted/stale mixed-artifact attempt. Fresh artifacts live under `/tmp/compare_fair_1000_20260429/`. CUDA reached final loss `0.046733` and PSNR `21.05 dB`; VK reached final loss `0.046796` and PSNR `21.16 dB`. The final VK-vs-CUDA render PSNR was `28.71 dB`, with `Gap (CUDA−VK) = -0.11 dB`, so both implementations converge to essentially the same GT quality while parameter/render trajectories remain visibly different at 1000 steps.

1000-step eval_3D VK↔CUDA validation was then run on basketball cam0. Because `/tmp/basketball_init_3dgs.ply` lacked `filter_3D`, a filtered init PLY was generated at `/tmp/basketball_init_3dgs_filter3d.ply` using the cam0 depth/focal formula (`filter_3D` nonzero for 2413/2892 Gaussians, max `0.0067259199`). The fair script now has explicit `--eval_3d` and `--init_ply` switches, and uses the VK training-emitted final render for eval_3D because `vk_train_main` currently drops `filter_3D` when saving the trained PLY. Fresh artifacts live under `/tmp/compare_eval3d_1000_20260429/`. CUDA reached final loss `0.044052` and PSNR `21.35 dB`; VK reached final loss `0.044153` and PSNR `21.32 dB`; final VK-vs-CUDA PSNR was `28.47 dB`, with `Gap (CUDA−VK) = 0.03 dB`. Quantizing both renders to 8-bit gives the same conclusion (`VK-vs-CUDA 28.465943 dB`, mean abs `0.021046`).

Eval_3D tile/block artifact investigation: rendering the CUDA-trained eval_3D PLY through `gs3d_vk_render --eval_3d 1` isolated the artifact to VK forward, not training trajectory (`VK-vs-CUDA 22.72 dB`, mean abs `0.04229`). CUDA default `ExtendedSettings(eval_3D=True)` keeps `sort_mode=GLOBAL` and `tile_based_culling=false`, whereas Vulkan eval_3D scatter used StopThePop per-tile max-depth keys and INVALID tile culling. Added `RenderConfig::eval_3D_parity_mode` so parity paths use normal view-space-depth scatter and disable rasterizer sub-tile resort; `gs3d_vk_render` now exposes `--parity_mode 0|1`. The same CUDA-trained PLY rendered with VK parity binning improved to `VK-vs-CUDA 51.74 dB`, mean abs `0.000738`, and the visible tile/block patches disappeared in the comparison image. A 1000-step eval_3D rerun under parity binning produced CUDA-vs-GT `21.35 dB`, VK-vs-GT `21.33 dB`, VK-vs-CUDA `29.80 dB`, gap `0.02 dB` at `/tmp/compare_eval3d_1000_parity_binning_20260429/`. Added `TileBinner.Eval3DParityMode_UsesViewDepthKey` to lock the CUDA GLOBAL/no tile-culling scatter contract. Full CTest after the fix is **277/277 PASS**.

## Session 12 — 2026-04-28 — Fair-path alignment + rotation-drift triage

Closed the fair-path setup bugs that made the CLI comparison worse than the focused harness: full SH is now active from step 1 when `sh_degree_warmup=0`, the CLI projection matrix matches CUDA/`camera_utils.cpp`, and `vk_train_main.cpp` no longer frees model storage before constructing the trainer. Regenerated 10/100-step CUDA goldens and verified focused 10-step and 100-step VK↔CUDA tests pass.

Key results: fair 10-step is now aligned (`VK vs CUDA 43.14 dB`), but fair 200-step still diverges. Found and fixed a reporting bug: CUDA final render used `eval_3D=false`, while `gs3d_vk_render` hardcoded `eval_3D=true`. Added strict `--eval_3d 0|1` to `gs3d_vk_render` and updated `compare_vk_cuda_fair.py` to pass `--eval_3d 0`. Re-rendering the existing 200-step isotropic model with VK2D gives `CUDA vs GT 14.65 dB`, `VK2D vs GT 11.33 dB`, `VK2D vs CUDA 14.68 dB` — still a real training gap, though the old render-mode mismatch affected the reported numbers.

Negative results: the conic off-diagonal convention is paired and unsafe to change alone. A scalarized `preprocess_backward.comp` `W/J/T/Vrk/VT` experiment was reviewed and tested, but it only moved 100-step drift marginally, so it was reverted. The 100-step chain diagnostics then exposed the first causal split: CUDA's actual backward kernel returns ~1e-11 rotation residuals at step 1 for identity-quaternion + isotropic-scale Gaussians, while VK cancels to exact zero; Adam `eps=1e-15` turns that into ~1e-3 raw-rotation updates immediately. This is numerical-cancellation amplification, not yet a Vulkan chain-rule bug. Next work should run a paired control that neutralizes near-zero rotation gradients or uses a non-degenerate init before changing shader math.

## Session 11 — 2026-04-25 — Phase A + C.0 + D.deep SH layout closure

Long, multi-arc session. Started after S10's master merge. Closed three major items: Phase A (CPU↔VK 99 dB via proper_ewa default flip), C.0 (Gate_P1_Means2D PASS via dilation gating + test wiring + masking + relative tolerance), and D.deep (the SH Adam-group layout bug — the most important find of the day). Plus latent-bug alignment in Python reference, full test sensitivity tightening with permanent sentinels, and ~3000× tighter parity verified to 10 steps.

### Headline findings

1. **MEMORY.md baseline numbers were stale.** Pre-session, MEMORY.md still claimed "VK eval_3D 42.8 dB / 17.2 dB gap to 60 dB". Empirically, post-S10 near-plane cull fix (commit `1b02ca2`) had already pushed `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` to **54.84 dB** — only 5.16 dB to 60 dB target. Updated MEMORY.md to reflect.

2. **The atomicAdd hypothesis was wrong for the SH bug.** Earlier in the day I attributed VK↔Python gradient drift to `rasterize_backward.comp` atomicAdd nondeterminism. The user pushed back: "别怀疑 atomicAdd 这种不好定位的问题". Forced through a step-1 intermediate-value audit (14 forward stages dumped, dL_dimage confirmed bit-identical) and a 3-step trajectory analysis. The 3-step result showed **deterministic** 469% rel_diff on G[2] post-Adam SH — same Gaussian every run, every step. atomicAdd noise can't produce deterministic divergence. Lesson saved as `gotchas.md` "Don't default-blame atomicAdd for backward drift" and `memory/sh_adam_group_layout.md`.

3. **Real root cause: SH Adam-group upload layout mismatch.** 3DGS uses two Adam groups for SH (DC lr=2.5e-3, REST lr=DC/20). VK trainer was `memcpy`-splitting the unified interleaved `[N,K,3]` CPU buffer by float-index — for K=16, the first N\*3 floats are NOT all DCs (G[0]'s entire 48 SH + G[1]'s first 4). G[2..N-1]'s DCs landed in REST and updated at 1/20 the correct lr. Fix: gather/scatter helpers in `vulkan_trainer.cpp` (5 call sites). Result: 3-step SH max-elem rel_diff 469% → 0.015%; step-2 loss 1.10e-3 → 5.5e-7; 10-step trajectory all groups bounded.

4. **L2-norm tests systemically hide single-element drift.** The pre-fix bug passed `Step1GradientAndLoss` because L2-norm rel_diff 9.85e-7 (norm dominated by DC-zero magnitudes ~1) is small even when ONE DC element is 469% off. Tightened test: per-element max rel_diff assertion at 1e-4 + abs_diff column + `%.6e` print format. Graduated `TempDiag3StepSHCompare` and `TempDiag10StepAllGroups` to permanent sentinels with calibrated EXPECT_LT.

### What was done (chronologically)

1. `/resume` to recover state. Realized MEMORY.md numbers were stale; queued correction.
2. **Phase A**: investigated why `VkVsCpuRender.FullFramePSNR` was 28.75 dB despite master claiming 99.15 dB after CPU 2D forward + h_conv backward fix. Diff between worktree-training and master: `proper_ewa` was added as a `PreprocessorVulkan` ctor parameter with default `false`, but the existing CPU↔VK test wasn't wired with explicit `true` opt-in; shader's else-branch (3.33σ basic, no convolution_scaling) was running. Fix: flip default to `true`, opt-out at parity-harness sites. Suite: `VkVsCpuRender` 28.75 → **99.15 dB**, `PreprocessPass.MatchesCUDAGolden_Tiny` also self-healed.
3. **C.0**: Investigated `Gate_P1_Means2D` failure. First diagnosed `compute_aabb_view` bounded loop hypothesis (refuted: bit-identical output with iter cap raised). Then looked at masking: 336 CUDA "huge means" are CUDA's `tan(±π/2 - ε)` degenerate-AABB fallback (CUDA's "I give up, full screen" semantic), 5903 only-CUDA-rasterizes are dilation_factor mismatch (CUDA gates on `proper_ewa_scaling`, VK was unconditional), ~2887 "both rasterize" residual disappeared once dilation gating fixed. Final: gate dilation in `preprocess.comp:967` + wire basketball test 3 sites with `proper_ewa=true` (matches its golden's `aaa.json features = proper_ewa_scaling=True`) + test logic mask + reltol. `Gate_P1` FAIL (8790 bad) → **PASS (0 bad)**. Basketball PSNR preserved at 54.84 dB.
4. **D.deep**: User pushed back on "atomicAdd is the bug". Did step-1 intermediate-value audit confirming forward sub-ULP-equivalent and `dL_dimage` bit-identical. Then 3-step trajectory analysis exposed the SH layout bug. Fixed `vulkan_trainer.cpp` with gather/scatter helpers. Verified at 10 steps: all groups bounded < 1e-3 rel_diff except documented gpos atomic noise debt.
5. **Latent algorithmic bugs**: prior audit had flagged 2 inactive-on-fixture differences (det floor, frustum 1.3× clamp). Confirmed CUDA is canonical; aligned Python ref proactively. Goldens regenerated (1114 files, 937 perturbed at ≤2.4e-7). Basketball cam0 unchanged.
6. **Test sensitivity**: tightened `Step1GradientAndLoss` (5% → 1e-3 norm; new 1e-4 per-elem; `%.6e` printing; abs_diff column). Graduated 2 sentinels with calibrated thresholds (`Step3PostAdamSHParity`, `Step10TrajectoryAllGroups`).
7. **Memory updates**: `gotchas.md` + new `sh_adam_group_layout.md` + new `feedback_l2_strict_gradient_parity.md`. Captured the methodology lesson + the bug pattern + the strict-parity preference.
8. **Commits**: `d8c388f` (Phase A + Gate_P1 + SH fix bundled), `c58f4eb` (test sensitivity + sentinels), `3dcfb04` (Python ref alignment + audit instrumentation + 1114 goldens).

### Outstanding debt

- **gpos atomic noise**: `Step1GradientAndLoss.gpos` per-element 1.89e-4 (sub-ULP abs 1.5e-9). `TODO(deterministic-backward)` in `rasterize_backward.comp` — 7 atomicAdd sites at lines 252, 282, 284, 297, 299, 301, 305. Adam smooths it; 10-step trajectory bounded; defer.
- **L1b 5.16 dB to 60 dB**: Gates P2-P7 + L1 still SKIP. Phase C.1 (P2-P5 implementation) and Phase C.2 (cascade trace harness) remain open.
- **L5 independent-train comparison test**: doesn't exist yet. Phase E.
- **Python reference single-group SH Adam at sh_degree>0**: when fixture moves to higher SH degree, Python ref's single-group Adam (lr=2.5e-3 for all SH) will diverge from VK's two-group split. Update Python ref before running degree>0 tests.
- **Basketball ply downsampled parity**: tried; full-resolution is days+ runtime in Python autograd; downsampled was launched and hit usage limit. Revisit after quota reset.

## Session 10 — 2026-04-25 — Merge master → worktree-training

Imported the latest `master` (CPU 2D forward + CPU/VK backward alignment) into
`worktree-training` while the in-progress first-loss-parity work was still
uncommitted on the working tree. Clean merge, no conflicts, build green, all
local edits preserved.

### What was done
1. **Investigation**: confirmed remote layout — primary branch is `master` (not `main`); remotes are `github` and `gitee` (no `origin`). `github/master` is far behind local `master` (still at "add harness"); local `master` was the actual integration target.
2. **Pre-merge analysis**: 3 commits on `master` not in `worktree-training`:
   - `09217bb` fix(preprocess-2d-cpu): align proper_ewa_scaling + tight_opacity_bounding with VK/CUDA
   - `612c125` fix(preprocess-backward): add h_conv_scaling chain rule to CPU and VK backward
   - `ccd093e` Merge branch 'worktree-white-table': CPU 2D forward + CPU/VK backward alignment
   Files changed by master: `src/cpu/preprocessor_cpu.cpp`, `src/cpu/preprocessor_backward_cpu.cpp`, `src/vulkan/shaders/preprocess_backward.comp`.
3. **Overlap check**: working-tree dirty list (preprocess_pass.{h,cpp}, preprocessor_vulkan.{h,cpp}, preprocess.comp [forward, NOT backward], vulkan_trainer.{h,cpp}, camera_utils.cpp, tests, tools) — no overlap with master's 3 files. `git merge-tree` dry-run produced a single tree hash with no conflict markers. Case A.
4. **Merge**: `git merge master --no-ff` with explanatory message, producing merge commit `5ee56bc`.
5. **Build verification**: `cmake --build build` green end-to-end. CPU preprocessor objects rebuilt, `preprocess_backward.spv` recompiled, all executables relinked.
6. **Working tree intact**: same 20 modified files + same untracked golden npy/tools/raw artifacts as before merge — nothing stashed, nothing reset.

### Key findings
- The user task spec named the branch `main` and the remote `origin`; this repo uses `master` and `github`/`gitee`. Adapted accordingly.
- The dev_notes location the user pointed at (`harmonyos_3dgs/dev_notes/{session_state,captains_log}.md`) does not exist — the canonical files live at the worktree root `dev_notes/{session_state,captains_log}.md` per `MEMORY.md`. Updated those.
- Master's changes are pure backward-parity fixes on CPU/VK backward; they unblock CPU↔VK numerical alignment but don't touch the VK forward `preprocess.comp` path the S10 first-loss-parity work is editing. Net safe pickup.

### Next steps
- Resume S10 first-loss-parity work on the now-rebased base. The new h_conv_scaling backward chain may shift VK gradient numbers slightly relative to pre-merge captures; if `VkVsPyReference.Step1GradientAndLoss` regresses, the chain-rule fix is the likely cause and should be reflected in any cached Python reference dumps.
- Evaluate whether the 5 dB VK-vs-CUDA gap discussion in S8 is now (with `VkVsCpuRender.FullFramePSNR=99.15` from master's CPU alignment) re-framable: CPU↔VK is now nearly bit-exact at 2D, so any remaining VK↔CUDA gap is firmly in the rasterizer cascade, not the preprocess.

---

## Session 8 — 2026-04-21

VK eval_3D rasterizer vs CUDA golden parity push: PSNR 25.3 → 42.8 dB (17.2 dB gap remains to 60 dB target).

### What was done
1. **Harness built**: `test_vk_vs_cuda_basketball.cpp` — loads basket-aaa.ply cam0, renders VK, compares against CUDA golden (aaa.json features). PSNR regression gate with `kBaselinePSNR=42.1f`.
2. **Feature fixes (Tasks 9-12)**:
   - `proper_ewa_scaling` in preprocess.comp: opacity *= dilation_factor (+11.6 dB, 25.3→36.9)
   - `rect_bounding` + `tight_opacity_bounding` in 2D path: (0 dB, 36.9→36.9)
   - `tile_based_culling` in scatter.comp: INVALID sentinel for non-contributing tiles (+3.4 dB, 36.9→40.3)
   - Hierarchical sub-tile TAIL re-sort in rasterize.comp: per-4×4 sub-tile depth sort (+2.3 dB, 40.3→42.6)
   - Tuning: HEAD_W=8, subtile_cx+2.0, z/w depth key (+0.2 dB, 42.6→42.8)
3. **Structural audit**: 5 parallel subagents audited preprocess/rasterize/tile_binning/sort/SH vs CUDA.
4. **Deep investigation** (`hierarchical_deep_compare.md`): Confirmed opacity, alpha formula, and maxContribRayPixel are identical. HEAD sort depth bias (+8.0) is neutral. Sub-tile coarseness is the remaining structural gap.

### Key findings
- **CUDA uses 3-level cascade**: TAIL(64/4×4) → MID(8/2×2) → HEAD(4/pixel). VK only has sub-tile sort → HEAD.
- **VK is systematically brighter** than CUDA (73% of pixels). Root cause: HEAD overflow blends entries prematurely when cross-batch depth ordering is incorrect.
- **Error concentration**: max_abs=0.89, concentrated in mid-image (depth complexity). Worst pixels: VK ~2.8× brighter than CUDA.
- The 17 dB gap is primarily from missing **cross-batch persistent TAIL buffer** in the rasterizer. Per-batch sub-tile sort only gives +2.3 dB.
- `new_aabb=true` (view-space AABB): VK already uses computeAABBView, matching CUDA. ✓
- `load_balancing=true`: pure performance, no PSNR impact.

### Next steps
- Implement persistent cross-batch TAIL buffer in rasterize.comp (stores global Gaussian IDs sorted by sub-tile-center depth across batches, flushes shallowest to HEAD with global memory loads)
- If TAIL alone doesn't reach 60 dB: investigate upstream pipeline (sort key precision, SH evaluation differences)

---

## Session 7 — 2026-04-19

SP-6 training gaps closed. 243/243 tests pass (non-basketball).

Four Python→C++ gaps identified and implemented:
1. **Opacity + scale regularization**: `dL/d_raw_opacity += (reg/N)*sig*(1-sig)`, `dL/d_raw_scale += (reg/N)*exp(raw_sc)` — injected after backward, before GPU Adam upload.
2. **Position noise injection**: `inject_position_noise(pos_lr)` — Sigma=L@L^T, N(0,1) noise for near-dead Gaussians (op_sigmoid(1-opacity) > 1e-6), applied after GPU Adam download.
3. **Spatial LR scale**: `pos_lr = spatial_lr_scale * lr_schedule(...)` — default 1.0 is backward-compatible; callers with COLMAP data set cameras_extent.
4. **op_sigmoid gate**: `1/(1+exp(-100*(x-0.995)))` — threshold at opacity=0.005 (not 0.995 as an early comment wrongly stated; threshold math: op_sigmoid(1-opacity) = sigmoid(-100*(opacity-0.005))).

SP-6 technical decisions:
- Regularization gradients added to CPU arrays (in-place, `+=`) after `preprocessor_bwd_`, before GPU upload — no new GPU kernels needed.
- `inject_position_noise` seeded with `step_count_` for per-step reproducibility.
- Basketball 100-step test: measured PSNR = 4.83 dB (not 10 dB as plan estimated). 10 dB requires densification + 2000 steps. Test asserts loss-decrease (0.573→0.481 = 16% decrease) instead.
- VkTrainingConfig: 4 new fields: `opacity_reg=0.01`, `scale_reg=0.01`, `noise_lr=5e5`, `spatial_lr_scale=1.0`.

Next: CB chaining (eliminate vkDeviceWaitIdle per dispatch) → 2000-step PSNR milestone.

---

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
