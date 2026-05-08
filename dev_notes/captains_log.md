# Captain's Log

## Session 20 — 2026-05-08 — L1b production eval_3D replay-order localization and cascade port

Localized the remaining basketball cam0 L1b production gap inside eval_3D contribution replay order. The prior top-level suspects are now ruled down: the same-ply golden matches `proper_ewa=1, parity_mode=0`, current CUDA render is bit-exact with the stored golden, touched Gaussian identity/tile ranges/candidate sets match, and `n_contrib`/`T_final` do not explain the highest-error pixels.

Added VK replay-order diagnostics to `VkVsCudaBasketball.Cam0_NContribDump`. For worst pixel `(453,52)`, the actual VK blended sequence is `145552 397629 166619 221332 283819 289947 125398 175573 236196 260620 218837 182148 355888 255866`. Offline replay with VK preprocess data reconstructs the VK pixel within `~5e-7` RMSE. Replaying the raw tile order swaps the near-depth high-alpha pair `182148/218837` back and reconstructs the CUDA pixel for that probe.

Two single-variable fixes were tested and rejected. Setting Vulkan `HEAD_W` from 8 to CUDA's configured 4 regressed full cam0 PSNR to **52.074 dB**. Adding a fixed `1e-6` near-depth tie tolerance to HEAD insertion regressed PSNR to **51.245 dB**. Both were reverted; the restored baseline passed at **54.730 dB**. These experiments proved the HEAD window only makes sense together with CUDA's MID/tail cull cascade, not as an isolated tuning knob.

Top30 worst-pixel replay quantification showed VK replay reconstructs VK in **30/30** pixels. Raw tile order was closer to CUDA than VK replay in **25/30**, but exactly reconstructed CUDA in only **1/30**. This meant the gap was replay-order/cascade semantics, but CUDA was not simply raw tile order.

Added `harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py` as an offline simulator. MID-only v1 top100 improved over raw (`mid_h8` mean RMSE `0.047803` vs raw `0.061830`, wins `mid_h4=30`, `mid_h8=37`, raw `31`). After adding CUDA `CULL_ALPHA=true` 4x4 frustum/alpha culling, the simulator strongly matched CUDA top-error pixels: wins `mid_h4=89/100`, mean RMSE `0.008797`, and `<1e-4` exact/near-exact reconstructions `78/100`.

Ported the production non-parity `eval_3D` path in `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` from the simplified sub-tile permutation to a CUDA-style full-tile private TAIL→MID→HEAD replay stream: 4x4 TAIL frustum/alpha culling, CUDA-like 32-input stages, a 64-entry TAIL queue draining the front 16 on overflow, `MID_WINDOW=8`, and `HEAD_WINDOW=4`. The parity raw-order path and the 2D conic path were left unchanged, and replay sideband writes remain tied to actual HEAD compositing order.

Validation opened the L1b production gate above target: build passed; `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` improved **54.730 dB → 60.696 dB**; `VkVsCudaBasketball.Cam0_NContribDump` passed; offline replay confirmed sideband reconstructs Vulkan with max RMSE **2.60e-7** and mean RMSE **6.10e-8**; `VkVsCudaFirstLoss.*` passed **12/12**; targeted eval_3D trainer/forward smoke tests passed **5/5**; targeted CTest regex passed **37/37**; initial full CTest passed **315/315** in **220.53s**.

Review/test-audit found and cleared one real blocker: the bounded TAIL/MID insertion helpers could write past private arrays when full and a worse candidate sorted after all residents. Fixed both helpers to drop that candidate instead, refreshed stale replay/binding comments, and strengthened `VulkanTrainer.Eval3DNonParityStepSmoke` to assert replay sideband count equals summed `n_contrib`. Post-fix targeted regression passed **15/15** with PSNR still **60.696 dB**; +1/+2 re-review found no blockers; final full CTest passed **315/315** in **243.58s**. Further PSNR work should isolate current residual cull/order differences per top-error pixel rather than tune constants blindly.

## Session 19 — 2026-05-07 — non-parity eval_3D replay and AAA-default training alignment

Implemented exact non-parity `eval_3D` backward replay for Vulkan training by recording the real forward HEAD flush order into replay sideband buffers and consuming that order in the eval_3D backward shader. This removes the previous `VulkanTrainer::step` fail-closed guard for `eval_3D && !parity_mode` while preserving the parity path.

A review found two real replay-safety issues after the first implementation: replay sideband storage was allocated from `FrameAllocator` without arena sizing coverage, and `replay_order_count` was stored as signed `int` after only a `uint32_t` capacity check. Fixed both by making `ForwardCache` own replay offsets/GIDs in `std::vector<uint32_t>` and storing `replay_order_count` as `size_t`; independent re-review found no blocking lifetime, sizing, or truncation findings.

Aligned formal training defaults with AAA-Gaussians expectations: `gs3d_vk_train` defaults to **30000** iterations, `lambda_dssim=0.2`, position LR `1.6e-4 -> 1.6e-6`, SH warmup `1000`, opacity/scale regularization `0.01`, noise LR `5e5`, and long densification until `25000`. The full-dataset comparison harness `--steps` default was also changed to **30000** so production runs do not silently use the earlier short 5000-step default; strict parity/stability harnesses must pass explicit zero-reg/noise/constant-LR knobs when needed.

Validation passed: build OK; direct CLI smoke `gs3d_vk_train --eval_3d 1 --parity_mode 0 --proper_ewa 1 --iterations 1` ran successfully on the 66-view basketball train split and saved a PLY; targeted eval_3D trainer tests passed **3/3**; P5-P7/L1 replay/render/loss parity gates passed **4/4**; final full CTest passed **315/315** in 210.10s.

## Session 18 — 2026-05-06 — CUDA intermediate Gate_P6-P7/L1 raster/render/loss opened

Opened `VkVsCudaFirstLoss.Gate_P6_TFinalNContrib` as a real first-loss raster-state gate. The gate compares CUDA/Vulkan `T_final` and `n_contrib` in eval_3D parity mode after validating sorted-ID range extents, sorted gid bounds, and the P5-derived sorted-ID ambiguity surface.

P6 locks both set-difference and near-depth order ambiguity tile identities, not just counts. `n_contrib` mismatches outside P5-ambiguous tiles are split into early-termination boundary drift near `T≈1e-4` versus true nontermination drift. The nontermination buckets are strict zero; total drift, outside-termination drift, max `n_contrib` delta, rare `T_final` pixel count, max `T_final` abs, and mean `T_final` abs are explicitly bounded.

Observed P6 evidence: sorted-ID ambiguity is fixed at 27 set-difference tiles and 45 near-depth order-only tiles; `n_contrib` nontermination mismatches outside ambiguity are 0; early-termination outside-ambiguity drift is 18 `>1` cases and 13 off-by-one cases; `T_final` rare drift is 14 pixels outside ambiguity with max abs `0.00194701552` and mean abs `1.79116949e-07`.

Opened `VkVsCudaFirstLoss.Gate_P7_RenderedImage` by comparing CUDA `rendered_image.npy` CHW output to `VulkanTrainer::rendered_image()` after eval_3D parity-mode `forward_only`. The gate checks artifact shape/layout, finite outputs, full-frame quality budgets, rare-pixel count budgets, and pinned first/max drift indices so localized render drift cannot silently move.

Observed P7 evidence: `max_abs=0.00301802158`, `mean_abs=4.05360525e-07`, `l2_rel=9.83802837e-06`, `PSNR=103.724446 dB`, `bad_abs_1e-5=1960`, `bad_abs_1e-4=127`, `first_bad_abs_1e-5=36386`, `first_bad_abs_1e-4=36386`, and `max_abs_idx=1516009`.

Opened `VkVsCudaFirstLoss.Gate_L1_L1Loss` by comparing CUDA `l1_loss.npy` against `VulkanTrainer::forward_only()` with `lambda_dssim=0` in eval_3D parity mode. The gate also recomputes serial-double L1 from CUDA/VK rendered images and GT to sanity-check CHW mean-absolute-error semantics, while intentionally not asserting naive serial-float recompute because the production VK path uses `compute_combined_loss_gradient` with parallel float partial reduction.

Observed L1 evidence: `vk_forward=0.0613510013`, `vk_serial_double=0.0613514965`, `cuda_golden=0.0613514706`, `cuda_serial_double=0.061351486`, `vk_cuda_abs=4.69386578e-07`, `vk_serial_double_abs=4.95205611e-07`, and `cuda_serial_double_abs=1.53396087e-08`.

Validation passed: targeted P6 passed after final tile-identity assertions; targeted P7 passed after drift-location pins; targeted L1 passed after reduction-semantics calibration; final `VkVsCudaFirstLoss` passed **12/12** with no skips in 144.22s; final full CTest passed **315/315** in 192.18s. +1 review blockers were fixed; +2 external re-review found no P6 blockers after contiguous range checks, sorted gid bounds, and exact ambiguity tile identity assertions; final +2 reviews found no P7 or L1 blockers.

After first-loss closure, the remaining L1b same-ply production gap was localized with the basketball cam0 harness. Full cam0 remains **54.730 dB**, while the table subset is **67.867 dB**, so the old table ROI is no longer the dominant gap. A four-way CLI config matrix showed the same-ply golden matches `proper_ewa=1, parity_mode=0` (`54.729887 dB`); disabling proper EWA drops to about 25 dB and enabling parity mode drops to 32.70 dB.

CUDA same-ply `n_contrib`/sort dumps show the active/touched set is already closed: both sides touch **181943** Gaussians with no one-sided IDs; tile ranges and candidate sets match exactly; touched RGB max abs is `9.54e-7`. `n_contrib` mismatches cover 12757 pixels but only explain **3.15%** of image SSE. The dominant residual is order-only sort drift: **1879** tiles have pure same-set order mismatches, **0** tiles have set mismatches, depth-key differences are only ±1 bit, and order-bad tiles cover **90.4%** of image SSE. A small clean-sort tail remains for later eval_3D raster/proper-EWA math investigation.

## Session 17 — 2026-05-06 — CUDA intermediate Gate_P5 sorted IDs opened

Opened `VkVsCudaFirstLoss.Gate_P5_SortedIds` as the next first-loss intermediate gate. The CUDA step-1 dump now writes `depths.npy` for parity-mode global sort keys and materializes `rects2D.npy`, `gauss2screen.npy`, and `aabb_debug.npy` diagnostics. VulkanTrainer intermediate capture now exposes per-Gaussian depths, anisotropic `radius_f`, and `gauss2screen` so P5 can inspect the same forward state used by bin/sort.

The gate now fails rather than silently skips when required CUDA dump artifacts are missing under an existing dump directory. It explicitly asserts `eval_3D && parity_mode`, because this path compares global `depths[i]` keys; default eval_3D tile-depth binning remains out of scope for P5 parity mode.

The sorted-ID assertion masks only one-sided `(tile,gid)` memberships adjacent to a specific near-integer CUDA/VK float-rect boundary edge. Shared memberships stay in the per-tile order comparison. After masking those boundary tile-instances, stable per-tile ID sets match exactly. The remaining 45 order-only tiles are accepted only when exact depth ties or pairwise-overlapping near-depth key windows explain the inversion; no separated-depth ordering bugs remain.

Validation passed: targeted P5 passed after final boundary-instance tightening, all `VkVsCudaFirstLoss` gates passed with expected skips for P6-P7/L1, and full CTest passed **315/315** in 113.42s. +1 review findings were fixed, and final +2 external re-review passed with no blocking findings.

## Session 16 — 2026-05-01 — CUDA intermediate Gate_P2-P4 opened

Opened `VkVsCudaFirstLoss.Gate_P2_ConicOpacity` as the next first-loss intermediate gate after P1. The first attempt compared `conic_opacity.npy` as rows of `{a,b,c,opacity}` and correctly failed; investigation of `dump_cuda_training_step.py` and the CUDA extension showed the basketball golden was generated with `eval_3D=true`, where CUDA writes opacity through `((float*)conic_opacity)[gid]`. The `[P,4]` dump is therefore a raw memory view, and only `conic_opacity.reshape(-1)[gid]` is semantically valid for this fixture.

Opened `VkVsCudaFirstLoss.Gate_P3_RgbColors` after confirming `rgb_colors.npy` is raw per-Gaussian `[N,3]` post-lower-clamp SH color. CUDA does not guarantee meaningful RGB payload for inactive Gaussians, so P3 follows the same overlap-active comparison discipline as P2.

Opened `VkVsCudaFirstLoss.Gate_P4_Radii` after the first strict equality attempt showed 364 mismatches, all with exact active-set agreement. 337 were sentinel-scale radii from the same `tan(±pi/2-epsilon)` degenerate-AABB fallback already masked by P1; the remaining finite deltas were 27 one-pixel `ceil(max_extent)` boundary cases out of 399663 finite radii, under the `max(32, ceil(0.01% of finite radii))` cap.

Validation passed: targeted P2 PASS, targeted P3 PASS, targeted P4 PASS, all `VkVsCudaFirstLoss` gates **12/12 PASS** with expected skips for P5-P7/L1, and full CTest **315/315 PASS** in 42.94s.

## Session 15 — 2026-04-30 — Full-dataset feature matrix

Exposed the next set of parity-controlled training features through `gs3d_vk_train`: DSSIM loss weight, position LR init/final, spatial LR scale, and SH warmup. Defaults preserve the prior strict baseline: L1-only, constant position LR, full SH from the first step, no regularization, no position noise.

Extended `harmonyos_3dgs/tools/compare_basketball_eval3d_proper_mcmc_full.py` into a reusable feature-matrix harness. It now writes `config.json` and `metrics.csv`, passes the new feature knobs to both CUDA and Vulkan, reports timing and quality metrics in machine-readable form, and defaults to not writing per-view render `.npy` files. The generated image `.npy` files under `harmonyos_3dgs/build/compare_runs` were cleaned.

Review found a real SH-warmup methodology issue: CUDA/training-final renders can use a lower active SH degree than a saved PLY's full degree. Added `--sh_degree` to `gs3d_vk_render`, propagated `--sh_degree` into `VkTrainingConfig::sh_degree_max`, and made the harness render saved PLYs with the final training-forward active degree. The rerun SH-warmup report records `render SH: 0`.

Feature matrix, all 76 basketball views, eval_3D/proper_ewa/MCMC, 1000 steps, `cap_max=100000`, no opacity reset: baseline PASS (`VK saved vs CUDA 28.71 dB`, GT gap `-0.33 dB`), DSSIM 0.2 PASS (`29.30 dB`, gap `-0.05 dB`), LR decay PASS (`30.71 dB`, gap `-0.04 dB`), SH warmup 1000 PASS after render-SH fix (`30.91 dB`, gap `-0.18 dB`). All runs matched final counts **3513/3513**. The DSSIM run exposed a real performance bug: 1000 Vulkan steps took **2632.7s**, far slower than the L1 baseline.

DSSIM bottleneck root cause was the C++ loss path, not Vulkan raster/Adam: `compute_combined_loss_gradient` recomputed direct 11x11 clamp-window SSIM statistics and analytical gradient accumulation on the CPU. Replaced it with separable Gaussian moment blurs plus transpose/adjoint blur accumulation, then parallelized the independent image/channel passes. The 50-step DSSIM training check dropped from **127.7s** after the separable-only rewrite to **7.2s** after parallelization; the full 1000-step DSSIM matrix rerun dropped to **167.6s** while still PASSing (`VK saved vs CUDA 28.25 dB`, counts **3513/3513**).

Validation: `python3 -m py_compile` passed, 1-view/2-step smoke passed with no render `.npy`, targeted matrix runs passed, no render `.npy` files remained under compare outputs, `DSSIM` tests passed **4/4**, targeted DSSIM/training regression passed **66/66** with expected skips, and final full CTest passed **315/315** in 40.03s. Added `DSSIM.SlowReferenceSlidingWindowEquivalence` to compare optimized loss and selected gradients against a direct clamp-window reference. Independent subagent review was performed before/after the SH-warmup fix and after the DSSIM optimization; remaining DSSIM notes are non-blocking performance/refinement follow-ups.

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
