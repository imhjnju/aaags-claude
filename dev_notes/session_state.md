# Session State

## Current Phase (S20 — 2026-05-08)
Vulkan end-to-end training gap closure now includes AAA-Gaussians MCMC densification, final-step CUDA parity, saved-PLY render-config parity, true multi-view/full-dataset parity control, first-loss Gate_P2-P7 plus L1 eval_3D parity, finite-gated split-aware comparison harnessing, exact non-parity `eval_3D` backward replay, and the L1b production eval_3D replay-order fix. The basketball cam0 same-ply production forward gate is now open above the 60 dB target after porting CUDA's `CULL_ALPHA=true` TAIL→MID→HEAD cascade semantics into Vulkan production `eval_3D`: 4x4 tail frustum/alpha culling, `MID_WINDOW=8`, and `HEAD_WINDOW=4`. Strict parity/stability harnesses must pass explicit zero-reg/noise/constant-LR knobs when needed; production full-dataset comparison defaults to 30000 steps.

## Test Counts (2026-05-08 current)
- Build: `cmake --build harmonyos_3dgs/build --target gs3d_vk_tests -j$(nproc)` OK after the production `eval_3D` CUDA cascade port in `rasterize.comp`.
- Basketball production forward gate: `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` PASSed and improved from the restored **54.730 dB** baseline to **60.696 dB**.
- Replay dump/sideband validation: `VkVsCudaBasketball.Cam0_NContribDump` PASSed; `python3 harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py --build-dir harmonyos_3dgs/build --top 100 --tail-alpha-cull` confirmed replay sideband reconstructs Vulkan output with max RMSE **2.60e-7** and mean RMSE **6.10e-8**.
- First-loss intermediate gate validation remains green: `VkVsCudaFirstLoss` **12/12 passed** with no skips; Gate_P2 validates eval_3D opacity-lane parity, Gate_P3 validates SH-evaluated RGB parity on overlap-active Gaussians, Gate_P4 validates exact active-set radii parity with sentinel-scale radii masked, Gate_P5 validates sorted-ID parity, Gate_P6 validates `T_final`/`n_contrib`, Gate_P7 validates rendered-image parity at **103.72 dB**, and Gate_L1 validates scalar loss parity with VK/CUDA abs diff **4.69e-7**.
- Targeted eval_3D trainer/forward smoke coverage passed **5/5**: `ForwardPipeline.FullChain_TinyFixture_Eval3D`, `VulkanTrainer.Eval3DOneStepSmoke`, `VulkanTrainer.Eval3DNonParityStepSmoke`, `VulkanTrainer.Eval3DStep1RawGradientParity`, and `McmcDensify.VulkanTrainerSupportsEval3DMcmcWithFilterPropagation`.
- Targeted CTest regression passed **37/37** for `VkVsCudaBasketball|VkVsCudaFirstLoss|VulkanTrainer|ForwardPipeline|RasterizePass|TileBinnerVulkan|McmcDensify`.
- Initial full CTest after the cascade port passed **315/315 tests** (`ctest --test-dir harmonyos_3dgs/build --output-on-failure`, 220.53s; expected inventory is 66 .cpp files).
- +1/+2 review found and cleared one blocking shader safety issue in bounded TAIL/MID insertion; post-fix targeted regression passed **15/15** with PSNR still **60.696 dB**, +1/+2 re-review found no blockers, and final full CTest passed **315/315** in **243.58s**.
- Added/retained MCMC/filter/full-dataset regression coverage and harnessing: CPU relocation, MCMC relocate/add/densify behavior, CUDA-golden MCMC replay fixtures, VulkanTrainer MCMC integration, Adam-state preservation/zeroing, opacity reset raw value + moment reset, eval_3D+MCMC filter propagation, legacy split child `filter_3D` inheritance, explicit Vulkan view schedules, strict all-GT validation, saved-PLY-vs-training-final render checks, DSSIM slow-reference clamp-window equivalence, and finite-gated split-aware metrics.
- Full-dataset basketball 1000-step validation remains the latest completed quality comparison: all 76 views, eval_3D/proper_ewa, `cap_max=30000`, shared schedule, no opacity reset. CUDA/VK final counts matched **3513/3513**; `CUDA vs GT 17.22 dB`, `VK saved vs GT 16.92 dB`, `VK saved vs CUDA 27.99 dB`, `VK saved vs train final view 59.24 dB`, GT gap **0.30 dB**, gate status PASS.
- Full-dataset feature matrix at 1000 steps with `cap_max=100000` and no opacity reset PASSed all gates: baseline `VK saved vs CUDA 28.71 dB`, DSSIM 0.2 `29.30 dB`, LR decay `30.71 dB`, SH warmup 1000 `30.91 dB`; optimized DSSIM reduced the 1000-step VK train time from **2632.7s** to **167.6s** while keeping the matrix gate PASS (`VK saved vs CUDA 28.25 dB`, counts **3513/3513**).
- Review/test-audit: +1/+2 review completed for the rasterizer shader/GPU test changes; the blocking bounded-queue insertion finding was fixed, and +1/+2 re-review found no remaining blockers.

## Parity Ladder (L1-L5)
| Level | Meaning | Status |
|-------|---------|--------|
| L1a | CPU↔VK same-ply forward | ✅ 99.15 dB (Phase A close) |
| L1b | VK↔CUDA same-ply forward (basketball cam0) | ✅ **60.696 dB** after porting CUDA eval_3D `CULL_ALPHA=true` TAIL→MID→HEAD cascade semantics into Vulkan production replay. Gate_I1-I4 + P1-P7 + L1 remain PASS; remaining residual is much smaller replay/cull/order drift, not active-set/count/color/preprocess |
| L2 | Backward gradient parity | ✅ All 5 groups bit-exact at L2 norm + sub-1e-4 per-element except gpos atomic noise |
| L3 | Post-Adam param parity | ✅ All groups (after SH layout bug fix); basketball 10/100-step now also captures Adam m/v |
| L4 | 100-step trajectory | IN PROGRESS — loss aligned, but scale/rotation gradient+moment drift grows over 100 steps |
| L5 | Independent-train final-eval | IN PROGRESS — camera-loader bug fixed; next rerun true 1000/2000 after scale/rotation drift is triaged |

## Milestones
| Milestone | Status | Sessions | Summary |
|-----------|--------|----------|---------|
| SP-0: CUDA golden infra | DONE | — | CPU reference + FD test harness |
| SP-1: Vulkan infra | DONE | S2 | VulkanContext/Buffer/Shader/Pipeline |
| SP-2: Vulkan forward pipeline | DONE | S2 | preprocess.comp + sort + rasterize.comp |
| SP-3: Vulkan backward pipeline | DONE | S3 | rasterize_backward.comp + preprocess_backward.comp |
| SP-4: Training integration | DONE | S4 | ForwardCache, GPU Adam, VulkanTrainer |
| SP-5: GPU optimizer + hyperparams | DONE | S5 | GPU Adam, LR+SH schedules, DSSIM, MCMC, basketball E2E smoke |
| SP-6: Training gaps closed | DONE | S7 | T1-T5; 243 tests + basketball loss-decrease |
| VK-CUDA L1b parity (PSNR≥60dB) | GATE OPEN / REVIEW CLEARED | S8-S20 | 25.3→60.696 dB. Phase 1 (Gate_I1-I4), Gate_P1-P7, Gate_L1, production eval_3D cascade replay, +1/+2 review, and final full CTest are done; remaining work is optional residual/performance triage |
| L2 backward parity | DONE | S11 | SH layout bug fix + test sensitivity tightening; 1 atomic noise debt |
| L5 independent-train comparison | TODO | — | No test exists; Phase E |
| M0: Foundation | IN PROGRESS | — | Interleaved with Vulkan migration |

## L1b Gate Status (Phase C of plan)
| Gate | Status | Notes |
|------|--------|-------|
| Gate_I1 ViewMatrix | PASS | max_abs 7.45e-9 |
| Gate_I2 ProjMatrix | PASS | max_abs 1.19e-7 |
| Gate_I3 RawParams | PASS | max_abs 4.77e-7 (opacities) |
| Gate_I4 ConfigFlags | PASS | parity_mode=1 |
| Gate_P1 Means2D | PASS (S11) | mask CUDA huge fallback + relative tol; 5903 only-CUDA-rasterizes closed by dilation gating fix |
| Gate_P2 ConicOpacity | PASS (S16) | validates eval_3D opacity-lane parity on overlap-active Gaussians; CUDA dump stores opacity as flat `((float*)conic_opacity)[gid]`, with single-sided active counts reported for P4 |
| Gate_P3 RgbColors | PASS (S16) | validates SH-evaluated RGB parity on overlap-active Gaussians; inactive CUDA RGB rows are not semantically stable, with single-sided active counts reported for P4 |
| Gate_P4 Radii | PASS (S16) | exact active-set match; sentinel-scale `tan(±pi/2-epsilon)` radii masked; finite ceil-boundary off-by-one deltas 27/399663 under the `max(32, ceil(0.01% of finite radii))` cap |
| Gate_P5 SortedIds | PASS (S17) | parity-mode global-depth sorted-ID sets match after masking only boundary-adjacent one-sided tile instances; remaining 45 order-only tiles are exact/near depth-key swaps, with no separated-depth ordering bugs |
| Gate_P6 TFinalNContrib | PASS (S18) | validates eval_3D parity-mode raster state: sorted-ID set/order ambiguity budgets are locked to P5 evidence; `n_contrib` nontermination mismatches are zero; early-termination drift near `T≈1e-4` and rare `T_final` drift are explicitly bounded |
| Gate_P7 RenderedImage | PASS (S18) | compares CUDA `rendered_image.npy` CHW output to `VulkanTrainer::rendered_image()` after eval_3D parity-mode `forward_only`; current VK↔CUDA render parity is **103.72 dB**, with `max_abs=0.00301802158`, `mean_abs=4.05360525e-07`, `l2_rel=9.83802837e-06`, and rare drift location/count budgets pinned |
| Gate_L1 L1Loss | PASS (S18) | compares CUDA `l1_loss.npy` scalar to `VulkanTrainer::forward_only()` loss in eval_3D parity mode; current `vk_cuda_abs=4.69386578e-07`, with CUDA rendered/GT serial-double sanity `1.53396087e-08` and VK parallel-reduction-vs-serial-double delta bounded |

## Python → C++ Gaps Closed (SP-6)
1. **Opacity reg**: `dL/d_raw_opacity += (0.01/N)*sig*(1-sig)` — after backward, before Adam upload
2. **Scale reg**: `dL/d_raw_scale += (0.01/N)*exp(raw_sc)` — per-component
3. **Position noise**: `Sigma @ N(0,1) * op_sigmoid(1-opacity) * noise_lr * pos_lr` — after Adam download
4. **Spatial LR**: `pos_lr = spatial_lr_scale * lr_schedule(...)` — default scale=1.0

## Latest Sessions

### S20 — 2026-05-08 — L1b production eval_3D replay-order localization and cascade port
- Reframed the remaining basketball cam0 L1b gap after new diagnostics: top-level config, CUDA golden freshness, touched set, tile ranges, candidate sets, RGB, `n_contrib`, and `T_final` are closed enough; the dominant remaining difference is the production eval_3D order in which valid contributions are blended.
- Added/used VK replay-order dumps for `VkVsCudaBasketball.Cam0_NContribDump`. Worst pixel `(453,52)` has VK replay count 14 and GIDs `145552 397629 166619 221332 283819 289947 125398 175573 236196 260620 218837 182148 355888 255866`. Offline replay with VK preprocess data reconstructs the VK pixel from this sequence within `~5e-7` RMSE; raw tile order swaps `182148/218837` back and reconstructs the CUDA pixel for that probe.
- Negative experiments: `HEAD_W=8→4` alone regressed basketball cam0 PSNR to **52.074 dB** and was reverted; a fixed near-depth HEAD tie tolerance (`id + 1e-6 < depth`) regressed PSNR to **51.245 dB** and was reverted. These proved the fix needed CUDA's full cascade semantics rather than isolated constant tuning.
- Added `harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py`. MID-only v1 improved over raw order, and adding CUDA `CULL_ALPHA=true` 4x4 tail frustum/alpha culling made `mid_h4` win **89/100** top-error pixels with mean CUDA-color RMSE **0.008797** and **78/100** exact/near-exact reconstructions.
- Ported the production non-parity `eval_3D` replay path in `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` to CUDA-style full-tile private TAIL→MID→HEAD replay: 4x4 TAIL frustum/alpha culling, CUDA-like 32-input stages, 64-entry TAIL queue draining front 16 on overflow, `MID_WINDOW=8`, and `HEAD_WINDOW=4`. The parity raw-order path and the 2D conic path were left unchanged; replay sideband still records only actual HEAD compositing order.
- Validation after the port: build OK; `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` improved **54.730 dB → 60.696 dB**; `VkVsCudaBasketball.Cam0_NContribDump` PASSed; offline replay confirmed sideband reconstructs Vulkan with max RMSE **2.60e-7** and mean RMSE **6.10e-8**; `VkVsCudaFirstLoss.*` PASSed **12/12**; targeted eval_3D trainer/forward smoke tests PASSed **5/5**; targeted CTest regex PASSed **37/37**; initial full CTest PASSed **315/315** in **220.53s**.
- Review/test-audit: +1/+2 review found one blocking shader safety issue in bounded TAIL/MID insertion when queues were full and a worse candidate sorted after all residents. Fixed both helpers to drop that candidate rather than write past the private array, refreshed stale replay/binding comments, and strengthened `VulkanTrainer.Eval3DNonParityStepSmoke` to assert replay sideband count equals summed `n_contrib`. Post-fix targeted regression PASSed **15/15** with PSNR still **60.696 dB**; +1/+2 re-review found no blockers; final full CTest PASSed **315/315** in **243.58s**.
- Further PSNR work should compare current shader replay against the offline simulator per top-error pixel and isolate cull/order differences one layer at a time; do not tune constants blindly.

### S19 — 2026-05-07 — non-parity eval_3D replay and AAA-default training alignment
- Implemented exact non-parity `eval_3D` backward replay: forward rasterize now materializes per-pixel blended Gaussian IDs in the actual HEAD flush order, and eval_3D backward consumes that replay order instead of raw tile-sorted IDs when `parity_mode=0`.
- Fixed two post-review replay safety issues: replay offsets/GIDs are now owned by `ForwardCache` vectors rather than allocated from `FrameAllocator`, and `replay_order_count` is `size_t` so the prior signed truncation risk is gone. Independent re-review found no blocking lifetime, sizing, or count findings.
- Confirmed Vulkan formal training defaults follow AAA-Gaussians: 30000 iterations, DSSIM 0.2, position LR decay to 1.6e-6, SH warmup 1000, opacity/scale regularization 0.01, noise LR 5e5, and densify-until 25000. The full-dataset comparison harness `--steps` default is also 30000; parity/stable runs must explicitly override zero-reg/noise/constant-LR settings.
- Validation: build OK; direct non-parity CLI smoke passed and saved a PLY; targeted eval_3D trainer tests **3/3 passed**; P5-P7/L1 parity gates **4/4 passed**; full CTest **315/315 passed** in 210.10s.

### S18 — 2026-05-06 — CUDA intermediate Gate_P6-P7/L1 raster/render/loss opened
- Opened `VkVsCudaFirstLoss.Gate_P6_TFinalNContrib` as a real eval_3D parity-mode raster-state gate comparing CUDA/Vulkan `T_final` and `n_contrib`.
- P6 now validates CUDA/Vulkan sorted range extents and sorted gid bounds, locks P5-derived set/order ambiguity tile identities, and splits `n_contrib` mismatches into P5-ambiguous tiles, early-termination boundary drift near `T≈1e-4`, and true nontermination drift.
- Current P6 evidence: ambiguity tiles are fixed at 27 set-difference and 45 near-depth order-only tiles; nontermination `n_contrib` mismatches outside ambiguity are 0; outside-ambiguity early-termination drift is bounded at 18 `>1` cases and 13 off-by-one cases; rare `T_final` drift is bounded at 14 pixels, max abs `0.00194701552`, mean abs `1.79116949e-07`.
- Opened `VkVsCudaFirstLoss.Gate_P7_RenderedImage` by comparing CUDA `rendered_image.npy` CHW output to `VulkanTrainer::rendered_image()` after eval_3D parity-mode `forward_only`. Current evidence: `max_abs=0.00301802158`, `mean_abs=4.05360525e-07`, `l2_rel=9.83802837e-06`, `PSNR=103.724446 dB`, `bad_abs_1e-5=1960`, `bad_abs_1e-4=127`, with first/max drift indices pinned.
- Opened `VkVsCudaFirstLoss.Gate_L1_L1Loss` by comparing CUDA `l1_loss.npy` to `VulkanTrainer::forward_only()` loss with `lambda_dssim=0`; the gate also recomputes serial-double L1 from CUDA/VK rendered images and GT to sanity-check artifact semantics while respecting VK's production parallel float partial reduction.
- Current L1 evidence: `vk_forward=0.0613510013`, `cuda_golden=0.0613514706`, `vk_cuda_abs=4.69386578e-07`, `vk_serial_double_abs=4.95205611e-07`, and `cuda_serial_double_abs=1.53396087e-08`.
- Validation: targeted P6 PASS; targeted P7 PASS; targeted L1 PASS; final `VkVsCudaFirstLoss` **12/12 passed** with no skips in 144.22s; final full CTest **315/315 passed** in 192.18s. +1/+2 review blockers were fixed or cleared; final +2 review found no L1 blockers.
- Remaining L1b same-ply production gap was localized after first-loss closure: `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline/TableSubset/NContribDump` all PASS, full cam0 is **54.730 dB** while table subset is **67.867 dB**. Config matrix proves the golden matches `proper_ewa=1, parity_mode=0` (`54.729887 dB`); `proper_ewa=0` drops to ~25 dB and `parity_mode=1` drops to 32.70 dB. CUDA/VK touched Gaussian identity matches exactly (**181943/181943**, no one-sided touched IDs), RGB on touched Gaussians is tight (`max_abs=9.54e-7`), tile ranges and candidate sets match exactly, and `n_contrib` mismatches explain only **3.15%** of image SSE. The main residual is pure same-set sort order drift: 1879 tiles have order-only mismatches, 0 have set mismatches, depth-key bit deltas are only ±1, and order-bad tiles cover **90.4%** of image SSE. A small clean-sort tail remains (e.g. tile 163) for later eval_3D raster/proper-EWA math triage.

### S17 — 2026-05-06 — CUDA intermediate Gate_P5 sorted IDs opened
- Opened `VkVsCudaFirstLoss.Gate_P5_SortedIds` as a real parity gate. The CUDA dump now includes `depths.npy`, `rects2D.npy`, `gauss2screen.npy`, and `aabb_debug.npy` diagnostics; VulkanTrainer intermediate capture now exposes depths, `radius_f`, and `gauss2screen` for first-loss parity checks.
- P5 compares parity-mode global depth keys (`eval_3D && parity_mode`, not default eval_3D tile-depth binning). Required CUDA artifacts under an existing dump directory now fail with `ASSERT_TRUE` instead of silently skipping the gate.
- The sorted-ID check masks only one-sided `(tile,gid)` memberships adjacent to the specific near-integer CUDA/VK float-rect boundary edge. Shared memberships remain in the sorted-ID comparison. Stable per-tile sets match exactly after this mask; remaining order-only differences are accepted only when exact depth ties or pairwise-overlapping near-depth key windows explain the inversion.
- Validation: targeted `VkVsCudaFirstLoss` **12/12 passed** with expected skips for P6-P7/L1, targeted P5 passed after final boundary-instance tightening, and full CTest **315/315 passed** in 113.42s. +1 review findings were fixed; final +2 external re-review passed with no blocking findings.

### S16 — 2026-05-01 — CUDA intermediate Gate_P2-P4 opened
- Opened `VkVsCudaFirstLoss.Gate_P2_ConicOpacity` as a real parity gate. The CUDA basketball step-1 dump is `eval_3D=true`, so `conic_opacity.npy` is a raw `[P,4]` memory view but only the flat opacity lane `conic_opacity.reshape(-1)[gid]` is semantically valid; the 2D `{a,b,c,opacity}` row interpretation would be wrong for this fixture.
- Opened `VkVsCudaFirstLoss.Gate_P3_RgbColors` as a real parity gate. `rgb_colors.npy` is per-Gaussian `[N,3]` post-lower-clamp SH color, but inactive CUDA rows are not semantically stable, so the gate compares only overlap-active Gaussians.
- Opened `VkVsCudaFirstLoss.Gate_P4_Radii` as a real parity gate. It requires exact active-set agreement, masks sentinel-scale radii from the same `tan(±pi/2-epsilon)` degenerate-AABB fallback already masked by P1, and allows only bounded finite `ceil(max_extent)` off-by-one boundary noise (27/399663, cap `max(32, ceil(0.01%))` = 40).
- Validation: `VkVsCudaFirstLoss.Gate_P2_ConicOpacity` PASS; `VkVsCudaFirstLoss.Gate_P3_RgbColors` PASS; `VkVsCudaFirstLoss.Gate_P4_Radii` PASS; all `VkVsCudaFirstLoss` gates **12/12 passed** with expected skips for P5-P7/L1; full CTest **315/315 passed** in 42.94s.

### S15 — 2026-04-30 — Full-dataset feature matrix
- Extended `gs3d_vk_train` parity controls for full-dataset experiments: `--lambda_dssim`, position LR init/final, spatial LR scale, and SH warmup. Defaults preserve the previous strict L1/constant-LR/full-SH baseline.
- Extended the full-dataset basketball harness to write `config.json` and `metrics.csv`, expose the same feature knobs on CUDA and Vulkan, and default to not writing per-view render `.npy` files. Generated image `.npy` outputs under `build/compare_runs` were cleaned; remaining retained artifacts are PPM/PNG/report/config/CSV/PLY and renderer workdirs.
- Added `--sh_degree` to `gs3d_vk_render` and made the feature harness render saved PLYs with the final training-forward active SH degree, fixing the SH warmup saved-render comparability issue found during review. `gs3d_vk_train` now also propagates the CLI render SH degree into `VkTrainingConfig::sh_degree_max`.
- Verification: feature matrix 1000-step full-dataset runs all PASSed with matched counts **3513/3513**. Baseline `VK saved vs CUDA 28.71 dB`, DSSIM 0.2 `29.30 dB`, LR decay `30.71 dB`, SH warmup 1000 `30.91 dB` after render-SH override. Full CTest after the final fixes passed **314/314** in 40.51s. Independent review flagged DSSIM timing as a performance follow-up, not a parity blocker.

### S14 — 2026-04-29 — Vulkan MCMC densification port
- Ported AAA-Gaussians MCMC densification from the `densification` worktree: relocation/add/growth to `min(cap_max, int(1.05*N))`, CUDA-golden replay fixtures, and deterministic test sample plans.
- Extended Vulkan training config with `cap_max` and `opacity_reset_interval`. `cap_max > 0` enables MCMC; `cap_max <= 0` keeps the legacy clone/split/prune path. `vk_train_main` keeps densification disabled by default for parity-safe CLI behavior.
- Extended `VulkanAdam` with state-preserving `extend_group`, `shrink_group`, `zero_moment_floats`, and `download_group_moments`; `VulkanTrainer` now preserves untouched MCMC Adam slots and zeros modified/replaced slots across all six parameter groups.
- Added opacity reset scheduling to Vulkan training using raw `logit(0.01)` and zeroing opacity Adam moments. Implemented `filter_3D` propagation through `OwnedRawParams`, legacy clone/split, DensityController clone/split, MCMC relocation/add, and VulkanTrainer eval_3D+MCMC.
- Verification: build OK; targeted MCMC/Adam/densification/training/config/filter tests **54/54 PASS**; targeted final-step/trainer/densification parity tests **62/62 PASS**; parity-sensitive regression **42/42 PASS** with expected skips; post-review full-dataset/trainer regression **52/52 PASS**; full CTest **314/314 PASS** after final full-dataset review fixes. 5000-step basketball 2D MCMC with `cap_max=50000` and no opacity reset reached CUDA/VK final N **24680/24680**; with final-step skip and `--proper_ewa 0`, saved-PLY standalone render matches the VK training-final render at **74.68 dB**. 5000-step basketball eval_3D/proper_ewa MCMC with `cap_max=30000` reached CUDA/VK final N **24680/24680**, `CUDA vs GT 27.13 dB`, `VK saved vs GT 27.25 dB`, `VK saved vs CUDA 29.37 dB`, and `VK saved vs train 59.04 dB`. Full-dataset basketball eval_3D/proper_ewa MCMC validation over all 76 views at 1000 steps with a shared explicit schedule reached CUDA/VK final N **3513/3513**, `CUDA vs GT 17.22 dB`, `VK saved vs GT 16.92 dB`, `VK saved vs CUDA 27.99 dB`, `VK saved vs train final view 59.24 dB`, and PASSed the count/PSNR gates. Independent subagent reviews found no blockers after the final-step/render-config/full-dataset pass; latest full-dataset review fixes added strict missing-GT handling, nested `img_name` PPM handling, CUDA/VK densification upper-bound alignment, saved-vs-train reporting, and explicit gate failures.

### S13 — 2026-04-28 — scan1 n_contrib replay-boundary fix + eval_3D backward smoke
- Targeted scan1 camera 0 at 1600×1200 with 28,747 Gaussians and measured VK↔CUDA independent-training drift at 10/200/1000 steps.
- Root cause: 2D `n_contrib` had count-based semantics in VK/CPU, but CUDA stores the 1-based position of the last candidate that actually blended. Fixed Vulkan forward/backward and CPU 2D reference to use the same replay-boundary contract.
- Verification: `RasterizerBackwardVulkan.MatchesCPU_TinyFixture` PASS. scan1 VK↔CUDA final-render PSNR improved to **107.14 dB / 49.82 dB / 27.77 dB** at 10/200/1000 steps. Full record: `dev_notes/scan1_n_contrib_parity_s13.md`.
- Eval_3D smoke correction: the first `gs3d_vk_train --eval_3d 1` run was not a true eval_3D trainer because the CLI only set `RenderConfig::eval_3D`, not `VkTrainingConfig::eval_3D` (the trainer constructor specialization source). After propagation, true eval_3D training reached the existing backward unsupported guard while forward eval_3D rendering was confirmed non-empty.
- Eval_3D backward enablement: added separate Vulkan eval_3D rasterizer/preprocessor backward passes, preserved the fixed 2D replay path, removed the CLI fail-fast guard, and added `VulkanTrainer.Eval3DOneStepSmoke`, `VulkanTrainer.Eval3DStepRequiresParityMode`, and `VulkanTrainer.Eval3DStep1RawGradientParity`. Two eval_3D parity bugs were fixed: parity-mode forward now uses CUDA-style direct blend/replay-boundary semantics, and preprocessor backward reads `d_gauss2screen` with the correct transposed storage interpretation. The parity test now also directly compares eval_3D raster backward `d_rgb` (`l2_rel=1.76e-3`, max_abs `1.09e-5`), `d_opacity` (`l2_rel=1.52e-4`, max_abs `1.82e-6`), and `d_gauss2screen` (`l2_rel=5.63e-4`, max_abs `2.65e-5`) against CUDA after adding a non-breaking CUDA dump binding. Non-parity eval_3D training now fails explicitly until default HEAD/sub-tile backward replay is implemented. A suspected missing SH→position gradient was investigated and intentionally not ported: CUDA `preprocessCUDA_3D` calls `computeColorFromSH`, but `computeGauss2Screen*` later assigns `dL_dmean[idx] = ...`, overwriting that SH mean contribution; adding it in Vulkan worsened pos parity from `1.16e-3` to `9.52e-2` L2-relative. The eval_3D rasterizer's per-subtile permutation was packed from 32-bit slots to four 8-bit batch indices per word, reducing the conservative shared-memory footprint to about 31.1 KiB while preserving focused forward/trainer parity. The CLI now maps `--eval_3d 1` training onto the validated `parity_mode=true` backward replay, so scan1 camera0 eval_3D 10-step training with the filtered PLY runs end-to-end (`loss 0.665102 → 0.660058`, final render PSNR vs JPEG-converted GT `2.664990 dB`, artifact `/tmp/scan1_vk_eval3d_filter_10_postbwd/`). Full CTest is now **276/276 PASS** after these changes.

### S12 — 2026-04-28 — fair-path alignment + rotation-drift triage
- Fixed fair-path mismatches: `VulkanTrainer` honors `sh_degree_warmup=0` from the first forward pass, `vk_train_main.cpp` projection now matches CUDA/`camera_utils.cpp`, and `model.free()` no longer precedes trainer construction. Updated 10/100-step CUDA dumpers to use full SH for all steps and regenerated goldens.
- Verification: focused build OK; `VkVsCudaBasketball10Step.PerStepParity` PASS; `VkVsCudaBasketball100Step.PerStepParity` PASS. Fair 10-step comparison is now aligned (`VK vs CUDA 43.14 dB`), while fair 200-step still diverges (`CUDA 14.65 dB`, `VK 11.73 dB`).
- Investigated remaining scale/rotation drift. The conic off-diagonal convention is paired (`rasterize_backward.comp` full `d_conics[1]` with preprocess full-parameter chain) and must not be changed alone. Scalarizing `preprocess_backward.comp` `W/J/T/Vrk/VT` ruled out a GLSL `mat3` layout bug as the primary cause: step100 drift changed only marginally (`g_sca≈4.43e-1`, `g_rot≈7.49e-1`).
- Added exact 100-step backward-chain diagnostics. Important correction: CUDA `diag_pre_d_qn` must use `_C.rasterize_gaussians_backward_dump`'s returned `d_rotations`; the earlier NumPy reconstruction hid CUDA's step-1 sub-ULP residual by producing exact zero. With actual kernel outputs, the first causal split is step1 `d_qn/g_rot`: VK exactly zero vs CUDA ~`1e-11`, which Adam `eps=1e-15` converts into ~`1e-3` raw-rotation updates. Treat this as numerical-cancellation amplification, not a proven Vulkan math bug; next experiment should neutralize the near-zero rotation-gradient cusp before chasing scale-chain drift.

### S11 — 2026-04-25 — Phase A + C.0 + D.deep SH layout closure
- **Phase A — proper_ewa default flip**: `PreprocessorVulkan` ctor default flipped from `false` → `true`. Production paths (CPU↔VK comparison, render) take AAA path; parity-harness tests opt out explicitly. Result: `VkVsCpuRender.FullFramePSNR` 28.75 → **99.15 dB** (CPU↔VK closed).
- **C.0 — Gate_P1_Means2D closure**: gated `opacity_3d *= dilation_factor` on `spec_proper_ewa` in `preprocess.comp` to match CUDA `forward.cu:157`. Wired `test_vk_vs_cuda_basketball.cpp` 3 sites with `proper_ewa=true` (matches its golden's actual generation config). Test logic: mask CUDA `tan(±π/2-ε)` degenerate fallback (`|m2d|>1e5`) + relative tolerance `max(1e-2 px, 1e-3·|cuda|)`. Result: `Gate_P1_Means2D` FAIL (8790 bad) → **PASS (0 bad)**.
- **D.deep — SH Adam-group layout bug**: 3-step trajectory analysis revealed deterministic 469% rel_diff on G[2] post-Adam SH (NOT atomicAdd noise — fully reproducible across runs). Root cause: `vulkan_trainer.cpp` was `memcpy`-splitting the unified `[N,K,3]` interleaved CPU buffer by float-index between DC `[N,3]` and REST `[N,K-1,3]` GPU groups. With K=16, first N\*3=60 floats are NOT all DCs (they're G[0]'s entire 48 SH + G[1]'s first 4); G[2..N-1]'s DCs landed in REST with **wrong lr (1/20 of correct)**. Fix: `sh_gather_dc/rest` + `sh_scatter_dc/rest` helpers + 5 call sites (ctor, reset_for_oracle, step gradient upload, step post-Adam download, reallocate_for_n). Verification: 3-step SH max-elem rel_diff **469% → 0.015%** (~3000× tighter); step-2 loss rel_diff **1.10e-3 → 5.5e-7** (2000× tighter); 10-step trajectory all groups bounded < 1e-3.
- **Latent-bug alignment (Python ref → CUDA)**: 2 algorithmic differences fixed in Python reference proactively, both inactive on tiny+basketball but would activate on edge-case fixtures: (a) `det.clamp(min=1e-10)` removed (CUDA does `1.f/det`, visibility gated upstream), (b) frustum 1.3× clamp added on `tx/tz, ty/tz` before computing J Jacobian (matches `forward_common.h:81-86`). 1114 goldens regenerated, 937 perturbed at ≤2.4e-7. Basketball cam0 PSNR unchanged at 54.84 dB.
- **Test infrastructure tightening**: `Step1GradientAndLoss` L2-norm threshold 5% → 1e-3, new per-element max rel_diff assertion at 1e-4, print format `%.4f` → `%.6e`, `abs_diff` column added. Two new permanent regression sentinels `Step3PostAdamSHParity` (6 EXPECT_LT) and `Step10TrajectoryAllGroups` (21 EXPECT_LT).
- **Auto-memory updates**: `memory/sh_adam_group_layout.md`, `memory/feedback_l2_strict_gradient_parity.md`, `memory/gotchas.md` (+ SH layout, +don't-default-blame-atomicAdd, +new_aabb).
- **Outstanding**: gpos per-element 1.89e-4 atomicAdd debt (sub-ULP abs 1.5e-9; `TODO(deterministic-backward)` in `rasterize_backward.comp`). Decision: defer; Adam smooths it; trajectory bounded 10 steps.
- **Commits today (this branch)**: `d8c388f` (Phase A + Gate_P1 + SH fix bundled), `c58f4eb` (test sensitivity), `3dcfb04` (Python ref + goldens + audit instrumentation).

### S10 — 2026-04-25 — merge master → worktree-training
- Merged `master` into `worktree-training` (merge commit `5ee56bc`, no-ff). Three master commits imported:
  - `09217bb` fix(preprocess-2d-cpu): align proper_ewa_scaling + tight_opacity_bounding with VK/CUDA — CPU 2D forward backport (28.75 → 99.15 dB CPU↔VK)
  - `612c125` fix(preprocess-backward): add `h_conv_scaling` chain rule to CPU and VK backward — `d_opacities_2d → d_cov2D` through `h_conv = sqrt(det_orig/det_dilated)` (restores `PreprocessorBackward.CovChain_RotationGradient`, `PreprocessorBackwardVulkan.MatchesCPU_TinyGolden`, `BackwardPipeline.FullChain_TinyFixture`, `CpuVkCompare.SingleStepConsistency`, `VkVsCpuRender.FullFramePSNR`)
  - `ccd093e` Merge branch 'worktree-white-table': CPU 2D forward + CPU/VK backward alignment (combined description)
- Files touched by merge: `src/cpu/preprocessor_cpu.cpp`, `src/cpu/preprocessor_backward_cpu.cpp`, `src/vulkan/shaders/preprocess_backward.comp`. **Disjoint** from in-progress S10 first-loss-parity work (VK forward `preprocess.comp`, `vulkan_trainer.cpp`, tests, `tools/*reference*.py`) — no merge conflicts, no working-tree disruption.
- Build verified post-merge: full `cmake --build build` green (gs3d_core, gs3d_vk_core, gs3d_tests, gs3d_vk_tests, gs3d_train, gs3d_vk_train, gs3d_vk_render, etc. all rebuilt and linked).
- Test count after merge: not re-counted in this session; pre-merge master claimed 249/257 (97%). In-progress training-parity test edits in this worktree are still uncommitted, so a clean test-run number will only be meaningful once those edits land.
- All 20 modified + many untracked files from S10 first-loss-parity work preserved exactly as before merge.

### S9 — 2026-04-23
- Active work: VK vs CUDA training first-loss parity harness. Phase 0+1 landed (commit `5366d38`). See `harmonyos_3dgs/dev_notes/vk_cuda_first_loss_parity_plan.md` and `vk_initial_loss_mismatch_s10.md`.
- Cascade equivalence harness for CUDA↔VK 3-level sort planned. See `memory/cascade_equivalence_harness.md`.

### S8 — 2026-04-20
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
