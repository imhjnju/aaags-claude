# Session State

## Current Phase (S16 — 2026-05-01)
Vulkan end-to-end training gap closure now includes the AAA-Gaussians MCMC densification path, final-step CUDA parity for CLI training, render-config parity for saved-PLY comparisons, true multi-view/full-dataset parity control, the first full-dataset feature-matrix harness for DSSIM, position LR decay, SH warmup, and first-loss intermediate Gate_P2 eval_3D opacity parity. `VkTrainingConfig::cap_max > 0` selects MCMC relocate/add densification; `cap_max <= 0` preserves legacy clone/split/prune. `gs3d_vk_train` now skips Adam on the terminal iteration to match the CUDA harness final forward/backward/no-step contract, exposes `--proper_ewa 0|1`, accepts an explicit `--view_schedule`, and can expose parity-controlled `--lambda_dssim`, position LR schedule, spatial LR scale, and SH warmup. Full-dataset comparisons should use `--require_all_gt 1` to prevent silent camera-list drift from missing GT images; mixed-resolution multi-view training is currently rejected because `VulkanTrainer` image buffers are sized from the first view.

## Test Counts (2026-05-01 current)
- Build: `cmake -S harmonyos_3dgs -B harmonyos_3dgs/build -DBUILD_TESTS=ON && cmake --build harmonyos_3dgs/build` OK after final-step no-update mode, train/render `--proper_ewa`, MCMC densification, `filter_3D` propagation, full-dataset view-schedule CLI changes, feature-matrix CLI knobs, and render `--sh_degree` override.
- Targeted MCMC/Adam/densification/training/config/relocation/filter tests: **54/54 passed** (`ctest --test-dir harmonyos_3dgs/build --output-on-failure -R "Mcmc|Densification|VulkanAdam|DensityController|TrainTypes|OwnedRawParams"`, 2.90s).
- Parity-sensitive regression: **42/42 passed** with expected skips (`ctest --test-dir harmonyos_3dgs/build --output-on-failure -R "VkVsPyReference|VkVsCuda|Basketball|Eval3D|TileBinner"`, 15.79s).
- Targeted post-review full-dataset/trainer regression: **52/52 passed** (`ctest --test-dir harmonyos_3dgs/build --output-on-failure -R "TrainingStepVk|VulkanTrainer|VkVsCuda|Mcmc|Densification|TrainTypes"`).
- Full CTest after feature-matrix, render-SH, DSSIM optimization, and Gate_P2 opening: **315/315 tests passed** (`ctest --test-dir harmonyos_3dgs/build --output-on-failure`, latest 40.87s; expected inventory is 66 .cpp files).
- DSSIM optimization validation added slow-reference sliding-window equivalence coverage; targeted DSSIM tests are **4/4 passed** after adding `DSSIM.SlowReferenceSlidingWindowEquivalence`, and targeted DSSIM/training regression is **66/66 passed** with expected skips (`ctest --test-dir harmonyos_3dgs/build --output-on-failure -R "DSSIM|TrainingStepVk|VulkanTrainer|VkVsCuda|Mcmc|Densification"`, 11.97s).
- First-loss intermediate gate validation: `VkVsCudaFirstLoss` **12/12 passed** with expected skips for P3-P7/L1; Gate_P2 validates eval_3D opacity-lane parity on overlap-active Gaussians by comparing CUDA flat opacity (`conic_opacity.reshape(-1)[gid]`) against VK captured opacity, while reporting single-sided active-set differences for Gate_P4 radii/AABB follow-up.
- Added MCMC/filter/full-dataset regression coverage and harnessing: CPU relocation, MCMC relocate/add/densify behavior, CUDA-golden MCMC replay fixtures, VulkanTrainer MCMC integration, Adam-state preservation/zeroing, opacity reset raw value + moment reset, eval_3D+MCMC filter propagation, legacy split child `filter_3D` inheritance, explicit Vulkan view schedules, strict all-GT validation, saved-PLY-vs-training-final render checks, and DSSIM slow-reference clamp-window equivalence.
- Full-dataset basketball validation: all 76 views, eval_3D/proper_ewa, `cap_max=30000`, 1000 steps, shared schedule, no opacity reset. CUDA/VK final counts matched **3513/3513**; `CUDA vs GT 17.22 dB`, `VK saved vs GT 16.92 dB`, `VK saved vs CUDA 27.99 dB`, `VK saved vs train final view 59.24 dB`, GT gap **0.30 dB**, gate status PASS.
- Full-dataset feature matrix at 1000 steps with `cap_max=100000` and no opacity reset PASSed all gates: baseline `VK saved vs CUDA 28.71 dB`, original DSSIM 0.2 `29.30 dB`, LR decay `30.71 dB`, SH warmup 1000 `30.91 dB` with render SH degree 0. All four runs matched final counts **3513/3513** and had GT gaps within **0.33 dB**. DSSIM VK training bottleneck was traced to host-side direct 11x11 sliding-window SSIM/gradient accumulation; the optimized separable/parallel path reduced the 1000-step DSSIM VK train time from **2632.7s** to **167.6s** while keeping the matrix gate PASS (`VK saved vs CUDA 28.25 dB`, counts **3513/3513**).
- Previous S13 parity status remains valid: scan1 2D replay-boundary fix improved VK↔CUDA PSNR to **107.136986 dB** at 10 steps, **49.822707 dB** at 200 steps, and **27.773822 dB** at 1000 steps; eval_3D parity-mode training remains enabled only for `parity_mode=true`.
- Hook/gate health: `.claude/gates/test-audit.md` and `.claude/gates/research.md` were not present in this worktree; review was still performed via independent subagent diff review plus local build/targeted/parity/full CTest.

## Parity Ladder (L1-L5)
| Level | Meaning | Status |
|-------|---------|--------|
| L1a | CPU↔VK same-ply forward | ✅ 99.15 dB (Phase A close) |
| L1b | VK↔CUDA same-ply forward (basketball cam0) | **54.84 dB** (5.16 dB to 60 dB target). Gate_I1-I4 + P1/P2 PASS; P3-P7 SKIP |
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
| VK-CUDA L1b parity (PSNR≥60dB) | IN PROGRESS | S8-S16 | 25.3→54.84 dB. Phase 1 (Gate_I1-I4) + Gate_P1/P2 done; need Gate_P3-P7 implementation |
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
| Gate_P3 RgbColors | SKIP | needs SH-evaluated RGB compare |
| Gate_P4 Radii | SKIP | blocked on new_aabb Phase-2 |
| Gate_P5 SortedIds | SKIP | sort gate, gateway to cascade-trace harness |
| Gate_P6 TFinalNContrib | SKIP | rasterize state |
| Gate_P7 RenderedImage | SKIP | full-frame PSNR |
| Gate_L1 L1Loss | SKIP | needs forward render parity first |

## Python → C++ Gaps Closed (SP-6)
1. **Opacity reg**: `dL/d_raw_opacity += (0.01/N)*sig*(1-sig)` — after backward, before Adam upload
2. **Scale reg**: `dL/d_raw_scale += (0.01/N)*exp(raw_sc)` — per-component
3. **Position noise**: `Sigma @ N(0,1) * op_sigmoid(1-opacity) * noise_lr * pos_lr` — after Adam download
4. **Spatial LR**: `pos_lr = spatial_lr_scale * lr_schedule(...)` — default scale=1.0

## Latest Sessions

### S16 — 2026-05-01 — CUDA intermediate Gate_P2 opened
- Opened `VkVsCudaFirstLoss.Gate_P2_ConicOpacity` as a real parity gate. The CUDA basketball step-1 dump is `eval_3D=true`, so `conic_opacity.npy` is a raw `[P,4]` memory view but only the flat opacity lane `conic_opacity.reshape(-1)[gid]` is semantically valid; the 2D `{a,b,c,opacity}` row interpretation would be wrong for this fixture.
- Gate_P2 compares that CUDA flat opacity against VK `captured_conic_opacity()[gid*4+3]` only for overlap-active Gaussians, and asserts a non-vacuous overlap population before checking values. Active-set disagreements are counted and reported, but left to Gate_P4 radii/AABB rather than mixed into the opacity gate.
- Validation: `VkVsCudaFirstLoss.Gate_P2_ConicOpacity` PASS; all `VkVsCudaFirstLoss` gates **12/12 passed** with expected skips for P3-P7/L1; full CTest **315/315 passed** in 40.87s. +1 and +2 independent reviews both approved with no findings.

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
