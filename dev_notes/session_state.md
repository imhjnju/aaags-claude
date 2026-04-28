# Session State

## Current Phase (S13 — 2026-04-28)
Independent-train VK↔CUDA parity investigation on `scan1` (camera 0, 1600×1200, 28,747 Gaussians). The stable porting issue found this session was `n_contrib` semantics in the 2D rasterizer/backward path: CUDA stores the 1-based position of the last candidate that actually blended, not the count of blended Gaussians. Vulkan forward/backward and the CPU 2D reference are now aligned to that replay boundary. This reduced scan1 VK↔CUDA final-render drift from **90.91→107.14 dB at 10 steps**, **22.42→49.82 dB at 200 steps**, and **8.24→27.77 dB at 1000 steps**. Stepdump after the fix shows step1 forward/position/opacity/scale/rotation gradients match exactly; only SH gradients retain tiny atomic-order differences (`l2_rel≈6.2e-4`, max `≈1.9e-7`). Step2 render remains very close (`l2_rel≈4.3e-5`, max `≈3.9e-6`), but Adam with `eps=1e-15` amplifies near-zero new gradients into finite parameter deltas, explaining the remaining long-run drift as numerical sensitivity unless a later controlled same-state experiment proves another deterministic porting bug.

## Test Counts (2026-04-28 current)
- Build: `cmake -S harmonyos_3dgs -B harmonyos_3dgs/build -DBUILD_TESTS=ON && cmake --build harmonyos_3dgs/build` OK after `dssim.cpp`, rasterizer shader, and CPU reference updates.
- Full CTest baseline after n_contrib fix: **273/273 tests passed** (`ctest --test-dir harmonyos_3dgs/build --output-on-failure`, 36.35s; expected inventory remains 24 .cpp files).
- Focused affected test: `RasterizerBackwardVulkan.MatchesCPU_TinyFixture` PASS after CPU 2D `n_contrib` replay-boundary alignment.
- scan1 fixed comparison reports: 10-step VK↔CUDA PSNR **107.136986 dB**, 200-step **49.822707 dB**, 1000-step **27.773822 dB**.
- Hook/gate health: `.claude/gates/research.md` was missing in this worktree; research was still performed via spec read + code search before modifying rasterizer/backward code.

## Parity Ladder (L1-L5)
| Level | Meaning | Status |
|-------|---------|--------|
| L1a | CPU↔VK same-ply forward | ✅ 99.15 dB (Phase A close) |
| L1b | VK↔CUDA same-ply forward (basketball cam0) | **54.84 dB** (5.16 dB to 60 dB target). Gate_I1-I4 + P1 PASS; P2-P7 SKIP |
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
| VK-CUDA L1b parity (PSNR≥60dB) | IN PROGRESS | S8-S11 | 25.3→54.84 dB. Phase 1 (Gate_I1-I4) + Gate_P1 done; need Gate_P2-P7 implementation |
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
| Gate_P2 ConicOpacity | SKIP | needs implementation |
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

### S13 — 2026-04-28 (current) — scan1 n_contrib replay-boundary fix
- Targeted scan1 camera 0 at 1600×1200 with 28,747 Gaussians and measured VK↔CUDA independent-training drift at 10/200/1000 steps.
- Root cause: 2D `n_contrib` had count-based semantics in VK/CPU, but CUDA stores the 1-based position of the last candidate that actually blended. Fixed Vulkan forward/backward and CPU 2D reference to use the same replay-boundary contract.
- Verification: `RasterizerBackwardVulkan.MatchesCPU_TinyFixture` PASS and full CTest **273/273 PASS**. scan1 VK↔CUDA final-render PSNR improved to **107.14 dB / 49.82 dB / 27.77 dB** at 10/200/1000 steps. Full record: `dev_notes/scan1_n_contrib_parity_s13.md`.
- Eval_3D smoke: `gs3d_vk_train --eval_3d 1` now exists. scan1 eval_3D 10-step runs complete, but render is all black even after adding `filter_3D`; preprocess has nonzero tiles, while rasterizer writes `n_contrib=0` everywhere. Next blocker is eval_3D rasterize contribution logic.

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
