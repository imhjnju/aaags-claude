# VK ↔ CUDA eval_3D Parity — PSNR ≥60 dB Design

**Date:** 2026-04-21
**Scope:** Close the VK-vs-CUDA PSNR gap on full-scene eval_3D rendering from ~32 dB to ≥60 dB.
**Author:** Claude (Opus 4.7, 1M ctx) via brainstorming skill.

---

## 1. Problem Statement

The Vulkan port of AAA-Gaussians' `eval_3D` render path currently reaches ~32 dB PSNR against the CUDA reference on full-scene inputs. Tiny-fixture tests report ≥83 dB post-recent-fixes (`d1b8ead`), confirming the core port is correct, but full-scene divergence remains. Target: **≥60 dB on a frozen basketball scene**, with the measurement automated and regression-gated.

32 dB PSNR corresponds to MSE ≈ 6.3e-4 (RMS ≈ 0.025 on [0,1] pixel range). At this scale we expect **several small cumulative issues**, not a single catastrophic bug — consistent with the already-documented fixes (depth `precise` keyword, transpose fix, rectangular tile rect) having closed the large divergences.

## 2. Goals / Non-Goals

### Goals
- PSNR ≥60 dB full-scene on frozen basketball input (`basket-aaa.ply` @ cam 0, 720×960).
- Direct VK-vs-CUDA comparison. **CPU reference is NOT consulted** during this work.
- Measurement harness that any later change must regression-test against.
- Each PSNR-affecting fix shipped as a separate commit with attributable delta.

### Non-Goals
- Cross-GPU determinism (held constant to Tegra Thor dev box).
- Bit-exact match across CUDA versions / compute capabilities.
- Performance optimization (separate work).
- Training/backward correctness — forward-only in this design.
- Eval_3D = false path — only eval_3D = true is in scope.

## 3. Frozen Inputs

| Input | Value |
|---|---|
| PLY | `/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply` (400000 trained Gaussians, SH degree 3) |
| Cameras | `/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json` |
| Primary camera | `cam 0` = `78899858295079` |
| Resolution | 720 × 960 |
| eval_3D flag | `true` |
| filter_3D | See §3.1 below |
| Background | `[0, 0, 0]` |
| Compute device | NVIDIA Tegra Thor (Vulkan 1.4, CUDA 12.x) |

### 3.1 `filter_3D` Sourcing

`filter_3D` is a per-Gaussian scalar used by AAA-Gaussians' `eval_3D` path. Verified: `basket-aaa.ply` bakes `filter_3D` as a per-vertex `property float filter_3D`, so both the CUDA golden generator and the VK C++ loader read the **same** tensor from the same file. The existing C++ `loadPly()` already populates `GaussianData.filter_3D` from this property (`types.h:47`). The harness must NOT recompute filter_3D — doing so would introduce a divergence between the two pipelines' inputs, invalidating the comparison.

If a future scene's PLY lacks the `filter_3D` property, the harness must fail loudly; recomputation is out of scope for this plan.

## 4. Architecture — Phased Approach

### Phase 0 — Measurement Harness (~4 h)

Produces a single gtest-invocable regression gate.

**Components**

- `tools/render_cuda_basketball.py`
  - Loads `basket-aaa.ply` via `tools/ply_loader.py`; loads camera 0 from `cameras.json`.
  - Bridges PLY → AAA-Gaussians tensor layout (positions, scales, rotations, opacities, SH coefficients, filter_3D).
  - Invokes `GaussianRasterizer(eval_3D=True)` → renders CHW float32.
  - Writes `tests/golden/basketball/cam0/cuda_image.npy` (shape `[3, 960, 720]`, float32 little-endian).
  - Cache key: SHA256(ply_bytes + cameras.json entry for cam 0 + script SHA256). Regenerated only on change.

- `tests/test_vk_vs_cuda_basketball.cpp` — new gtest.
  - Skips if golden file absent or no Vulkan device.
  - Builds the VK preprocess+sort+rasterize chain with the same frozen inputs (data-loader reuses `ply_loader.cpp` + `cameras.json` parsing from existing basketball e2e test).
  - Downloads VK image in CHW float32.
  - Computes:
    - Global PSNR (headline)
    - Per-channel PSNR (R, G, B separately)
    - Max-abs, mean-abs, p99 absolute error
    - Count of pixels with abs-err > 1e-3
    - Spatial diff heatmap PNG (`vk_cuda_diff_cam0.png` in `CMAKE_BINARY_DIR`)
  - Asserts `PSNR ≥ kBaselinePSNR` (a compile-time constant; G0 = 32.0).

**Why not reuse `FullChain_TinyFixture_Eval3D`?** That test targets tiny `.npy` fixtures and is already at ≥83 dB. The full-scene case is a distinct regime and deserves its own gate.

### Phase 1 — Structural Audit + Top-Ranked Fixes

**Parallel audit** — each subagent owns one concern and produces a structured report of divergences:

| # | Scope | CUDA anchor | VK anchor |
|---|---|---|---|
| 1 | Preprocess numerics (3D→2D cov, dilation factor, opacity scaling, cutoff thresholds) | `forward.cu::preprocessCUDA`, `consistent_common.cuh::store_gauss2screen` | `preprocess.comp` lines 200–1100 |
| 2 | Rasterize k-buffer + alpha blend (plane math, early-exit ≤0.0001, accum order) | `forward.cu::renderCUDA`, `consistent_common.cuh::max_contrib_ray` | `rasterize.comp` lines 100–330 |
| 3 | Tile binning + AABB + `tiles_touched` | `forward.cu::duplicateWithKeys_extended`, `compute_aabb_view/screen` | `preprocess.comp` AABB path + `scatter.comp` |
| 4 | Sort semantics (64-bit key format, tie-breaking, stability, bit patterns for negative depths) | CUB `DeviceRadixSort::SortPairs` | `radix_sort_count.comp` + `radix_sort_scatter.comp` |
| 5 | SH evaluation + RGB clamp / training-mode flag | `forward.cu` SH inline | `preprocess.comp` SH block |

**Synthesis:** a single ranked divergence list, grouped by likely PSNR impact. Top 3–5 items become the Phase 1 fix targets.

**Fix loop per target:**
1. Research gate (read `.claude/gates/research.md`)
2. Spec-read for the affected subsystem
3. Write failing test (RED) — per-buffer numeric check or harness regression
4. Implement
5. GREEN — harness passes new threshold
6. Commit with PSNR delta in message

### Phase 2 — Bisect Fallback (only if Phase 1 stalls below 60 dB)

**Trigger:** All Phase 1 items shipped and PSNR < 60 dB.

**Instrumentation:**
- Python: patch AAA-Gaussians (non-invasive fork under `tools/cuda_dump/`) to export post-preprocess buffers (`means2D`, `depths`, `radii`, `conic_opacity`, `rgb`, `gauss2screen`), post-sort (`keys_sorted`, `values_sorted`), post-tile-range (`tile_ranges`).
- VK: extend render CLI with `--dump-stage <name>` that downloads the equivalent buffer.
- Diff script: per-buffer `max_abs`, `mean_abs`, count of `|diff| > 1e-4`.

**Bisect logic:** first stage with non-trivial divergence is the bug. Stage ordering: preprocess → sort keys → tile_ranges → per-pixel output.

**Explicit out of scope:** dumping per-pixel rasterize state (intermediate T, C[3] during the 256-Gaussian blend loop). Too expensive; rarely the bottleneck given 32 dB is a *compound* small-error signature.

## 5. Milestones

| Gate | PSNR | Description |
|---|---|---|
| G0 | 32 dB | Harness lands, baseline reproduced, CI-gated |
| G1 | ≥40 dB | Top Phase 1 divergence fixed |
| G2 | ≥50 dB | Remaining Phase 1 divergences |
| G3 | **≥60 dB** | Final fix (may need Phase 2 bisect) |

## 6. Acceptance Criteria

- Full-scene PSNR ≥60 dB on frozen basketball input
- Existing `FullChain_TinyFixture_Eval3D` still passes (no tiny regression)
- No previously-passing test regresses
- Per-channel (R, G, B) PSNR within 2 dB of each other (no channel-specific residual)
- Harness runs in CI without external dependencies beyond the basketball assets already on the dev box

## 7. Failure Modes + Emergency Exit Clause

The target in §6 is **unconditional ≥60 dB**. This section defines an emergency exit that requires **explicit user approval** to engage — it is not an automatic fallback.

If after Phase 2 bisect the residual divergence traces to inherent CUDA non-determinism (e.g., CUB tie-breaking in depth-equal cases that cannot be mirrored in a software radix sort without unreasonable re-implementation cost), the engineer pauses and asks the user whether to engage the exit. The exit permits:

- Accepting ≥58 dB **if and only if** the remaining delta is traced to a specific documented source.
- Filing an `investigations/vk_cuda_residual_<date>.md` note with the root cause and reproduction script.
- Citing a benchmark of "CUDA-vs-CUDA on same scene across GPUs is < 80 dB" as context for the irreducible gap.

A cargo-cult fix that raises PSNR without explaining *why* is **not** an acceptable path to passing §6.

## 8. Process Constraints

- **Subagent-driven execution** per user global preference (`~/.claude/CLAUDE.md`).
- **Evidence, not reasoning** per project `CLAUDE.md` — each fix cites the divergence report line and shows the measured PSNR delta.
- **One variable per experiment** — fixes ship separately; no bundled multi-fix commits.
- **Spec-first** — every fix consults `spec/README.md` for its subsystem before coding.

## 9. Open Questions (deferred to writing-plans stage)

- Exact order to tackle Phase 1 fix list — depends on synthesis report.
- Whether to add a second camera (e.g., cam 10) to the harness for cross-camera validation — defer to after G1.
- CUDA golden caching policy — inline regeneration or explicit `make golden` target.

---

## Appendix A — Recent-Commit Context

Three fix commits moved PSNR from 28.7 → 32 dB on full scene; tiny fixture is ≥83 dB:

- `86252d9` — consistent CHW→HWC conversion in `download_image` + test golden
- `2412223` — read opacity from `conic_opacity` in eval_3D rasterizer
- `d1b8ead` — transpose fix (`max_contrib` matrix indexing) + rectangular tile rect (float `radius_f[N*2]`)
- `33d414c` — merge into master

The `precise` keyword in `preprocess.comp` (line 118 comment) documents the depth-precision fix (28 → 60+ dB on tiny). This is **already in place**; full-scene regression must be elsewhere.
