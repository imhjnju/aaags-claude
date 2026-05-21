# forward / noise_loss Worktree Migration Deep Research

Date: 2026-05-19

Target worktree:

```text
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/best-3dgs-optimized-20260512/harmonyos_3dgs
```

Source worktrees:

```text
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/forward/harmonyos_3dgs
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/noise_loss/harmonyos_3dgs
```

## Executive summary

Do **not** wholesale sync either source worktree into the target.

The target is already ahead of both source worktrees in several correctness-critical areas, especially canonical CUDA-compatible SortPairs, GPU DSSIM target precompute/cache, GPU position noise, GPU raw activation handling, GPU regularization, raw replay, and fused backward support.

The best migration strategy is selective:

1. **Keep target canonical SortPairs as the only correctness path.**
2. **Treat `noise_loss` as mostly already integrated.** Only a small missing DSSIM cache test is worth porting.
3. **Port selected `forward` steady-state ideas manually**, especially persistent buffer reuse and recorded steady-state dispatch, but rewrite them around the target canonical ABI rather than forward's packed-keyval ABI.
4. **Skip all packed-keyval correctness-path changes** from `forward` and stale sort/binning changes from `noise_loss`.

Expected performance opportunity:

- `noise_loss`: almost no remaining large runtime win because the target already contains the major GPU noise and DSSIM precompute/cache changes.
- `forward`: still potentially meaningful, because current Vulkan `forward_total_ms` is about **34.406 ms** per step, **27.4%** of the current mean step time. The remaining forward opportunities are allocation/re-recording/host-mirror reduction and possibly render-only no-host modes.
- Main remaining full-training bottleneck is still backward: current `backward_gpu_ms` mean is **89.497 ms**, vs CUDA `backward_ms` mean **54.315 ms**.

## Current performance baseline

Latest completed full-feature Vulkan benchmark:

```text
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/best-3dgs-optimized-20260512/harmonyos_3dgs/reports/aaa_full_features_10000_cap200000_20260519_dssim_precompute
```

Key metrics:

| Metric | Value |
|---|---:|
| Steps | 10000 |
| Wall time | 1306.95 s |
| Mean `total_ms` | 125.621 ms |
| Median `total_ms` | 104.388 ms |
| Throughput from wall | 7.65 step/s |
| Final N | 200000 |
| Final loss | 0.007142 |
| Speed vs CUDA wall | 58.9% |
| Speed vs CUDA mean step | 53.8% |

Current Vulkan stage means:

| Stage | Mean ms | Share of mean step | Migration relevance |
|---|---:|---:|---|
| `backward_gpu_ms` | 89.497 | 71.2% | largest remaining gap; not primarily a forward/noise_loss issue unless replay/input layout changes |
| `forward_total_ms` | 34.406 | 27.4% | highest-value target for forward worktree ideas |
| `raster_ms` | 17.787 | 14.2% | possible forward/raster path gains |
| `preprocess_process_ms` | 8.133 | 6.5% | possible host-mirror/preprocess-output reduction |
| `bin_ms` | 3.044 | 2.4% | possible persistent buffer / recorded path gains |
| `sort_ms` | 2.194 | 1.7% | canonical SortPairs already near target; do not regress correctness |
| `loss_ms` | 2.044 | 1.6% | major DSSIM win already integrated |
| `noise_ms` | 0.181 | 0.1% | no longer a performance bottleneck |
| `adam_gpu_ms` | 1.019 | 0.8% | already GPU-resident |

Previous Vulkan vs current Vulkan:

| Metric | Previous | Current | Change |
|---|---:|---:|---:|
| Wall time | 1476.84 s | 1306.95 s | 1.13x faster |
| Mean `total_ms` | 143.170 ms | 125.621 ms | 1.14x faster |
| Mean `forward_total_ms` | 49.732 ms | 34.406 ms | 1.45x faster |
| Mean `loss_ms` | 10.935 ms | 2.044 ms | 5.35x faster |
| Mean `backward_gpu_ms` | 91.538 ms | 89.497 ms | 1.02x faster |

The big `noise_loss`/DSSIM performance win is already present in the target.

## Hard invariants to preserve

Any migration must preserve these target invariants:

1. Canonical SortPairs contract:

   ```text
   key   = (tile_id << 32) | depth_bits
   value = full uint32 gaussian_index
   ```

2. Packed Fuchsia records may be an internal implementation detail only after canonical packing; raster/backward/consumers must use canonical `values_sorted` and `tile_ranges`.
3. No correctness path may truncate Gaussian IDs to 19 or 20 bits.
4. `tile_ranges` must be zero-initialized so empty tiles remain `[0,0)`.
5. GPU-resident fast paths may intentionally null host mirrors; users must check GPU handles before CPU pointers.
6. `pre.num_tile_pairs` is authoritative when GPU `tiles_touched` is authoritative.
7. CHW layout for image and `dL_dpixels` must remain consistent across raster, L1, DSSIM, and backward.
8. `ForwardCache` GPU handles are producer-owned and lifetime-sensitive; retained buffers must outlive loss/backward use.
9. `raw_cpu_dirty_` controls GPU raw activation/noise vs CPU materialization. Naive merges can reintroduce full raw downloads or stale CPU params.
10. Fused eval3D replay backward must not bypass the `proper_ewa` gate.

## `noise_loss` worktree audit

### Already integrated in target

The following `noise_loss` performance-critical changes are already present in the target:

#### DSSIM split / precompute / cache

Files/symbols:

- `include/vulkan/dssim_loss_pass.h`
- `src/vulkan/dssim_loss_pass.cpp`
- `src/vulkan/shaders/dssim_blur5_h.comp`
- `src/vulkan/shaders/dssim_blur5_v_terms.comp`
- `src/vulkan/shaders/dssim_target_blur5_h.comp`
- `src/vulkan/shaders/dssim_target_blur5_v.comp`
- `src/vulkan/shaders/dssim_transpose3_h_grad.comp`
- `src/vulkan/shaders/dssim_transpose3_v.comp`
- `DssimLossPass::precompute_target_terms`
- `DssimLossPass::dispatch_sync`
- `VulkanTrainer::run_gpu_dssim_loss`
- `gpu_dssim_cached_target_*`
- `gpu_dssim_pass_bound_`

Performance impact already measured:

- `loss_ms`: `10.935 ms -> 2.044 ms`
- Stage speedup: `5.35x`

#### GPU position noise

Files/symbols:

- `include/vulkan/position_noise_pass.h`
- `src/vulkan/position_noise_pass.cpp`
- `src/vulkan/shaders/position_noise.comp`
- `VulkanTrainer::inject_position_noise`
- `position_noise_pass_`

The target also improves over the `noise_loss` version by adding safer raw materialization helpers:

- `VulkanTrainer::download_active_noise_inputs`
- `VulkanTrainer::materialize_raw_positions`
- `RawMaterializationKind`
- `raw_cpu_dirty_`

Do not overwrite these with the simpler `noise_loss` version.

#### Trainer GPU-resident loss / forward-output reuse

Files/symbols:

- `VulkanTrainer::run_gpu_l1_loss`
- `VulkanTrainer::run_gpu_dssim_loss`
- `last_forward_gpu_resident_outputs_`
- `out_cache.gpu_resident_outputs`
- `out_cache.retain_gpu_outputs`

Target already avoids forward image/cache downloads when GPU loss and forward output reuse are enabled.

#### GPU raw activation / Adam / regularization integration

The target is ahead of `noise_loss` because it keeps GPU raw activation enabled with position noise and supports GPU regularization rather than disabling fast paths when opacity/scale regularization is nonzero.

Do not port old fast-path exclusions like:

```text
opacity_reg == 0 && scale_reg == 0
```

### Safe `noise_loss` port candidates

#### P2: add missing mutated-target DSSIM cache test

Candidate:

```text
noise_loss/harmonyos_3dgs/tests/test_training_step_vk.cpp
```

Test to port/adapt:

```text
GpuDssimFastPathRefreshesMutatedTargetStorage
```

Expected runtime impact: none.

Value: protects target-buffer mutation/cache correctness for the already-ported DSSIM fast path.

### `noise_loss` changes to skip

Skip these from `noise_loss`:

1. Any stale sorter/radix/Fuchsia/packed-keyval files.
2. Any change that removes or weakens canonical SortPairs.
3. Any fallback where `GS3D_TRAIN_GPU_L1_LOSS=1` implicitly enables DSSIM when `lambda_dssim != 0`.
4. Any old `opacity_reg == 0 && scale_reg == 0` fast-path gating.
5. Wholesale `src/vulkan_trainer.cpp` or `include/vulkan_trainer.h` replacement.

## `forward` worktree audit

The `forward` worktree has one performance-critical idea that remains valuable: reducing steady-state render overhead through recorded/persistent GPU paths. But its implementation is tied to a packed-keyval ABI that conflicts with current target correctness requirements.

### Already integrated in target

#### Raw replay raster shader

Already present and byte-identical:

```text
src/vulkan/shaders/rasterize_raw_replay.comp
```

CMake shader compile/embed is already present.

`RasterizePass` shader selection is already present:

```text
src/vulkan/rasterize_pass.cpp
```

#### Forward cache / GPU-resident outputs

Target already supports:

- `ForwardCache::gpu_resident_outputs`
- `ForwardCache::retain_gpu_outputs`
- retained `rendered_image_gpu`
- retained `T_final_gpu`
- retained `n_contrib_gpu`
- GPU `values_sorted`
- GPU `tile_ranges`
- eval3D replay-order generation

Primary file:

```text
src/vulkan/rasterizer_vulkan.cpp
```

#### CMake raw replay wiring

Target already compiles/embeds `rasterize_raw_replay` and also includes newer canonical SortPairs, DSSIM, position-noise, and backward shader wiring not present in `forward`.

Do not replace target CMake wholesale.

### High-value `forward` ideas to port manually

#### P0: adapt recorded steady-state SortPairs path to canonical ABI

Forward implements a recorded steady-state path around packed Fuchsia keyvals. The idea is valuable, but the ABI is not acceptable.

Skip forward's direct packed path:

- `prepare_record_fuchsia_keyvals`
- `record_fuchsia_keyvals`
- `sort_via_fuchsia_gpu`
- `keyval_pack` 13/32/19 layout
- any `keyvals_sorted & mask` Gaussian ID extraction

Manual target-compatible design:

1. Use target `TileBinnerVulkan` canonical outputs:

   ```text
   keys_unsorted_gpu
   values_unsorted_gpu
   total_pairs / pre.num_tile_pairs
   ```

2. Record canonical SortPairs pack:

   ```text
   CanonicalSortPairPackPass
   ```

3. Run Fuchsia sort over internal u96/key48 records.

4. Record canonical extract:

   ```text
   CanonicalSortPairExtractPass
   values_sorted_gpu
   tile_ranges_gpu
   ```

5. Ensure downstream raster/backward read canonical `values_sorted[]` and `tile_ranges[]`, never packed-keyval low bits.

Files to modify later:

- `src/vulkan/sorter_vulkan.cpp`
- `include/vulkan/sorter_vulkan.h`
- `src/vulkan/packed_keyval_extract_pass.cpp` if existing canonical extract/tilerange helpers are reused
- `src/vulkan/tile_binner_vulkan.cpp` if extra record-mode hooks are required
- `src/vulkan/vk_render_main.cpp` if the steady-state benchmark path is updated

Expected impact:

- Reduces per-frame/step command construction and sort/binner preparation overhead.
- May improve `forward_total_ms`, `bin_ms`, `sort_ms`, and wall-time variance.
- Must be measured; current `sort_ms` is only `2.194 ms`, so the larger win is likely dispatch/record/host-bound overhead rather than sort kernel time alone.

Correctness risk:

- High if any packed-keyval consumer slips back in.
- Medium if ping-pong record parity or `key_bits` handling is wrong.
- Must include full `uint32_t` Gaussian ID regression tests.

#### P1: port `RasterizerVulkan::rasterize` persistent buffer reuse

Forward has layer-1 buffer reuse in `RasterizerVulkan::rasterize`:

- `r_img_`
- `r_tfinal_`
- `r_ncontrib_`
- `r_ubo_`
- `r_dummy4_`
- `r_eval3d_ubo_`

Target should adapt this if not already present.

File:

```text
src/vulkan/rasterizer_vulkan.cpp
```

Expected impact:

- Reduces per-step allocation churn.
- Should modestly improve `forward_total_ms`, `raster_ms`, and p95/p99 variance.

Correctness risk:

- Low if buffer lifetimes are tied to retained `ForwardCache` ownership correctly.
- Medium if retained GPU outputs are overwritten before backward/loss consumes them.

#### P1: port `TileBinnerVulkan` persistent scratch/UBO reuse

Forward has persistent layer-1 buffer reuse for binner scratch/UBO-like buffers:

- `bin_scatter_ubo_buf_`
- `bin_dummy4_buf_`

File:

```text
src/vulkan/tile_binner_vulkan.cpp
```

Expected impact:

- Small-to-moderate allocation overhead reduction.
- More likely to improve tail latency than mean kernel time.

Correctness risk:

- Low if canonical `keys_unsorted` / `values_unsorted` remain authoritative.

#### P2: optional render-only preprocessor no-host-output mode

Forward adds the concept of:

```text
RenderConfig::gpu_resident_forward
```

and preprocessor CPU-materialization skips.

Files:

- `include/types.h`
- `src/vulkan/preprocessor_vulkan.cpp`
- call sites that build `RenderConfig`

Target currently has more nuanced training cache/lifetime needs. This should be ported only narrowly:

- render-only benchmark/export paths first;
- training path only after proving backward/loss do not require CPU mirrors;
- never size `R` from stale CPU-side `tiles_touched` if GPU `tiles_touched` is authoritative.

Expected impact:

- Potentially reduces preprocess downloads/materialization.
- Current `preprocess_process_ms` mean is `8.133 ms`, so there is some opportunity.

Correctness risk:

- Medium-to-high if CPU callers expect preprocessor outputs.
- Must be gated explicitly.

### `forward` changes to skip

Skip the following from `forward`:

1. Packed-keyval external correctness ABI.
2. `keyval_pack` 13 tile / 32 depth / 19 ID layout.
3. `prepare_record_fuchsia_keyvals` and `record_fuchsia_keyvals` as correctness paths.
4. Shader paths that extract Gaussian IDs via low-bit masks such as `& 0x7FFFFu` or other packed-keyval assumptions.
5. Packed-keyval tests that assert low-ID constraints rather than canonical full-`uint32_t` value preservation.
6. Wholesale `CMakeLists.txt` replacement.
7. Any change that reintroduces `MAX_GAUSS` correctness limits.

## Proposed migration plan

### Phase 0: freeze baseline and protect invariants

No code migration before confirming these are stable:

- current benchmark report exists;
- canonical SortPairs tests pass;
- GPU DSSIM/noise tests pass;
- full CTest passes.

Already known from current session:

- full CTest: `367/367` passed;
- current 10000-step benchmark completed;
- final N: `200000`;
- final loss: `0.007142`.

### Phase 1: low-risk test-only noise_loss sync

Port/adapt:

```text
GpuDssimFastPathRefreshesMutatedTargetStorage
```

from:

```text
noise_loss/harmonyos_3dgs/tests/test_training_step_vk.cpp
```

Validation:

```bash
cmake --build build_release -j$(nproc)
build_release/gs3d_vk_tests --gtest_filter='VulkanTrainer*GpuDssim*:DssimLossPassVulkan.*'
ctest --test-dir build_release --output-on-failure
```

Expected performance: none.

Purpose: protect already-ported DSSIM cache behavior.

### Phase 2: port forward persistent buffer reuse

Port/adapt:

- `RasterizerVulkan::rasterize` persistent output/UBO/dummy buffer reuse.
- `TileBinnerVulkan` persistent scatter UBO/dummy buffer reuse.

Do not touch SortPairs ABI in this phase.

Validation:

```bash
cmake --build build_release -j$(nproc)
build_release/gs3d_vk_tests --gtest_filter='Rasterize*:*ForwardCache*:TileBinnerVulkan*:VulkanTrainer*Gpu*'
ctest --test-dir build_release --output-on-failure
```

Performance benchmark:

- short basketball stage-timing run, e.g. 1000 steps, cap 30000, fixed view schedule;
- compare `forward_total_ms`, `raster_ms`, `bin_ms`, `total_ms`, p95/p99.

Decision rule:

- keep if `forward_total_ms` or p95/p99 improves measurably without correctness regressions;
- if mean movement is <1% and tail does not improve, deprioritize.

### Phase 3: canonical recorded steady-state SortPairs

Implement forward's recorded steady-state concept, but using target canonical SortPairs:

```text
TileBinner canonical keys/values
  -> CanonicalSortPairPackPass
  -> Fuchsia sort internal records
  -> CanonicalSortPairExtractPass
  -> values_sorted_gpu + tile_ranges_gpu
```

Files likely involved:

- `include/vulkan/sorter_vulkan.h`
- `src/vulkan/sorter_vulkan.cpp`
- `include/vulkan/tile_binner_vulkan.h`
- `src/vulkan/tile_binner_vulkan.cpp`
- `src/vulkan/vk_render_main.cpp`
- canonical sortpair shader pass wrappers, if record-only APIs are missing

Tests to add/adapt first:

1. full `uint32_t` Gaussian IDs above `1 << 20` survive recorded SortPairs;
2. depth low bits are preserved, no `depth >> 4` quantization;
3. stable ordering for equal keys if expected by canonical path;
4. tile ranges match CUDA-style sorted keys;
5. GPU-resident no-host-mirror path still gives correct raster inputs.

Validation:

```bash
cmake --build build_release -j$(nproc)
build_release/gs3d_vk_tests --gtest_filter='SorterVulkan.*:RadixSortPassVk.*:FuchsiaRadixE2E*.*:TileBinnerVulkan.*:Rasterize*:*ForwardCache*'
ctest --test-dir build_release --output-on-failure
```

Performance benchmark:

- sorter-only synthetic sizes: `R=257`, `4096`, `65536`, and about `1M`;
- short training smoke with stage timing;
- promote to 10000-step cap 200000 only if short run shows meaningful movement.

Decision rule:

- must preserve canonical correctness;
- keep only if it improves `forward_total_ms`, `bin_ms`, `sort_ms`, or wall/tail latency enough to matter.

### Phase 4: optional no-host preprocessor mode

Only after phases 1-3 are stable.

Port concept, not code wholesale:

```text
RenderConfig::gpu_resident_forward
```

Gating:

- render-only first;
- training only when backward/loss inputs are all GPU-resident;
- diagnostic/capture modes should keep CPU mirrors.

Validation:

```bash
build_release/gs3d_vk_tests --gtest_filter='PreprocessorVulkan*:RasterizerVulkan*:VulkanTrainer*:*ForwardCache*'
ctest --test-dir build_release --output-on-failure
```

Performance benchmark:

- compare `preprocess_process_ms`, `forward_total_ms`, and wall time;
- check final render/output parity.

## Recommended benchmark matrix

Use a staged benchmark matrix instead of immediately running 10000 steps for every slice.

### Short slice benchmark

For each migration slice:

- base current target;
- target + slice;
- fixed basketball view schedule;
- stage timing on;
- `GS3D_USE_FUCHSIA_SORT=1`;
- `GS3D_TRAIN_GPU_DSSIM_LOSS=1`;
- `GS3D_REUSE_FORWARD_OUTPUTS=1`;
- `GS3D_TRAIN_GPU_GRAD_ADAM=1`;
- `GS3D_TRAIN_GPU_RAW_ACTIVATE=1`;
- 1000 steps, cap 30000 or equivalent fast cap.

Compare:

- `total_ms` mean/median/p95/p99;
- `forward_total_ms`;
- `preprocess_process_ms`;
- `bin_ms`;
- `sort_ms`;
- `raster_ms`;
- `loss_ms`;
- `backward_gpu_ms`;
- final loss trend;
- final N.

### Promotion benchmark

Only after tests pass and short run improves relevant stages:

- 10000 steps;
- cap 200000;
- same GT directory and view schedule policy as latest report;
- compare against latest current run:

```text
reports/aaa_full_features_10000_cap200000_20260519_dssim_precompute/summary.json
```

Acceptance criteria:

- no correctness/test regression;
- final N reaches `200000`;
- final loss does not regress unexpectedly;
- stage improvement matches hypothesis;
- wall time improves, or tail latency improves enough to justify keeping the change.

## Risk assessment

| Risk | Severity | Mitigation |
|---|---:|---|
| Accidentally reintroducing packed-keyval correctness ABI | High | skip wholesale merges; add full-`uint32_t` ID tests |
| Breaking retained GPU buffer lifetimes | High | forward/cache/backward tests; short training smoke |
| Reintroducing CPU downloads through raw/noise materialization | Medium | track `raw_download_ms`; preserve `raw_cpu_dirty_` semantics |
| No measurable speedup despite code complexity | Medium | slice benchmarks and stop rules |
| Stage timing scope confusion | Medium | compare same-run Vulkan stages first; CUDA only as directional reference |
| DSSIM cache stale-target bugs | Medium | port missing mutated-target test |

## Final recommendation

Proceed with selective migration in this order:

1. **Test-only `noise_loss` sync**: port the missing DSSIM mutated-target cache test.
2. **Forward buffer reuse**: port rasterizer and binner persistent buffer reuse, preserving target GPU-resident cache semantics.
3. **Canonical recorded SortPairs**: adapt forward's recorded steady-state concept to target canonical SortPairs; do not use forward packed-keyval ABI.
4. **Optional no-host preprocessor mode**: only after the previous phases are stable and benchmarks show forward/preprocess remains worth attacking.

Do not sync either worktree wholesale. The target is already the authoritative branch for correctness and full-feature training; source worktrees should be mined for isolated performance ideas only.
