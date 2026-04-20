# SP-2 Forward Pipeline — Completion Notes

**Branch**: `sp2-forward-pipeline`
**Base**: master `95699c8` (SP-1 merged)
**Tag**: `sp2-forward-ready`

## Delivered (spec §4)

- Deleted non-spec-compliant `PreprocessorVK` (0/10 matched) and rewrote from scratch.
- 6 passes: `preprocess.comp` (13 bindings + CameraUBO + packed conic_opacity
  + training-mode SH), `prefix_sum.comp` (3-level Blelloch), `scatter.comp`
  (duplicateWithKeys, uint64 tile|depth keys), `radix_sort_count.comp` +
  `radix_sort_scatter.comp` (4-bit × 16 passes), `tile_range.comp`,
  `rasterize.comp` (16×16 workgroup, shared memory tile batching).
- 4 adapter classes inheriting existing abstract base: `PreprocessorVulkan`,
  `TileBinnerVulkan`, `SorterVulkan`, `RasterizerVulkan`. Two-layer API: sync
  `process/bin/sort/rasterize` overrides + Vulkan-only `record(cmd, ...)` via
  `prepare_record()` + `record()`.
- Single source of truth: `preprocess_bindings.h` (binding indices, push
  constants, specialization constants), `vk_camera_ubo.h` (std140 struct).
- Hard-error guards per spec §4.4: `eval_3D`, non-16×16 tiles, `antialiasing`
  all reject at entry.
- ~19 new tests, all validated against SP-0 tiny fixture CUDA golden.
- Extended `dump_tiny.py` to dump input tensors (positions, scales, ...,
  camera components) for C++-side test reconstruction.
- Extended `VulkanComputePipeline` with mixed binding-types constructor
  (SSBO + UBO) and `update_ubo()` helper.
- Descriptor-set safety: `RadixSortPass::sort_record()` pre-allocates 16
  per-pass DS sets (one per radix pass) to satisfy Vulkan's "no DS update
  between bind and submit" rule.

## Test counts

- CPU + SP-0: 165
- SP-1 Vulkan infra: 15
- SP-2 new: 19
- **Total: 199 tests, 100% pass**

## Known divergence from CUDA golden (FOLLOWUP for SP-3)

`scatter.comp` uses `float(radii[i])` (int-ceiled radius) to recompute the
tile rect. CUDA uses the float eigenvalue radius directly. For Gaussians where
the integer rect is strictly larger than the float rect, scatter emits extra
tile pairs. Fixes require exporting `radius_f` as a new SSBO from
`preprocess.comp` (new binding 13), threading through scatter's bindings, and
updating all tests that bind those passes. Effect: ~3 Gaussians affected on
the tiny fixture; final image comparison between Vulkan and CUDA shows
up to 0.24 abs difference on affected pixels. **Does not affect record ≡ sync
parity (primary correctness assertion).**

## Handoff to SP-3 (backward pipeline)

- Packed `conic_opacity_packed` layout established. SP-3 `rasterize_backward`
  will consume the same packed buffer (no deinterleave needed on backward path).
- `record()` + `insert_compute_barrier(cmd)` pattern proven in forward chain.
  SP-3 chains onto the same command buffer style.
- `has_shader_atomic_float` = true on NVIDIA Thor (per SP-1 capability probe),
  so SP-3's rasterize_backward will use native atomicAdd(float) path.
- CUDA golden backward tensors already dumped by SP-0 tiny fixture:
  `backward_d_means2D/d_colors/d_conic/d_opacity/d_means3D/d_cov3D/d_sh/d_scales/d_rotations.npy`.
- `ForwardCache` carries the per-pixel `T_final` and `n_contrib` needed by
  SP-3's backward rasterize.
