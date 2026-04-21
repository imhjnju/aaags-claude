# Tile Binning + Scatter Audit — VK vs CUDA

**Date:** 2026-04-21
**CUDA anchor:** `forward.cu::duplicateWithKeys_extended`, `auxiliary.h::getRect/constructSortKey`
**VK anchor:** `preprocess.comp` AABB path, `scatter.comp`, `tile_range.comp`

## Summary Table

| # | Item | CUDA (aaa.json) | VK | Match? | PSNR Impact |
|---|---|---|---|---|---|
| 1 | rect_bounding 2D path | extent_x=extent*sqrt(cov2D.x); extent_y=extent*sqrt(cov2D.z), capped to radius | Always radius_f symmetric | **MISMATCH** | High |
| 2 | tight_opacity_bounding 2D path | extent=min(3.33, sqrt(2*opacity_power_threshold)) | 3.33 hardcoded | **MISMATCH** | Medium |
| 3 | new_aabb eval_3D | true → compute_aabb_view | true → computeAABBView | Match | — |
| 4 | Tile key encoding | (tile_id<<32) | floatBitsToUint(depth) | (uint64_t(tile_id)<<32) | uint64_t(depth_bits) | Match | — |
| 5 | Depth bias eval_3D | depth + 8.0f | max_pos_depth + 8.0 | Match | — |
| 6 | Depth eval non-eval_3D | depths[idx] = viewspace_z + 8.0f | floatBitsToUint(depths[i]) — NO +8 bias | **MISMATCH for eval_3D=false** | N/A (eval_3D=true in aaa.json) |
| 7 | load_balancing + tile_based_culling | computeTilebasedCullingTileCount; per-tile opacity test | Not implemented; all rect tiles always written | **MISMATCH** | High |
| 8 | Prefix sum | InclusiveSum → exclusive via offsets[idx-1] | Blelloch exclusive scan | Functionally equivalent | — |
| 9 | Screen AABB clipping | getRect clamps to [0, grid] | rectMin/Max clamp to [0, grid] | Match | — |
| 10 | Tile range | identifyTileRanges: boundary detection | tile_range.comp: same detection | Match | — |

## Detailed Findings

### 1. rect_bounding 2D path — HIGH IMPACT

CUDA (forward.cu:256–258):
```cpp
const float extent_x = min(rect_bounding ? (extent * sqrt(cov2D.x)) : radius, radius);
const float extent_y = min(rect_bounding ? (extent * sqrt(cov2D.z)) : radius, radius);
```

VK (preprocess.comp:1177–1178):
```glsl
ivec2 rmn = rectMin(pix, radius_f, radius_f, ...);  // symmetric
ivec2 rmx = rectMax(pix, radius_f, radius_f, ...);
```

For elongated Gaussians, VK has more tile pairs than CUDA → different sort buffer contents.

### 7. tile_based_culling — HIGH IMPACT

CUDA (forward.cu:278): `computeTilebasedCullingTileCount()` — per-tile opacity test.
CUDA scatter: writes INVALID_TILE_ID keys for culled slots.

VK: `tiles_touched[i] = n_tiles` = full rect area always. No culling in scatter.

## Root Cause

VK implementation matches `duplicateWithKeysCUDA` (basic path), not `duplicateWithKeys_extended`.
aaa.json activates rect_bounding=true, tight_opacity_bounding=true, load_balancing=true, tile_based_culling=true — none implemented in VK.
