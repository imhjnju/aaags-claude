# Rasterize Stage Audit — VK vs CUDA

**Date:** 2026-04-21
**CUDA anchor:** `forward.cu::renderCUDA`, `consistent_common.cuh::max_contrib_ray`, `hierarchical_render.cuh`
**VK anchor:** `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp`

## Summary Table

| Item | CUDA (aaa.json) | VK | Match? | PSNR Impact |
|---|---|---|---|---|
| Sort mode | HIERARCHICAL (sort_mode=3); per-pixel HEAD_WINDOW=4 k-buffer | Simple front-to-back over pre-sorted list | **NO — CRITICAL** | Catastrophic |
| hierarchical_4x4_culling | 3-level (tail/mid/head) priority queues; per-4x4 alpha culling | Not implemented | **NO — CRITICAL** | Very High |
| tile_based_culling | TILE_BASED_CULLING=true; culls per-tile based on max contribution | Not implemented | **NO** | High |
| load_balancing | LOAD_BALANCING=true; work redistribution | Not implemented | **NO** | Medium |
| eval_3D k-buffer sorting | Per-pixel k-buffer (HEAD_WINDOW=4) re-sorts by depth | No sorting; global pre-sort order used | **NO** | Very High |
| Alpha blend T threshold | T_MIN = 0.0001f | T_MIN = 0.0001 | Match | None |
| Alpha threshold | min(0.99f, ...) | min(0.99, ...) | Match | None |
| Gaussian weight formula | -0.5*(a*dx^2 + c*dy^2) - b*dx*dy | Same | Match | None |
| eval_3D weight (max_contrib_ray) | plane intersection formula | Faithful port (maxContribRayPixel) | Match | None |
| Pixel coordinates | (float)pix.x, no +0.5 | float(pixel_x), no +0.5 | Match | None |
| Background color | C + T*bg_color | C + T*bg_color | Match | None |
| Output layout | CHW | CHW | Match | None |

## Detailed Findings

### 1. HIERARCHICAL sort_mode — CRITICAL

CUDA uses `sortGaussiansRayHierarchicalCUDA_forward` (hierarchical_render.cuh, lines 1062–1159).

Mechanism:
- **Tail buffer** per 4×4 sub-tile (64 entries): sorted by depth at 4×4 center
- **Mid buffer** per 2×2 sub-tile (MID_WINDOW=8)
- **Head buffer** per pixel (HEAD_WINDOW=4): top-4 by per-pixel depth, blended in order

With CULL_ALPHA=true (hierarchical_4x4_culling): Gaussian dropped from entire 4×4 tile if
`alpha < ALPHA_THRESHOLD` for the whole tile.

VK (rasterize.comp lines 182–258): simple front-to-back traversal, no per-pixel depth re-sort.

Impact: pixels away from tile centers receive Gaussians in wrong depth order → major alpha-compositing artifacts across entire image.

### 2. Alpha blend — MATCH

CUDA (forward.cu lines 475–492):
```cpp
float test_T = T * (1 - alpha);
if (test_T < 0.0001f) { done = true; continue; }
C[ch] += collected_color[j*CHANNELS+ch] * alpha * T;
T = test_T;
```

VK (rasterize.comp lines 302–311):
```glsl
float test_T = T * (1.0 - alpha);
if (test_T < T_MIN) { pixel_done = true; break; }
C.r += s_rgb_r[k] * alpha * T; ...
T = test_T;
```
Arithmetic order matches.

### 3. Gaussian weight eval_3D — MATCH

VK maxContribRayPixel (lines 139–157) is a faithful port of CUDA's max_contrib_ray (consistent_common.cuh lines 89–106).

## Root Cause Summary

Primary driver of 25 dB PSNR gap (in priority order):
1. CRITICAL: sort_mode=HIERARCHICAL not implemented (no per-pixel k-buffer, no head queue)
2. CRITICAL: hierarchical_4x4_culling absent (different Gaussian sets per sub-tile)
3. HIGH: tile_based_culling absent (different tile-pair count)
4. HIGH: sort_order=PER_TILE_DEPTH_MAXPOS absent (depth key mismatch)
5. MEDIUM: load_balancing absent
