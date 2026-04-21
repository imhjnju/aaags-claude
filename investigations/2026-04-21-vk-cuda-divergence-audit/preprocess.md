# Preprocess Stage Audit — VK vs CUDA

**Date:** 2026-04-21
**CUDA anchor:** `forward.cu::preprocessCUDA`, `forward_common.h::computeCov2D / dilateCov2D`, `consistent_common.cuh::compute_gauss2screen / compute_aabb_view`
**VK anchor:** `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp` (1225 lines)

## Summary Table

| # | Item | CUDA (aaa.json = enabled) | VK | Match? | PSNR Impact |
|---|---|---|---|---|---|
| 1 | `proper_ewa_scaling` | opacity *= dilation_factor; convolution_scaling_factor = sqrt(det_orig/det_dilated); conic_opacity.w = opacity * scaling | Explicitly DISABLED at line 959; no opacity scaling; convolution_scaling_factor = 1.0 in 2D path | **NO** | HIGH |
| 2 | `new_aabb` | true — always uses `compute_aabb_view` in eval_3D path | VK always calls `computeAABBView` in eval_3D path (line 1033) | **YES** | N/A |
| 3 | `tight_opacity_bounding` | extent = min(3.33, sqrt(2 * opacity_power_threshold)); radius = extent * sqrt(lambda) | Hardcoded extent = 3.33 (line 1164); no opacity-based cutoff | **NO** | MEDIUM |
| 4 | 3D→2D covariance math | computeCov2D with limx/limy clamp; dilateCov2D with proper_ewa_scaling | Same J/W/T formula; precise qualifiers; dilation +0.3 fixed | Mostly YES | MEDIUM |
| 5 | Dilation factor | dilation_factor from det ratio; applied to opacity (proper_ewa_scaling=true) | dilation_factor computed correctly but application DISABLED at line 959 | **NO** | HIGH |
| 6 | Opacity scaling eval_3D | opacity *= proper_ewa_scaling ? dilation_factor : 1.0f | opacity_3d *= dilation_factor; // DISABLED (comment says proper_ewa_scaling=false) | **NO** | HIGH |
| 7 | Depth computation | viewspace Z; near-plane < 0.2f | p_view.z; near-plane <= 0.2 | MOSTLY YES (<= vs < negligible) | LOW |
| 8 | `rect_bounding` 2D path | extent_x = extent*sqrt(cov2D.x); extent_y = extent*sqrt(cov2D.z), capped to radius | Always isotropic: radius_f, radius_f | **NO** | MEDIUM |
| 9 | `rect_bounding` eval_3D path | asymmetric extents | VK uses asymmetric extent_x_3d/extent_y_3d | YES | — |
| 10 | Radii (tight_opacity_bounding) | extent = min(3.33, sqrt(2*opacity_power_threshold)) | 3.33 hardcoded | **NO** | MEDIUM |

## Detailed Findings

### 1. proper_ewa_scaling — DIVERGENCE (HIGH)

CUDA (forward_common.h lines 118–124):
```cpp
if (proper_ewa_scaling) {
    const float det_orig = cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1];
    convolution_scaling_factor = sqrt(max(0.000025f, det_orig / det_dilated));
} else {
    convolution_scaling_factor = 1.0f;
}
conic_opacity.w = opacity * convolution_scaling_factor;
```
With aaa.json: "proper_ewa_scaling": true, stored opacity = opacity * sqrt(det_orig / det_dilated).

VK (preprocess.comp line 959):
```glsl
// opacity_3d *= dilation_factor;  // DISABLED: proper_ewa_scaling=false
```
Comment says "default=false" but aaa.json has "proper_ewa_scaling": true.

PSNR impact: HIGH — all alpha values during rasterization are systematically wrong.

### 3. tight_opacity_bounding — DIVERGENCE (MEDIUM)

CUDA (forward.cu line 239):
```cpp
const float extent = tight_opacity_bounding ? min(3.33, sqrt(2.0f * opacity_power_threshold)) : 3.33f;
```
VK (preprocess.comp line 1164): `float radius_f = 3.33 * sqrt(lambda);` — always 3.33.

### 8. rect_bounding 2D path — DIVERGENCE (MEDIUM)

CUDA (forward.cu lines 256–258):
```cpp
const float extent_x = min(rect_bounding ? (extent * sqrt(cov2D.x)) : radius, radius);
const float extent_y = min(rect_bounding ? (extent * sqrt(cov2D.z)) : radius, radius);
```
VK: `rectMin(pix, radius_f, radius_f, ...)` — symmetric. eval_3D path already correct.

## Priority for Fixes

| Priority | Item |
|---|---|
| 1 (CRITICAL) | proper_ewa_scaling (#1, #5, #6) |
| 2 (MEDIUM) | tight_opacity_bounding (#3, #10) |
| 3 (MEDIUM) | rect_bounding 2D path (#8) |
| 4 (LOW) | Near-plane cull <= vs < (#7) |
