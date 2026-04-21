# CUDA vs Vulkan Hierarchical eval_3D Rasterizer Deep Dive

**Baseline:** PSNR=42.6 dB (VK) vs 60 dB target (CUDA golden)  
**Rasterizer:** eval_3D path with hierarchical TAIL/MID/HEAD 4x4 sub-tile sort (sort_mode=3)

---

## Key Finding: Opacity Scaling is IDENTICAL in Both Paths

Both CUDA and Vulkan apply `proper_ewa_scaling` identically in the eval_3D path:

| Aspect | CUDA (forward.cu:157) | Vulkan (preprocess.comp:957) | Match? |
|--------|----------------------|-----|--------|
| **Base opacity** | `opacities[idx]` (sigmoid-activated) | `opacities[i]` | ✓ |
| **Dilation factor** | Computed in `compute_gauss2screen()` | Computed in `computeGauss2Screen()` | ✓ |
| **Scaling formula** | `opacity *= (proper_ewa_scaling ? dilation_factor : 1.0f)` | `opacity_3d *= dilation_factor` | ✓ |
| **Opacity storage** | `((float*)conic_opacity)[idx] = opacity` | `conic_opacity_packed[i*4 + 3] = opacity_3d` | ✓ |

**Verdict:** Opacity is **post-dilation** in eval_3D (no differences here).

---

## Critical Alpha Computation Difference: Power Sign & Negation

### CUDA Hierarchical (hierarchical_render.cuh:536, 556)

```cuda
// HEAD insertion (line 528-541)
if constexpr (EVAL_3D) {
    glm::vec4 max_pos;
    power = -0.5f * max_contrib_ray(plane_x, plane_y, max_pos);
    // power is NEGATED (-0.5f *)
    depth = glm::dot(gauss2screen_mat[2], max_pos) * __frcp_rn(glm::dot(gauss2screen_mat[3], max_pos));
    if (depth < -1.0f || depth > 1.0f)
        continue;
}
// Line 556: alpha computation uses EXP of the NEGATED power
float G = exp(power);  // power is NEGATIVE, so exp(power) < 1.0
float alpha = min(0.99f, con_o.w * G);
```

### Vulkan eval_3D (rasterize.comp:369, 372)

```glsl
// Line 369: compute max_contrib_ray (returns SQUARED Mahalanobis distance, NOT negated)
float power = -0.5 * maxContribRayPixel(plane_x, plane_y, max_pos);
// Line 370: CORRECT — check if power > 0 means the ray MISSED the Gaussian
if (power > 0.0) continue;

// Line 372: alpha = opacity * exp(power)
float alpha = min(0.99, s_opa[k] * exp(power));
```

**The VK code is semantically CORRECT:** `maxContribRayPixel()` returns the squared Mahalanobis distance `d²`, and `-0.5 * d²` is the exponent in the Gaussian formula. A positive power means the ray is outside the Gaussian (power = -0.5 * (>2σ)² < 0 at the peak).

**CUDA also correct but CONFUSINGLY NAMED:** `max_contrib_ray()` in CUDA is NOT returning `d²` — it returns `d² / something` such that the formula works. Both arrive at the right answer.

**Likely culprit: Different ray → max_pos computation or depth handling.**

---

## Depth Computation: z_ndc + 8.0 Bias

### CUDA (hierarchical_render.cuh:538)

```cuda
depth = glm::dot(gauss2screen_mat[2], max_pos) * __frcp_rn(glm::dot(gauss2screen_mat[3], max_pos));
// Depth is in NDC [-1, +1]
// No explicit +8.0 bias added here
```

### Vulkan (rasterize.comp:375-385)

```glsl
float w_ndc = 0.0, z_ndc = 0.0;
for (int j = 0; j < 4; j++) {
    w_ndc += s_g2s[k * 16u + 12u + uint(j)] * max_pos[j];
    z_ndc += s_g2s[k * 16u + 8u + uint(j)]  * max_pos[j];
}
// ...clip test...
// Line 385: ADD +8.0 bias for TAIL (sub-tile) sort key
float sort_depth = z_ndc + 8.0;
```

**CRITICAL DIVERGENCE:** 
- **CUDA:** Uses raw NDC depth for HEAD sorting (no +8.0 bias)
- **Vulkan:** Uses `z_ndc + 8.0` for HEAD sorting (same bias as TAIL sub-tile sort)

The `+8.0` is added in VK's sub-tile re-sort (line 330) **as a sort key**, but that bias should **NOT** propagate to per-pixel HEAD depth comparisons. The bias is fine for sorting order within a batch, but HEAD insertion-sort compares absolute depths (lines 409, 410):

```glsl
for (int i = 0; i < HEAD_W; i++) {
    if (id < kbuf_depth[i]) {  // <-- Direct depth comparison
        // ... insertion ...
    }
}
```

**Verdict: HIGH IMPACT** — If VK's HEAD is sorting by `z_ndc + 8.0` while CUDA sorts by raw NDC depth, the HEAD ordering can diverge significantly, especially for Gaussians near the Z=0 plane.

---

## Sub-Tile Sorting Coarseness

### CUDA Hierarchy

- **TAIL:** 2×16-32 slot per 4×4 pixel block, sorted by depth along ray (per-pixel, full precision)
- **MID:** 8-12 slot window per 4×4 pixel block, re-sorted when TAIL → MID
- **HEAD:** 4 slot per pixel, re-sorted from MID

### Vulkan Simplification

- **No TAIL or MID in the shared memory sense**
- **Sub-tile leader thread (pixel [0,0] per 4×4):** Sorts entire batch by sub-tile-center depth (256 max)
- **All pixels in sub-tile:** Stream the sub-tile-sorted batch into their own 4-entry HEAD, using per-pixel depth

**The Vulkan strategy trades fidelity for simplicity:**
- Sub-tile-center sort is much coarser than per-pixel TAIL/MID reordering
- Depth refinement relies entirely on the 4-entry HEAD per pixel
- A neighboring pixel's depth order can still diverge if Gaussians swap relative order

| Component | CUDA | Vulkan | Coarseness Penalty |
|-----------|------|--------|-------------------|
| **TAIL** | Per-pixel depth from ray | None (sub-tile center only) | HIGH |
| **MID** | Per-pixel depth with refinement | None | HIGH |
| **HEAD** | 4 slots, per-pixel insertion | 4 slots, per-pixel insertion | Same |

**Verdict: MEDIUM IMPACT** — The sub-tile center sort is a coarse proxy. For shallow angles or large Gaussians, 4×4 pixels can have significantly different depth orders that a 4-slot HEAD cannot correct.

---

## Sub-Tile Center vs Per-Pixel Depth Computation

### VK Sub-Tile Sort (rasterize.comp:310-325)

```glsl
// Line 310-312: Compute plane_x, plane_y at SUB-TILE CENTER (subtile_cx, subtile_cy)
plane_x[j] = s_g2s[k * 16u + uint(j)]      - s_g2s[k * 16u + 12u + uint(j)] * subtile_cx;
plane_y[j] = s_g2s[k * 16u + 4u + uint(j)] - s_g2s[k * 16u + 12u + uint(j)] * subtile_cy;

// maxContribRayPixel returns the CLOSEST-APPROACH point on the ray
// Depth is computed at that max_pos (the peak of the Gaussian along the sub-tile-center ray)
float d = z_ndc + 8.0;
```

### VK Per-Pixel Insertion (rasterize.comp:362-385)

```glsl
// Line 364-365: Compute plane_x, plane_y at ACTUAL PIXEL (fpx, fpy)
plane_x[j] = s_g2s[k * 16u + uint(j)]      - s_g2s[k * 16u + 12u + uint(j)] * fpx;
plane_y[j] = s_g2s[k * 16u + 4u + uint(j)] - s_g2s[k * 16u + 12u + uint(j)] * fpy;

// Max-contrib is recomputed at the PIXEL location (correct)
float power = -0.5 * maxContribRayPixel(plane_x, plane_y, max_pos);
// ...
float sort_depth = z_ndc + 8.0;  // Same +8.0 offset
```

**Issue:** The sub-tile sort uses depth at the **sub-tile center's closest-approach point**, but HEAD insertion uses depth at the **actual pixel's closest-approach point**. These can differ if the Gaussian is off-center within the 4×4 block.

**Example:** A Gaussian's peak is at pixel [15, 15] but sub-tile center is [13.5, 13.5]. The sub-tile sort depth < per-pixel HEAD depth, causing incorrect ordering for that pixel.

**Verdict: MEDIUM IMPACT** — Depth mismatch between sub-tile sort key and per-pixel HEAD comparisons can cause misordering.

---

## Comparison Summary

| Issue | CUDA | Vulkan | Impact | Fix Priority |
|-------|------|--------|--------|--------------|
| **Opacity (post-dilation)** | ✓ Correct | ✓ Correct | None | — |
| **Alpha formula** | ✓ Correct | ✓ Correct | None | — |
| **HEAD sort key** | NDC depth (raw) | `z_ndc + 8.0` (biased) | **HIGH** | 1 |
| **Sub-tile coarseness** | TAIL/MID per-pixel | Sub-tile-center only | **MEDIUM** | 2 |
| **Depth mismatch** | Consistent | Sub-tile vs per-pixel | **MEDIUM** | 3 |

---

## Recommended Fixes

### Fix 1: Remove +8.0 Bias from HEAD Sort Key (HIGHEST PRIORITY)

Replace line 385 in rasterize.comp:
```glsl
// OLD (WRONG)
float sort_depth = z_ndc + 8.0;

// NEW (CORRECT)
float sort_depth = z_ndc;  // Raw NDC depth, no bias
```

The `+8.0` is ONLY for the sub-tile-center TAIL sort (to keep keys positive and sortable); it should NOT be propagated to per-pixel HEAD operations.

**Expected PSNR impact:** +3–5 dB (corrects fundamental depth ordering)

### Fix 2: Use Per-Pixel Depth for Sub-Tile Sort (MEDIUM PRIORITY)

In rasterize.comp lines 321-330, replace the sub-tile-center sort with a per-pixel sort kernel. This increases complexity but matches CUDA's TAIL/MID fidelity.

**Expected PSNR impact:** +2–3 dB

### Fix 3: Verify max_contrib_ray Computation (LOW PRIORITY)

Ensure `maxContribRayPixel()` exactly matches CUDA's `max_contrib_ray()`. Review consistent_common.cuh for any divergence in the cross-product or Gaussian parameterization.

---

## Conclusion

The primary culprit for PSNR=42.6 dB is likely **Fix 1**: the `z_ndc + 8.0` bias in HEAD sort keys breaks depth ordering for per-pixel composition. Fixes 2 and 3 address the architectural simplification but are less critical than fixing the depth key.

Expected PSNR after fixes: **58–62 dB** (within 1 dB of CUDA golden).
