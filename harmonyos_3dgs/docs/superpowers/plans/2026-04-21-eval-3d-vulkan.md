# eval_3D=true Vulkan Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the AAA-Gaussians eval_3D=true code path in the Vulkan forward pipeline (preprocess, scatter, rasterize) so the Vulkan renderer produces output matching the CUDA golden reference.

**Architecture:** The eval_3D path replaces 2D conic splatting with full 3D Gaussian evaluation. The preprocess shader computes a 4x4 gauss2screen matrix per Gaussian (instead of 2D conics). The scatter shader uses per-tile depth keys via `depthAlongRay`. The rasterizer evaluates each Gaussian per-pixel via plane intersection (`maxContribRay`) and uses a StopThePop k-buffer (K=16) for correct per-pixel depth ordering. All three shaders branch on `spec_eval_3D` specialization constant. The CPU reference in `preprocessor_cpu.cpp`, `tile_binner_cpu.cpp`, and `rasterizer_cpu.cpp` provides the exact algorithm to port.

**Tech Stack:** Vulkan 1.3 compute shaders (GLSL 450), C++17, CMake

**Numerical target:** Match CUDA AAA-Gaussians golden render with eval_3D=true, proper_ewa_scaling=true, sort_mode=3, sort_order=3. Current gap: ~12.37 dB PSNR (due to algorithm difference: 2D conic vs 3D evaluation).

**Bug review lessons (from debug worktree `dev_notes/bug_review_2026-04-21.md`):**
1. **Pixel center convention**: Per-pixel evaluation MUST use `(px + 0.5, py + 0.5)` — pixel centers, not corners. The viewport matrix has a -0.5 offset.
2. **gauss2screen row-major**: Construction is `transpose(viewport * viewproj * gauss2world)`. Any row/col mismatch → incorrect Mahalanobis distance.
3. **AABB center vs NDC center**: Tile assignment uses NDC projection center (`ndc2Pix`); AABB extent only determines radius. Using AABB center causes 16x16 block artifacts.
4. **kBuffer depth instability**: `maxContribRayPixel` divides by `dd` (cross product magnitude squared). When dd→0 (Gaussian center near pixel ray), must guard: `if (dd < 1e-8) fallback to view_space_z`.
5. **Dilation factor is mandatory**: `opacity *= sqrt(det_original / det_dilated)`. Without it, far-distance Gaussians become over-transparent.
6. **`precise` qualifier**: Must use for depth and pixel coordinates to avoid FMA-induced sort order divergence (established in 2D path).
7. **exp() not native_exp()**: Use standard `exp()` in GLSL (≤3 ULP). `native_exp` errors accumulate through 2000+ Gaussians/tile.

---

## File Structure

### Files to modify

| File | Responsibility | Changes |
|------|---------------|---------|
| `include/vulkan/preprocess_bindings.h` | Binding constants for all shaders | Add bindings 19-21 (preprocess), 8-11 (scatter), 9-14 (rasterize); add spec constants |
| `src/vulkan/shaders/preprocess.comp` | Per-Gaussian projection | Add eval_3D branch: gauss2screen, frustum cull, AABB, new output SSBOs |
| `src/vulkan/preprocess_pass.cpp` | Preprocess pipeline wrapper | Increase binding count from 19 to 22; wire new SSBOs |
| `include/vulkan/preprocess_pass.h` | Preprocess pass header | Add 3 new buffer fields to `Buffers` struct |
| `src/vulkan/preprocessor_vulkan.cpp` | Preprocess adapter (Layer-1 + Layer-2) | Remove eval_3D hard error; allocate/bind/download new buffers; make spec_eval_3D configurable |
| `include/vulkan/preprocessor_vulkan.h` | Preprocess adapter header | Add buffer getters for gauss2screen, cov3D_inv, mean_offset |
| `src/vulkan/shaders/scatter.comp` | Key-value scatter | Add depthAlongRay per-tile depth key; new input SSBOs + UBO + spec constant |
| `src/vulkan/scatter_pass.cpp` | Scatter pipeline wrapper | Add spec constant; increase binding count; wire new buffers |
| `include/vulkan/tile_binner_passes.h` | Scatter pass header | Add new buffer fields + spec constant |
| `src/vulkan/tile_binner_vulkan.cpp` | Tile binner adapter | Pass new buffers through to scatter; compute inverse_vp UBO |
| `include/vulkan/tile_binner_vulkan.h` | Tile binner adapter header | Update prepare_record signature |
| `src/vulkan/shaders/rasterize.comp` | Per-tile alpha blending | Add eval_3D branch: maxContribRay, k-buffer (K=16), depthAlongRay; new SSBOs |
| `src/vulkan/rasterize_pass.cpp` | Rasterize pipeline wrapper | Add spec constant; increase binding count; wire new buffers |
| `include/vulkan/rasterize_pass.h` | Rasterize pass header | Add new buffer fields + spec constant |
| `src/vulkan/rasterizer_vulkan.cpp` | Rasterizer adapter | Pass gauss2screen/cov3D_inv/mean_offset/opacities/inverse_vp to rasterize; handle eval_3D |
| `include/vulkan/rasterizer_vulkan.h` | Rasterizer adapter header | Update prepare_record signature |
| `src/vulkan/vk_render_main.cpp` | Vulkan render CLI | Set eval_3D=true |
| `src/vulkan/renderer_vulkan.cpp` | Full pipeline orchestrator | Pass eval_3D config; wire new buffers between stages |
| `tests/test_preprocess_pass_vk.cpp` | Preprocess GPU tests | Add eval_3D=true test case |

### Reference files (read-only, not modified)

| File | Used for |
|------|----------|
| `src/cpu/preprocessor_cpu.cpp:65-188` | CPU eval_3D preprocess reference |
| `src/cpu/rasterizer_cpu.cpp:61-135` | CPU eval_3D k-buffer rasterizer reference |
| `src/cpu/tile_binner_cpu.cpp:86-98` | CPU per-tile depth key reference |
| `src/math_utils.cpp:148-651` | CPU math functions: computeGauss2Screen, maxContribRay, computeAABBScreen/View, depthAlongRay, pixelToWorldDir, computeCov3DInv |

---

## Task 1: Binding Infrastructure

**Files:**
- Modify: `include/vulkan/preprocess_bindings.h`
- Modify: `include/vulkan/preprocess_pass.h`
- Modify: `include/vulkan/tile_binner_passes.h`
- Modify: `include/vulkan/rasterize_pass.h`

This task adds all new binding constants, push constant updates, and Buffers struct fields needed by later tasks. No shader or C++ implementation changes — just the shared interface definitions.

- [ ] **Step 1: Add preprocess bindings 19-21 to preprocess_bindings.h**

In `namespace preprocess_bind`, after `COV2D_DET_CACHE = 18`, add:

```cpp
constexpr uint32_t GAUSS2SCREEN = 19; // WO float[N*16] gauss2screen row-major (eval_3D only)
constexpr uint32_t COV3D_INV    = 20; // WO float[N*6]  inverse 3D covariance upper tri (eval_3D only)
constexpr uint32_t MEAN_OFFSET  = 21; // WO float[N*3]  world-space (pos - cam_pos) (eval_3D only)
```

- [ ] **Step 2: Add scatter bindings 8-11 to preprocess_bindings.h**

In `namespace scatter_bind`, after `RADIUS_F = 7`, add:

```cpp
constexpr uint32_t COV3D_INV    = 8;  // RO float[N*6]  inverse 3D covariance (eval_3D per-tile depth)
constexpr uint32_t MEAN_OFFSET  = 9;  // RO float[N*3]  world-space (pos - cam_pos)
constexpr uint32_t SCATTER_UBO  = 10; // UB ScatterUBO  (inverse_vp + eval_3D flag)
```

Add ScatterPushConstants update — extend with eval_3D flag or use the existing `_pad` field:

```cpp
// Replace existing ScatterPushConstants:
struct ScatterPushConstants {
    uint32_t num_gaussians;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t eval_3D;   // 1 = use depthAlongRay per-tile, 0 = use view-space z
};
static_assert(sizeof(ScatterPushConstants) == 24, "ScatterPushConstants must be 24 bytes");
```

Add ScatterUBO for inverse viewproj matrix + camera info:

```cpp
struct alignas(16) ScatterUBO {
    float inverse_vp[16];   // 64B: inverse viewproj matrix (column-major)
    float cam_pos[4];       // 16B: xyz=cam_pos, w=0
    float img_size[4];      // 16B: x=width, y=height, z=0, w=0
};
static_assert(sizeof(ScatterUBO) == 96, "ScatterUBO must be 96 bytes (std140)");
```

- [ ] **Step 3: Add rasterize bindings 9-14 to preprocess_bindings.h**

In `namespace rasterize_bind`, after `RASTER_UBO = 8`, add:

```cpp
constexpr uint32_t GAUSS2SCREEN     = 9;  // RO float[N*16] gauss2screen row-major
constexpr uint32_t OPACITIES_2D     = 10; // RO float[N]    pre-dilated opacity (eval_3D uses separately)
constexpr uint32_t COV3D_INV        = 11; // RO float[N*6]  inverse 3D covariance
constexpr uint32_t MEAN_OFFSET      = 12; // RO float[N*3]  world-space offset
constexpr uint32_t RASTER_EVAL3D_UBO = 13; // UB  RasterEval3DUBO (inverse_vp + cam info)
```

Add RasterEval3DUBO:

```cpp
struct alignas(16) RasterEval3DUBO {
    float inverse_vp[16];   // 64B: inverse viewproj matrix
    float cam_pos[4];       // 16B: xyz=cam_pos, w=0
    float img_size[4];      // 16B: x=width, y=height, z=0, w=0
};
static_assert(sizeof(RasterEval3DUBO) == 96, "RasterEval3DUBO must be 96 bytes (std140)");
```

Add spec constant namespace for rasterize:

```cpp
namespace rasterize_spec {
constexpr uint32_t EVAL_3D = 0;  // 1 = eval_3D k-buffer path, 0 = 2D conic path
}
```

- [ ] **Step 4: Update PreprocessPass::Buffers in preprocess_pass.h**

Add 3 new fields after `cov2D_det_cache`:

```cpp
VkBuffer gauss2screen;   // WO float[N*16]  (binding 19, eval_3D only)
VkBuffer cov3D_inv;      // WO float[N*6]   (binding 20, eval_3D only)
VkBuffer mean_offset;    // WO float[N*3]   (binding 21, eval_3D only)
```

- [ ] **Step 5: Update ScatterPass::Buffers in tile_binner_passes.h**

Add 3 new fields after `radius_f`:

```cpp
VkBuffer cov3D_inv;      // RO float[N*6]   (binding 8, eval_3D only)
VkBuffer mean_offset;    // RO float[N*3]   (binding 9, eval_3D only)
VkBuffer scatter_ubo;    // UB ScatterUBO   (binding 10, eval_3D only)
```

- [ ] **Step 6: Update RasterizePass::Buffers in rasterize_pass.h**

Add 5 new fields after `raster_ubo`:

```cpp
VkBuffer gauss2screen;      // RO float[N*16]  (binding 9, eval_3D only)
VkBuffer opacities_2d;      // RO float[N]     (binding 10, eval_3D only)
VkBuffer cov3D_inv;         // RO float[N*6]   (binding 11, eval_3D only)
VkBuffer mean_offset;       // RO float[N*3]   (binding 12, eval_3D only)
VkBuffer raster_eval3d_ubo; // UB  96 bytes    (binding 13, eval_3D only)
```

- [ ] **Step 7: Build and verify no regressions**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
```

Expected: compiles cleanly (new struct fields are additive, no callers changed yet).

- [ ] **Step 8: Commit**

```bash
git add include/vulkan/preprocess_bindings.h include/vulkan/preprocess_pass.h \
        include/vulkan/tile_binner_passes.h include/vulkan/rasterize_pass.h
git commit -m "feat: add eval_3D binding constants and buffer struct fields

Add preprocess bindings 19-21 (gauss2screen, cov3D_inv, mean_offset),
scatter bindings 8-10 (cov3D_inv, mean_offset, scatter_ubo),
rasterize bindings 9-13 (gauss2screen, opacities_2d, cov3D_inv,
mean_offset, raster_eval3d_ubo).

Add ScatterUBO and RasterEval3DUBO structs for inverse viewproj.
Add rasterize_spec::EVAL_3D specialization constant ID."
```

---

## Task 2: preprocess.comp eval_3D Branch

**Files:**
- Modify: `src/vulkan/shaders/preprocess.comp`
- Reference: `src/cpu/preprocessor_cpu.cpp:65-188` (CPU eval_3D preprocess)
- Reference: `src/math_utils.cpp:474-582` (computeGauss2Screen)
- Reference: `src/math_utils.cpp:253-340` (maxContribGaussianFrustum3D)
- Reference: `src/math_utils.cpp:342-472` (computeAABBScreen, computeAABBView)
- Reference: `src/math_utils.cpp:590-614` (computeCov3DInv)
- Reference: `src/math_utils.cpp:186-248` (maxContribRay, maxContribPlane, etc.)

This task ports the eval_3D=true preprocess path from CPU to GLSL. The shader already has `spec_eval_3D` specialization constant (line 31) — we add the actual code path.

**CRITICAL numerical notes:**
- Use `precise` qualifier for depth and pixel coordinate computations (established pattern in 2D path).
- `-ffp-contract=off` is enforced on CPU; shader `precise` matches this behavior.
- gauss2screen is stored ROW-MAJOR (matches AAA-Gaussians convention; CPU `computeGauss2Screen` returns row-major).
- The CPU uses column-major mat4 multiply (`mat4Mul`) to build gauss2screen; the GLSL version must produce identical results.
- `ALPHA_THRESHOLD = 1.0/255.0` — same as 2D path.
- Near-plane cull is `< 0.2` (strict less-than) for eval_3D, vs `<= 0.2` for 2D.

- [ ] **Step 1: Add new output SSBOs at bindings 19-21**

After the existing ForwardCache SSBOs (bindings 14-18), add:

```glsl
// ---- eval_3D output SSBOs (bindings 19..21) --------------------------------
layout(std430, set = 0, binding = 19) writeonly buffer BufGauss2Screen { float gauss2screen_out[]; };  // [N*16]
layout(std430, set = 0, binding = 20) writeonly buffer BufCov3DInv     { float cov3D_inv_out[];     };  // [N*6]
layout(std430, set = 0, binding = 21) writeonly buffer BufMeanOffset   { float mean_offset_out[];   };  // [N*3]
```

- [ ] **Step 2: Add mat4 multiply helper**

Port `mat4Mul` from math_utils.cpp:166-174. This multiplies two column-major 4x4 matrices:

```glsl
// Column-major 4x4 matrix multiply: out = A * B
// A[col][row] in GLSL mat4 convention.
// We work with float[16] arrays stored column-major: arr[col*4+row].
void mat4Mul16(float A[16], float B[16], out float result[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0.0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[col*4 + k];
            result[col*4 + row] = sum;
        }
}
```

- [ ] **Step 3: Add computeGauss2Screen function**

Port from `math_utils.cpp:474-582`. Returns dilation_factor. Parameters match the CPU function signature.

Key steps (follow CPU code line-by-line):
1. Build rotation matrix R from quaternion (reuse existing `quat2mat`)
2. Compute dilated scale with MIP filter (kernel_size=0.3, filter_3d from input)
3. Compute dilation_factor from determinant ratio
4. Build L = transpose(S*R) where S = diag(scale_dilated * sqrt(scale_modifier))
5. Build gauss2world column-major 4x4: columns = L rows, last col = mean3D
6. Build viewport matrix (maps NDC to pixels)
7. world2screen = viewport * viewproj
8. gauss2screen_colmajor = world2screen * gauss2world
9. Transpose to row-major for output

The function signature:

```glsl
float computeGauss2Screen(vec3 mean3D, vec3 scale, vec4 rot,
                           float scale_mod, vec3 cam_pos,
                           float focal, float kernel_size, float filter_3d_val,
                           float W, float H,
                           out float g2s_rowmajor[16],
                           out float g2v_rowmajor[16])
```

Use `cam.projmatrix` for viewproj_matrix and `cam.viewmatrix` for view_matrix (both are in the CameraUBO).

- [ ] **Step 4: Add computeCov3DInv function**

Port from `math_utils.cpp:590-614`:

```glsl
void computeCov3DInv(vec3 scale_dilated, float scale_mod, vec4 rot,
                      out float cov3D_inv[6]) {
    float R[9];
    quat2mat(rot, R);
    float sm = sqrt(scale_mod);
    float inv_sd2[3];
    for (int j = 0; j < 3; j++) {
        float s = scale_dilated[j] * sm;
        inv_sd2[j] = 1.0 / (s * s);
    }
    // cov3D_inv = R * diag(1/sd^2) * R^T (upper triangle)
    int idx = 0;
    for (int r = 0; r < 3; r++) {
        for (int c = r; c < 3; c++) {
            float val = 0.0;
            for (int k = 0; k < 3; k++)
                val += RMC(R, k, r) * inv_sd2[k] * RMC(R, k, c);
            cov3D_inv[idx] = val;
            idx++;
        }
    }
}
```

- [ ] **Step 5: Add maxContribRay and supporting functions**

Port from `math_utils.cpp:186-248`. These functions are used for 3D frustum culling and AABB computation.

```glsl
// maxContribRay: squared Mahalanobis distance along ray from plane intersection
float maxContribRay(float plane_a[4], float plane_b[4], out float max_pos[4]) {
    // cross(plane_a.xyz, plane_b.xyz)
    float dx = plane_a[1]*plane_b[2] - plane_a[2]*plane_b[1];
    float dy = plane_a[2]*plane_b[0] - plane_a[0]*plane_b[2];
    float dz = plane_a[0]*plane_b[1] - plane_a[1]*plane_b[0];
    // m = plane_a.w * plane_b.xyz - plane_a.xyz * plane_b.w
    float mx = plane_a[3]*plane_b[0] - plane_a[0]*plane_b[3];
    float my = plane_a[3]*plane_b[1] - plane_a[1]*plane_b[3];
    float mz = plane_a[3]*plane_b[2] - plane_a[2]*plane_b[3];
    float dd = dx*dx + dy*dy + dz*dz;
    float m_div_dd[3] = float[3](mx/dd, my/dd, mz/dd);
    max_pos[0] = dy*m_div_dd[2] - dz*m_div_dd[1];
    max_pos[1] = dz*m_div_dd[0] - dx*m_div_dd[2];
    max_pos[2] = dx*m_div_dd[1] - dy*m_div_dd[0];
    max_pos[3] = 1.0;
    return mx*m_div_dd[0] + my*m_div_dd[1] + mz*m_div_dd[2];
}
```

Also port: `maxContribPlane`, `inScreenRange` (both overloads), `maxContribRayScreen`, `maxContribGaussianFrustum3D`, `computeAABBScreen`, `computeAABBView`.

These are all direct translations from `math_utils.cpp:210-472`. Follow the CPU code exactly.

- [ ] **Step 6: Add eval_3D branch in main()**

In the `main()` function, after the near-plane cull (line 409), add an `if (spec_eval_3D == 1u)` branch. The entire eval_3D branch replaces steps 5-15 of the 2D path. Follow `preprocessor_cpu.cpp:65-188` exactly:

```glsl
if (spec_eval_3D == 1u) {
    // === AAA-Gaussians 3D evaluation path ===
    // Near-plane cull uses strict < (not <=)
    if (p_view.z < 0.2) return;

    float opacity = opacities[i];
    float focal = max(focal_x, focal_y);
    float filter_3d_val = filter_3D[i];

    // 1. Compute gauss2screen + dilation_factor
    float g2s[16], g2v[16];
    float dilation_factor = computeGauss2Screen(
        position, scale, rot, pc.scale_modifier, cam_pos,
        focal, 0.3, filter_3d_val, width, height, g2s, g2v);

    opacity *= dilation_factor;
    if (opacity < ALPHA_THRESHOLD) return;

    float opacity_power_threshold = log(opacity / ALPHA_THRESHOLD);
    float cutoff = min(11.11, 2.0 * opacity_power_threshold);

    // 2. Camera-inside-ellipsoid test
    // ... (port from preprocessor_cpu.cpp:90-133)

    // 3. 3D frustum culling via maxContribGaussianFrustum3D
    // ... (port from preprocessor_cpu.cpp:136-141)

    // 4. NDC projection for tile assignment
    // ... (reuse existing precise p_hom computation)

    // 5. AABB screen/view for radius
    // ... (port from preprocessor_cpu.cpp:152-161)

    // 6. Tile rect + SH color (same as 2D)
    // ... (port from preprocessor_cpu.cpp:164-176)

    // 7. Store outputs
    depths[i] = p_view.z;
    radii[i] = my_radius;
    means2D[i*2u] = pixel_x;
    means2D[i*2u+1u] = pixel_y;
    // Pack opacity into conic_opacity_packed (conics unused in 3D mode)
    conic_opacity_packed[i*4u+3u] = opacity;
    rgb[i*3u] = color.x; rgb[i*3u+1u] = color.y; rgb[i*3u+2u] = color.z;
    tiles_touched[i] = n_tiles;

    // Store gauss2screen (row-major, 16 floats)
    for (int j = 0; j < 16; j++) gauss2screen_out[i*16u+uint(j)] = g2s[j];
    // Store cov3D_inv (6 floats)
    // ... (call computeCov3DInv with scale_dilated)
    // Store mean_offset (3 floats)
    mean_offset_out[i*3u]   = position.x - cam_pos.x;
    mean_offset_out[i*3u+1u] = position.y - cam_pos.y;
    mean_offset_out[i*3u+2u] = position.z - cam_pos.z;

} else {
    // === Standard 2D path (existing code, unchanged) ===
    // ... existing code from lines 410-541
}
```

The complete GLSL for the eval_3D branch must faithfully translate `preprocessor_cpu.cpp:65-188`. Every line of CPU code has a corresponding GLSL line. The key algorithmic steps are:
1. Near-plane cull (`p_view.z < 0.2`)
2. `computeGauss2Screen` → gauss2screen + dilation_factor
3. Opacity dilate + ALPHA_THRESHOLD cull
4. Camera-inside-ellipsoid test (campos_gauss → dist_sq < cutoff → skip)
5. `maxContribGaussianFrustum3D` → frustum cull
6. NDC projection → pixel coords
7. `computeAABBScreen` / `computeAABBView` → extent → radius → tile rect
8. SH color evaluation (reuse `shToRGB`)
9. Write all outputs including gauss2screen[N*16], cov3D_inv[N*6], mean_offset[N*3]

- [ ] **Step 7: Zero-fill eval_3D outputs for culled Gaussians**

At the top of main(), alongside existing zero-fill (lines 385-395), add:

```glsl
// Zero-fill eval_3D outputs for culled paths.
if (spec_eval_3D == 1u) {
    for (uint j = 0u; j < 16u; j++) gauss2screen_out[i*16u+j] = 0.0;
    for (uint j = 0u; j < 6u; j++)  cov3D_inv_out[i*6u+j] = 0.0;
    mean_offset_out[i*3u] = 0.0;
    mean_offset_out[i*3u+1u] = 0.0;
    mean_offset_out[i*3u+2u] = 0.0;
}
```

- [ ] **Step 8: Build and verify shader compiles**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
```

The build compiles GLSL to SPIR-V via glslangValidator — any syntax errors caught here.

- [ ] **Step 9: Commit**

```bash
git add src/vulkan/shaders/preprocess.comp
git commit -m "feat: add eval_3D=true branch in preprocess.comp

Port AAA-Gaussians 3D preprocess path from CPU to GLSL:
- computeGauss2Screen: 4x4 matrix with MIP filter dilation
- computeCov3DInv: inverse 3D covariance
- maxContribGaussianFrustum3D: 3D frustum culling
- computeAABBScreen/View: screen-space bounding box
- Camera-inside-ellipsoid test
- Output gauss2screen[N*16], cov3D_inv[N*6], mean_offset[N*3]

Branched on spec_eval_3D specialization constant."
```

---

## Task 3: PreprocessorVulkan eval_3D Plumbing

**Files:**
- Modify: `src/vulkan/preprocess_pass.cpp`
- Modify: `src/vulkan/preprocessor_vulkan.cpp`
- Modify: `include/vulkan/preprocessor_vulkan.h`
- Test: `tests/test_preprocess_pass_vk.cpp`

This task wires up the C++ side: makes `spec_eval_3D` configurable from `RenderConfig`, allocates the 3 new output buffers, downloads outputs, and exposes buffer handles for downstream stages.

- [ ] **Step 1: Update PreprocessPass to accept 22 bindings**

In `preprocess_pass.cpp`, change binding count from 19 to 22:

```cpp
// Line 58: change 19 → 22
std::vector<VkDescriptorType> binding_types(22,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
```

Add bind_buffers entries for the 3 new SSBOs:

```cpp
// After cov2D_det_cache binding (line 107):
pipeline_->update_ssbo(descriptor_set_, preprocess_bind::GAUSS2SCREEN, b.gauss2screen);
pipeline_->update_ssbo(descriptor_set_, preprocess_bind::COV3D_INV,    b.cov3D_inv);
pipeline_->update_ssbo(descriptor_set_, preprocess_bind::MEAN_OFFSET,  b.mean_offset);
```

- [ ] **Step 2: Make PreprocessorVulkan accept eval_3D config**

In `preprocessor_vulkan.cpp`, change the constructor to accept eval_3D:

```cpp
PreprocessorVulkan::PreprocessorVulkan(VulkanContext& ctx, bool eval_3D)
    : ctx_(ctx), eval_3D_(eval_3D) {
    pass_ = std::make_unique<PreprocessPass>(ctx_,
                                             /*spec_training=*/1u,
                                             /*spec_eval_3D=*/eval_3D ? 1u : 0u);
}
```

Update the header to add `eval_3D_` member and new constructor signature. Remove the eval_3D hard error from `process()` and `prepare_record()`.

- [ ] **Step 3: Allocate eval_3D output buffers in process()**

In `process()`, after existing buffer allocation, add conditional allocation:

```cpp
// eval_3D output buffers (always allocated for descriptor binding validity)
auto g2s_buf = std::make_unique<VulkanBuffer>(
    ctx_, static_cast<VkDeviceSize>(N) * 16 * sizeof(float),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
auto c3i_buf = std::make_unique<VulkanBuffer>(
    ctx_, static_cast<VkDeviceSize>(N) * 6 * sizeof(float),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
auto mo_buf = std::make_unique<VulkanBuffer>(
    ctx_, static_cast<VkDeviceSize>(N) * 3 * sizeof(float),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
```

Wire into Buffers struct:

```cpp
b.gauss2screen = g2s_buf->handle();
b.cov3D_inv    = c3i_buf->handle();
b.mean_offset  = mo_buf->handle();
```

- [ ] **Step 4: Download eval_3D outputs and populate PreprocessOutput**

After dispatch, when `eval_3D_` is true:

```cpp
if (eval_3D_) {
    out.gauss2screen = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 16);
    out.cov3D_inv    = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 6);
    out.mean_offset  = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 3);
    out.eval_3D      = true;
    g2s_buf->download(out.gauss2screen, static_cast<std::size_t>(N) * 16 * sizeof(float));
    c3i_buf->download(out.cov3D_inv,    static_cast<std::size_t>(N) * 6 * sizeof(float));
    mo_buf->download(out.mean_offset,   static_cast<std::size_t>(N) * 3 * sizeof(float));
}
```

Also extract per-Gaussian opacity from conic_opacity_packed[i*4+3] into opacities_2d[i] (same as 2D path — the packing is identical, just conics are unused).

- [ ] **Step 5: Update prepare_record() for eval_3D**

Mirror the process() changes in prepare_record(): allocate new buffers in record_bufs_, add RecBufIdx entries, wire into PreprocessPass::Buffers, add buffer getters.

Add new RecBufIdx entries:

```cpp
kGauss2Screen,   // binding 19
kCov3DInv,       // binding 20
kMeanOffset,     // binding 21
kRecBufCount,    // update count
```

Add buffer getters:

```cpp
VkBuffer gauss2screen_buffer() const;
VkBuffer cov3D_inv_buffer() const;
VkBuffer mean_offset_buffer() const;
```

- [ ] **Step 6: Add eval_3D preprocess test**

In `tests/test_preprocess_pass_vk.cpp`, add a test that verifies the eval_3D=true preprocess output matches the CPU reference. Use a single Gaussian at a known position:

```cpp
TEST(PreprocessPass, Eval3D_BasicOutput) {
    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan device";

    // Single Gaussian with filter_3D
    GaussianData g = makeSingleGaussian();
    g.filter_3D = new float[1]{0.01f};

    // Build PreprocessorVulkan with eval_3D=true
    PreprocessorVulkan pp(ctx, /*eval_3D=*/true);
    RenderConfig cfg = makeBasicCfg();
    cfg.eval_3D = true;

    FrameAllocator alloc(4 * 1024 * 1024);
    auto out = pp.process(g, makePerspCamera(), cfg, alloc);

    // Verify eval_3D outputs are non-null
    ASSERT_NE(out.gauss2screen, nullptr);
    ASSERT_NE(out.cov3D_inv, nullptr);
    ASSERT_NE(out.mean_offset, nullptr);
    EXPECT_TRUE(out.eval_3D);

    // Compare with CPU reference
    PreprocessorCPU cpu_pp;
    FrameAllocator cpu_alloc(4 * 1024 * 1024);
    auto cpu_out = cpu_pp.process(g, makePerspCamera(), cfg, cpu_alloc);

    // gauss2screen should match within tolerance
    for (int j = 0; j < 16; j++) {
        EXPECT_NEAR(out.gauss2screen[j], cpu_out.gauss2screen[j], 1e-4f)
            << "gauss2screen[" << j << "] mismatch";
    }
    // radii, means2D, depths should match
    EXPECT_EQ(out.radii[0], cpu_out.radii[0]);
    EXPECT_NEAR(out.depths[0], cpu_out.depths[0], 1e-5f);

    delete[] g.filter_3D;
}
```

- [ ] **Step 7: Build and test**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
ctest --test-dir build -R PreprocessPass -V --output-on-failure
```

- [ ] **Step 8: Commit**

```bash
git add src/vulkan/preprocess_pass.cpp src/vulkan/preprocessor_vulkan.cpp \
        include/vulkan/preprocessor_vulkan.h tests/test_preprocess_pass_vk.cpp
git commit -m "feat: wire eval_3D=true through PreprocessorVulkan

- Make spec_eval_3D configurable from RenderConfig
- Allocate/bind/download gauss2screen, cov3D_inv, mean_offset buffers
- Add Eval3D_BasicOutput test comparing Vulkan vs CPU preprocess
- Expose buffer getters for downstream pipeline stages"
```

---

## Task 4: scatter.comp Per-Tile Depth Key

**Files:**
- Modify: `src/vulkan/shaders/scatter.comp`
- Reference: `src/cpu/tile_binner_cpu.cpp:86-98` (CPU per-tile depth key)
- Reference: `src/math_utils.cpp:616-651` (depthAlongRay, pixelToWorldDir)

When eval_3D=true, the scatter shader computes per-tile depth keys using `depthAlongRay` instead of using the view-space z depth. This improves sort quality for anisotropic Gaussians.

- [ ] **Step 1: Add new input bindings to scatter.comp**

After existing binding 7 (radius_f):

```glsl
layout(std430, set = 0, binding = 8) readonly buffer Cov3DInvIn   { float cov3D_inv[];   };  // [N*6]
layout(std430, set = 0, binding = 9) readonly buffer MeanOffsetIn { float mean_offset[];  };  // [N*3]

layout(std140, set = 0, binding = 10) uniform ScatterUBO {
    mat4 inverse_vp;       // 64B: inverse viewproj (column-major)
    vec4 cam_pos_pad;      // 16B: xyz=cam_pos, w=0
    vec4 img_size_pad;     // 16B: x=width, y=height
} scatter_ubo;
```

Update push constants to include eval_3D flag (replace `_pad`):

```glsl
layout(push_constant) uniform PC {
    uint num_gaussians;
    uint num_tiles_x;
    uint num_tiles_y;
    uint tile_w;
    uint tile_h;
    uint eval_3D;   // 1 = use depthAlongRay per-tile
} pc;
```

- [ ] **Step 2: Add depthAlongRay and pixelToWorldDir functions**

Port from `math_utils.cpp:616-651`:

```glsl
float depthAlongRay(uint idx, vec3 viewdir) {
    // Sigma_inv * viewdir (symmetric 3x3 upper triangle)
    float c0 = cov3D_inv[idx*6u+0u], c1 = cov3D_inv[idx*6u+1u], c2 = cov3D_inv[idx*6u+2u];
    float c3 = cov3D_inv[idx*6u+3u], c4 = cov3D_inv[idx*6u+4u], c5 = cov3D_inv[idx*6u+5u];
    vec3 Sv = vec3(
        c0*viewdir.x + c1*viewdir.y + c2*viewdir.z,
        c1*viewdir.x + c3*viewdir.y + c4*viewdir.z,
        c2*viewdir.x + c4*viewdir.y + c5*viewdir.z);
    vec3 mo = vec3(mean_offset[idx*3u], mean_offset[idx*3u+1u], mean_offset[idx*3u+2u]);
    float num = dot(mo, Sv);
    float den = dot(viewdir, Sv);
    if (abs(den) < 1e-10) return length(mo);
    return num / den;
}

vec3 pixelToWorldDir(float px, float py) {
    float W = scatter_ubo.img_size_pad.x;
    float H = scatter_ubo.img_size_pad.y;
    float ndc_x = (2.0 * px + 1.0) / W - 1.0;
    float ndc_y = (2.0 * py + 1.0) / H - 1.0;
    vec4 p = vec4(ndc_x, ndc_y, 0.0, 1.0);
    vec4 wp = scatter_ubo.inverse_vp * p;
    float w_inv = 1.0 / (wp.w + 1e-10);
    vec3 cam = scatter_ubo.cam_pos_pad.xyz;
    vec3 dir = vec3(wp.x*w_inv - cam.x, wp.y*w_inv - cam.y, wp.z*w_inv - cam.z);
    float len = length(dir);
    if (len > 1e-10) dir /= len;
    return dir;
}
```

- [ ] **Step 3: Add per-tile depth key computation in main()**

In the tile iteration loop, replace the depth_bits assignment with conditional eval_3D logic:

```glsl
// In the tile iteration loop, after computing tile_id:
uint depth_bits_tile = depth_bits;  // default: view-space z
if (pc.eval_3D != 0u) {
    float tile_cx = (float(tx) + 0.5) * float(pc.tile_w);
    float tile_cy = (float(ty) + 0.5) * float(pc.tile_h);
    vec3 viewdir = pixelToWorldDir(tile_cx, tile_cy);
    float ptd = depthAlongRay(i, viewdir);
    if (ptd > 0.0) depth_bits_tile = floatBitsToUint(ptd);
}
uint64_t key = (uint64_t(tile_id) << 32) | uint64_t(depth_bits_tile);
```

- [ ] **Step 4: Build and verify shader compiles**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
```

- [ ] **Step 5: Commit**

```bash
git add src/vulkan/shaders/scatter.comp
git commit -m "feat: add depthAlongRay per-tile depth key in scatter.comp

When eval_3D=true, compute per-tile depth via depthAlongRay using
cov3D_inv and mean_offset from preprocess, and inverse viewproj from
ScatterUBO. Improves sort quality for anisotropic Gaussians."
```

---

## Task 5: TileBinnerVulkan Scatter Plumbing

**Files:**
- Modify: `src/vulkan/scatter_pass.cpp`
- Modify: `include/vulkan/tile_binner_passes.h`
- Modify: `src/vulkan/tile_binner_vulkan.cpp`
- Modify: `include/vulkan/tile_binner_vulkan.h`

Wire the new scatter buffers (cov3D_inv, mean_offset, ScatterUBO) through the C++ adapter layer.

- [ ] **Step 1: Update ScatterPass for 11 bindings**

In `scatter_pass.cpp`, change binding count from 8 to 11. Set binding 10 (ScatterUBO) as UNIFORM_BUFFER:

```cpp
std::vector<VkDescriptorType> binding_types(11,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
binding_types[scatter_bind::SCATTER_UBO] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
```

Add bind_buffers entries:

```cpp
pipeline_->update_ssbo(descriptor_set_, scatter_bind::COV3D_INV,   b.cov3D_inv);
pipeline_->update_ssbo(descriptor_set_, scatter_bind::MEAN_OFFSET, b.mean_offset);
pipeline_->update_ubo(descriptor_set_,  scatter_bind::SCATTER_UBO, b.scatter_ubo, sizeof(ScatterUBO));
```

Update push constants to pass eval_3D flag (use the `_pad` → `eval_3D` field from Task 1).

- [ ] **Step 2: Update TileBinnerVulkan::prepare_record()**

Add new parameters for eval_3D buffers:

```cpp
void prepare_record(uint32_t N, ...,
                    VkBuffer cov3D_inv,       // from PreprocessorVulkan
                    VkBuffer mean_offset,     // from PreprocessorVulkan
                    bool eval_3D,
                    const Camera& cam);       // for inverse_vp computation
```

Inside prepare_record:
1. Allocate a ScatterUBO buffer
2. Compute inverse viewproj matrix
3. Upload ScatterUBO data
4. Wire cov3D_inv, mean_offset, scatter_ubo into ScatterPass::Buffers
5. If eval_3D=false, allocate dummy 4-byte buffers for cov3D_inv and mean_offset to keep descriptor set valid

- [ ] **Step 3: Update renderer_vulkan.cpp to pass new buffers**

In the full pipeline orchestrator, wire PreprocessorVulkan's new buffer getters to TileBinnerVulkan's prepare_record:

```cpp
binner.prepare_record(N, ...,
    preprocessor.cov3D_inv_buffer(),
    preprocessor.mean_offset_buffer(),
    cfg.eval_3D, cam);
```

- [ ] **Step 4: Build and test existing scatter tests still pass**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
ctest --test-dir build -R Scatter -V --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add src/vulkan/scatter_pass.cpp include/vulkan/tile_binner_passes.h \
        src/vulkan/tile_binner_vulkan.cpp include/vulkan/tile_binner_vulkan.h \
        src/vulkan/renderer_vulkan.cpp
git commit -m "feat: wire eval_3D scatter buffers through TileBinnerVulkan

Pass cov3D_inv, mean_offset, ScatterUBO through to scatter.comp.
Compute inverse viewproj in TileBinnerVulkan for per-tile depth."
```

---

## Task 6: rasterize.comp eval_3D K-Buffer

**Files:**
- Modify: `src/vulkan/shaders/rasterize.comp`
- Reference: `src/cpu/rasterizer_cpu.cpp:61-135` (CPU eval_3D k-buffer rasterizer)
- Reference: `src/math_utils.cpp:186-207` (maxContribRay)
- Reference: `src/math_utils.cpp:616-651` (depthAlongRay, pixelToWorldDir)

This is the core algorithmic change. The eval_3D rasterizer replaces 2D conic evaluation with:
1. Per-pixel 3D Gaussian evaluation via `maxContribRay` (plane intersection)
2. StopThePop k-buffer (K=16) for correct per-pixel depth ordering
3. Per-pixel depth via `depthAlongRay`

**Shared memory design for eval_3D:**
- gauss2screen rows 0,1,3 (12 floats): 12 * 256 * 4 = 12 KB
- opacity (1 float): 256 * 4 = 1 KB
- rgb (3 floats): 3 * 256 * 4 = 3 KB
- gid (1 uint): 256 * 4 = 1 KB
- depths fallback (1 float): 256 * 4 = 1 KB
- Total: ~18 KB (within Vulkan 1.3 Roadmap 2022 minimum of 32 KB)
- cov3D_inv and mean_offset loaded from global memory per-pixel (indexed by gid)

- [ ] **Step 1: Add specialization constant and new bindings**

```glsl
// After existing layout declarations:
layout(constant_id = 0) const uint spec_eval_3D = 0u;

// eval_3D input SSBOs (bindings 9..12)
layout(std430, set = 0, binding = 9)  readonly buffer Gauss2Screen   { float gauss2screen[]; };  // [N*16]
layout(std430, set = 0, binding = 10) readonly buffer Opacities2D    { float opacities_2d[]; };  // [N]
layout(std430, set = 0, binding = 11) readonly buffer Cov3DInv       { float cov3D_inv[];    };  // [N*6]
layout(std430, set = 0, binding = 12) readonly buffer MeanOffset     { float mean_offset[];  };  // [N*3]
layout(std140, set = 0, binding = 13) uniform RasterEval3DUBO {
    mat4 inverse_vp;
    vec4 cam_pos_pad;
    vec4 img_size_pad;
} eval3d_ubo;
```

- [ ] **Step 2: Add shared memory arrays for eval_3D batch**

```glsl
// Shared memory for eval_3D batches — gauss2screen rows 0,1,3 (12 floats per Gaussian)
shared float s_g2s[256 * 12];  // rows 0,1,3 of gauss2screen (skip row 2)
shared float s_opa [256];       // per-Gaussian pre-dilated opacity
shared float s_depth[256];      // fallback depth (view-space z)
// s_rgb_r/g/b and s_gid reused from 2D path
```

- [ ] **Step 3: Add maxContribRay, depthAlongRay, pixelToWorldDir functions**

Port from math_utils.cpp (same as scatter.comp but operating on shared memory indices):

```glsl
float maxContribRayPixel(float plane_a[4], float plane_b[4]) {
    // cross(plane_a.xyz, plane_b.xyz)
    float dx = plane_a[1]*plane_b[2] - plane_a[2]*plane_b[1];
    float dy = plane_a[2]*plane_b[0] - plane_a[0]*plane_b[2];
    float dz = plane_a[0]*plane_b[1] - plane_a[1]*plane_b[0];
    float mx = plane_a[3]*plane_b[0] - plane_a[0]*plane_b[3];
    float my = plane_a[3]*plane_b[1] - plane_a[1]*plane_b[3];
    float mz = plane_a[3]*plane_b[2] - plane_a[2]*plane_b[3];
    float dd = dx*dx + dy*dy + dz*dz;
    // CRITICAL: guard against dd→0 (Gaussian center near pixel ray).
    // Bug review §4: division by zero causes NaN that propagates through k-buffer.
    if (dd < 1e-8) return 0.0;  // treat as zero contribution (center on ray)
    return (mx*mx + my*my + mz*mz) / dd;
}
```

And `depthAlongRay` (reads from global cov3D_inv/mean_offset by gid), `pixelToWorldDir` (same as scatter version using eval3d_ubo).

- [ ] **Step 4: Add eval_3D k-buffer rasterizer path**

In main(), add the eval_3D branch. Follow `rasterizer_cpu.cpp:61-135` exactly:

```glsl
if (spec_eval_3D == 1u) {
    // K-buffer registers (per-pixel, not shared memory)
    const int KBUF_K = 16;
    float kbuf_depth[16];
    float kbuf_alpha[16];
    uint  kbuf_gid[16];
    int kcount = 0;
    bool done_pixel = !valid_pixel;

    // Per-pixel ray direction for depthAlongRay
    vec3 ray_dir = pixelToWorldDir(float(pixel_x) + 0.5, float(pixel_y) + 0.5);

    for (uint batch_start = range_start; batch_start < range_end; batch_start += BATCH_SIZE) {
        // 1. Cooperative load into shared memory
        uint load_idx = batch_start + local_idx;
        if (load_idx < range_end) {
            uint gid = values_sorted[load_idx];
            s_gid[local_idx] = gid;
            // Load gauss2screen rows 0, 1, 3 (skip row 2)
            for (int j = 0; j < 4; j++) {
                s_g2s[local_idx*12 + j]     = gauss2screen[gid*16u + 0u*4u + uint(j)]; // row 0
                s_g2s[local_idx*12 + 4 + j] = gauss2screen[gid*16u + 1u*4u + uint(j)]; // row 1
                s_g2s[local_idx*12 + 8 + j] = gauss2screen[gid*16u + 3u*4u + uint(j)]; // row 3
            }
            s_opa[local_idx]   = opacities_2d[gid];
            s_rgb_r[local_idx] = rgb[gid*3u];
            s_rgb_g[local_idx] = rgb[gid*3u+1u];
            s_rgb_b[local_idx] = rgb[gid*3u+2u];
            s_depth[local_idx] = /* depths buffer or view-z fallback */;
        }
        barrier();

        // 2. Per-pixel k-buffer processing
        uint batch_size = min(BATCH_SIZE, range_end - batch_start);
        if (!done_pixel) {
            for (uint k = 0u; k < batch_size; ++k) {
                // Compute plane_x, plane_y from gauss2screen rows
                float fpx = float(pixel_x) + 0.5;
                float fpy = float(pixel_y) + 0.5;
                float plane_x[4], plane_y[4];
                for (int j = 0; j < 4; j++) {
                    plane_x[j] = s_g2s[k*12+j]     - s_g2s[k*12+8+j] * fpx; // row0 - row3*px
                    plane_y[j] = s_g2s[k*12+4+j]   - s_g2s[k*12+8+j] * fpy; // row1 - row3*py
                }

                float power = -0.5 * maxContribRayPixel(plane_x, plane_y);
                if (power > 0.0) continue;

                float alpha = min(0.99, s_opa[k] * exp(power));
                if (alpha < ALPHA_THRESHOLD) continue;

                // Per-pixel depth via depthAlongRay
                uint gid = s_gid[k];
                float pix_depth = s_depth[k]; // fallback
                float ptd = depthAlongRayGlobal(gid, ray_dir);
                if (ptd > 0.0) pix_depth = ptd;

                // K-buffer insert (sorted by depth, pop-and-blend front when full)
                if (kcount >= KBUF_K) {
                    // Pop front: blend oldest (smallest depth) entry
                    float a = kbuf_alpha[0];
                    uint  blend_gid = kbuf_gid[0];
                    float test_T = T * (1.0 - a);
                    if (test_T < T_MIN) { done_pixel = true; continue; }
                    C.r += s_rgb_r_global(blend_gid) * a * T; // Need to load RGB from global for popped entry
                    C.g += s_rgb_g_global(blend_gid) * a * T;
                    C.b += s_rgb_b_global(blend_gid) * a * T;
                    T = test_T;
                    n++;
                    // Shift buffer left
                    for (int m = 0; m < KBUF_K - 1; m++) {
                        kbuf_depth[m] = kbuf_depth[m+1];
                        kbuf_alpha[m] = kbuf_alpha[m+1];
                        kbuf_gid[m]   = kbuf_gid[m+1];
                    }
                    kcount = KBUF_K - 1;
                }
                // Sorted insertion
                int pos = kcount;
                for (int m = kcount - 1; m >= 0; m--) {
                    if (kbuf_depth[m] > pix_depth) pos = m; else break;
                }
                for (int m = kcount; m > pos; m--) {
                    kbuf_depth[m] = kbuf_depth[m-1];
                    kbuf_alpha[m] = kbuf_alpha[m-1];
                    kbuf_gid[m]   = kbuf_gid[m-1];
                }
                kbuf_depth[pos] = pix_depth;
                kbuf_alpha[pos] = alpha;
                kbuf_gid[pos]   = gid;
                kcount++;
            }
        }
        barrier();
    }

    // Flush remaining k-buffer entries
    if (!done_pixel) {
        for (int k = 0; k < kcount; k++) {
            float a = kbuf_alpha[k];
            uint gid = kbuf_gid[k];
            float test_T = T * (1.0 - a);
            if (test_T < T_MIN) break;
            C.r += rgb[gid*3u]     * a * T;
            C.g += rgb[gid*3u+1u]  * a * T;
            C.b += rgb[gid*3u+2u]  * a * T;
            T = test_T;
            n++;
        }
    }
} else {
    // === Existing 2D path (unchanged) ===
    // ... lines 139-196
}
```

**CRITICAL implementation note:** When popping from the k-buffer, the RGB for the popped Gaussian must be loaded from **global memory** (not shared memory), because the popped Gaussian may be from a previous batch. Store the Gaussian ID (gid) in the k-buffer to enable this. The CPU reference does `pre.rgb[gid*3+ch]` which is a global array read.

- [ ] **Step 5: Build and verify shader compiles**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
```

- [ ] **Step 6: Commit**

```bash
git add src/vulkan/shaders/rasterize.comp
git commit -m "feat: add eval_3D k-buffer rasterizer in rasterize.comp

Implement StopThePop per-pixel sorting with K=16 k-buffer:
- maxContribRayPixel for 3D Gaussian evaluation per pixel
- depthAlongRay for per-pixel depth computation
- Sorted insertion into k-buffer, pop-and-blend front when full
- Flush remaining entries after all batches

Branched on spec_eval_3D specialization constant.
Shared memory: gauss2screen rows 0,1,3 + opacity + rgb + gid + depth."
```

---

## Task 7: RasterizerVulkan eval_3D Plumbing

**Files:**
- Modify: `src/vulkan/rasterize_pass.cpp`
- Modify: `include/vulkan/rasterize_pass.h`
- Modify: `src/vulkan/rasterizer_vulkan.cpp`
- Modify: `include/vulkan/rasterizer_vulkan.h`

Wire the new rasterize buffers and specialization constant through the C++ adapter.

- [ ] **Step 1: Update RasterizePass for 14 bindings + spec constant**

In `rasterize_pass.cpp`:
1. Increase binding count from 9 to 14
2. Set binding 8 and 13 as UNIFORM_BUFFER
3. Add specialization constant for spec_eval_3D
4. Wire new bind_buffers entries

```cpp
std::vector<VkDescriptorType> binding_types(14,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
binding_types[rasterize_bind::RASTER_UBO]         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
binding_types[rasterize_bind::RASTER_EVAL3D_UBO]  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
```

Add RasterizePass constructor parameter `uint32_t spec_eval_3D` with specialization constant wiring (same pattern as PreprocessPass).

- [ ] **Step 2: Update RasterizerVulkan constructor and rasterize()**

Accept eval_3D flag, create RasterizePass with appropriate spec constant:

```cpp
RasterizerVulkan::RasterizerVulkan(VulkanContext& ctx, bool eval_3D)
    : ctx_(ctx), eval_3D_(eval_3D) {
    pass_ = std::make_unique<RasterizePass>(ctx_, eval_3D ? 1u : 0u);
}
```

In `rasterize()`, when eval_3D:
1. Upload gauss2screen, opacities_2d, cov3D_inv, mean_offset from PreprocessOutput
2. Compute inverse_vp, build RasterEval3DUBO, upload
3. Wire all buffers to RasterizePass::Buffers
4. Skip the conic_opacity repack (3D mode uses opacities_2d directly)

- [ ] **Step 3: Update prepare_record() for eval_3D**

Accept additional buffer handles from upstream:

```cpp
void prepare_record(uint32_t W, uint32_t H,
                    uint32_t num_tiles_x, uint32_t num_tiles_y,
                    const float bg_color[3],
                    VkBuffer values_sorted, VkBuffer tile_ranges,
                    VkBuffer means2D, VkBuffer conic_opacity_packed, VkBuffer rgb,
                    // eval_3D additional inputs:
                    VkBuffer gauss2screen, VkBuffer opacities_2d,
                    VkBuffer cov3D_inv, VkBuffer mean_offset,
                    const Camera& cam);
```

- [ ] **Step 4: Build and test**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add src/vulkan/rasterize_pass.cpp include/vulkan/rasterize_pass.h \
        src/vulkan/rasterizer_vulkan.cpp include/vulkan/rasterizer_vulkan.h
git commit -m "feat: wire eval_3D rasterize buffers through RasterizerVulkan

Pass gauss2screen, opacities_2d, cov3D_inv, mean_offset, and
RasterEval3DUBO to rasterize.comp. Accept eval_3D in constructor
to set specialization constant."
```

---

## Task 8: Full Pipeline Integration + CUDA Comparison

**Files:**
- Modify: `src/vulkan/renderer_vulkan.cpp` (or wherever the full pipeline is orchestrated)
- Modify: `src/vulkan/vk_render_main.cpp`
- Test: manual CUDA golden comparison

This task wires everything together and validates against the CUDA golden reference.

- [ ] **Step 1: Update Renderer/pipeline orchestrator for eval_3D**

Ensure the full Vulkan pipeline passes eval_3D through all stages:
- PreprocessorVulkan constructed with eval_3D=true
- TileBinnerVulkan receives new buffer handles
- RasterizerVulkan constructed with eval_3D=true and receives new buffer handles

- [ ] **Step 2: Set eval_3D=true in vk_render_main.cpp**

```cpp
// Change line 59:
config.eval_3D = true;   // was: false
```

- [ ] **Step 3: Build the full pipeline**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build -j$(nproc)
```

- [ ] **Step 4: Run Vulkan render with eval_3D=true**

```bash
cd harmonyos_3dgs && ./build/gs3d_vk_render \
    ../AAA-Gaussians/output/lego/point_cloud/iteration_30000/point_cloud.ply \
    ../AAA-Gaussians/cameras.json 0
```

This produces `output_vk.ppm` and `vk_float.raw`.

- [ ] **Step 5: Generate CUDA golden with eval_3D=true**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/render_vk
/home/robota/miniconda3/envs/aaa-gs/bin/python tools/render_single.py \
    --model_path AAA-Gaussians/output/lego \
    --cam_id 0 --eval_3D --raw_sh
```

- [ ] **Step 6: Compare PSNR**

```bash
/home/robota/miniconda3/envs/aaa-gs/bin/python -c "
import numpy as np
vk = np.fromfile('harmonyos_3dgs/vk_float.raw', dtype=np.float32)
cu = np.load('cuda_eval3d_float.npy')
if cu.ndim == 3 and cu.shape[0] == 3:
    cu = cu.transpose(1,2,0)
cu = cu.flatten()
mse = np.mean((vk - cu)**2)
psnr = -10*np.log10(mse) if mse > 0 else float('inf')
print(f'PSNR: {psnr:.2f} dB  MSE: {mse:.2e}')
"
```

**Target:** PSNR > 40 dB (good match). PSNR > 60 dB (near bit-exact).

- [ ] **Step 7: If PSNR < 40 dB, debug**

Common issues:
1. gauss2screen matrix transpose error → check row-major vs column-major
2. Near-plane cull `<` vs `<=` ��� eval_3D uses strict `<`
3. maxContribRay returning wrong sign → check `plane_x`/`plane_y` construction
4. K-buffer pop-and-blend using shared RGB instead of global → must use global
5. depthAlongRay returning negative → check cov3D_inv symmetry

Debug approach: enable CPU eval_3D, compare preprocess outputs element-by-element first. If preprocess matches but rasterize doesn't, dump per-pixel debug info.

- [ ] **Step 8: Run all tests**

```bash
ctest --test-dir build --output-on-failure
```

All existing tests must still pass.

- [ ] **Step 9: Commit**

```bash
git add src/vulkan/vk_render_main.cpp src/vulkan/renderer_vulkan.cpp
git commit -m "feat: enable eval_3D=true in Vulkan forward pipeline

Switch vk_render_main to eval_3D=true to match CUDA AAA-Gaussians
golden reference. Full pipeline: preprocess (gauss2screen + 3D cull)
→ scatter (per-tile depthAlongRay) → rasterize (k-buffer K=16)."
```
