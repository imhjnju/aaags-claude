# SP-3: Vulkan Backward Pipeline

**Branch**: `sp3-backward-pipeline` (new worktree off sp2-forward-pipeline)
**Base**: sp2-forward-pipeline HEAD (after radius_f fix commit `9f58380`)
**Reference**: `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
**CPU reference**: `src/cpu/rasterizer_backward_cpu.cpp`, `src/cpu/preprocessor_backward_cpu.cpp`
**Golden data**: `tests/golden/tiny/step000001/cam0000/backward_*.npy`

---

## Context

SP-2 delivered the full Vulkan forward pipeline. The backward pipeline is needed
for training. SP-3 implements two Vulkan compute passes:

1. **Rasterize backward** — back-to-front traversal of the sorted Gaussian list;
   accumulates per-Gaussian gradients via float atomics.
2. **Preprocess backward** — per-Gaussian kernel computing gradient w.r.t. cov3D
   (cov2D Jacobian chain), then scale/rotation (computeCov3D) and SH.

SP-3 scope is **`eval_3D=false`** only, matching SP-2's constraint. The 3D eval
backward is a separate, more complex undertaking left for a later sprint.

---

## Key Design Decisions

### Float Atomics
`rasterize_backward.comp` accumulates per-pixel-per-Gaussian gradients into
per-Gaussian SSBOs using `atomicAdd(float)`. Requires `GL_EXT_shader_atomic_float`,
which is available on NVIDIA Thor (confirmed in SP-1 capability probe:
`has_shader_atomic_float = true`). Guard with `VK_EXT_shader_atomic_float`
device extension at pipeline creation time; throw if unavailable.

### ForwardCache Upload
The backward passes need intermediate values saved during the forward pass:
`T_final[H*W]`, `n_contrib[H*W]`, `cov3D[N*6]`, `p_view[N*3]`, `p_hom_w[N]`.
In Layer-1 (sync) mode these are populated by the CPU rasterizer/preprocessor.
In Layer-2 (record) mode, `RasterizerVulkan::download_cache()` retrieves them
after the forward submit. Backward adapters upload the ForwardCache arrays to
GPU SSBOs before dispatching backward.

### Gradient Zero-Initialization
All per-Gaussian gradient SSBOs (dL_dmeans2D, dL_dconic, dL_dopacity, dL_dcolors,
dL_dmeans3D, dL_dsh, dL_dscales, dL_drotations) must be zero-filled before dispatch.
The shaders use atomicAdd; non-zero initial values corrupt gradients.

### Binding Namespaces
Follow `preprocess_bind::` / `scatter_bind::` pattern. Add `rasterize_backward_bind::`
and `preprocess_backward_bind::` namespaces to a new header:
`include/vulkan/backward_bindings.h`.

### Adapter Interfaces
The existing CPU adapters (`RasterizerBackwardCPU`, `PreprocessorBackwardCPU`) are
plain classes (no abstract base). The Vulkan backward adapters will follow the same
pattern — no virtual dispatch needed, just concrete classes.

```cpp
// New classes (modeled after CPU counterparts):
class RasterizerBackwardVulkan {
public:
    void backward(const PreprocessOutput& pre, const BinningOutput& bin,
                  const Camera& cam, const RenderConfig& cfg,
                  const ForwardCache& cache, const float* dL_dpixels,
                  RasterGradOutput& rgrad, FrameAllocator& alloc);
};

class PreprocessorBackwardVulkan {
public:
    void backward(const GaussianData& g, const Camera& cam,
                  const RenderConfig& cfg, const ForwardCache& cache,
                  const RasterGradOutput& rgrad, const RawGaussianParams& raw,
                  GradientOutput& grads, FrameAllocator& alloc);
};
```

---

## rasterize_backward.comp — Shader Spec

**Workgroup**: `local_size_x=16, local_size_y=16, local_size_z=1` (same as forward)
**Dispatch**: `ceil(W/16) × ceil(H/16) × 1`

**Extension**: `#extension GL_EXT_shader_atomic_float : require`

**Bindings**:
```
binding 0  = tile_ranges       RO  uint[num_tiles*2]
binding 1  = values_sorted     RO  uint[R]
binding 2  = means2D           RO  float[N*2]
binding 3  = conic_opacity_packed RO float[N*4]   -- (a, b, c, opacity)
binding 4  = colors            RO  float[N*3]
binding 5  = T_final           RO  float[H*W]
binding 6  = n_contrib         RO  uint[H*W]
binding 7  = dL_dpixels        RO  float[H*W*3]   -- row-major RGB
binding 8  = dL_dmeans2D       RW  float[N*2]     -- atomicAdd
binding 9  = dL_dconics        RW  float[N*3]     -- atomicAdd (a,b,c)
binding 10 = dL_dopacity       RW  float[N]       -- atomicAdd
binding 11 = dL_dcolors        RW  float[N*3]     -- atomicAdd
binding 12 = RasterizeBackwardUBO (std140)
```

**RasterizeBackwardUBO** (std140, 32 bytes):
```glsl
layout(std140, binding = 12) uniform RasterizeBackwardUBO {
    uint  W;
    uint  H;
    uint  num_tiles_x;
    uint  _pad;
    vec3  bg_color;
    float _pad2;
};
```

**Algorithm** — mirrors `renderCUDA` backward (EVAL_3D=false path):

```glsl
// Block = tile (tx, ty)
uint tile_id = gl_WorkGroupID.y * num_tiles_x + gl_WorkGroupID.x;
uint range_start = tile_ranges[tile_id*2];
uint range_end   = tile_ranges[tile_id*2+1];
int  toDo        = int(range_end - range_start);

// Shared memory — same layout as forward rasterize.comp
shared int   collected_id     [BLOCK_SIZE];   // BLOCK_SIZE = 256
shared vec2  collected_xy     [BLOCK_SIZE];
shared vec4  collected_conic_o[BLOCK_SIZE];
shared float collected_colors [3 * BLOCK_SIZE];

uint pix = gl_GlobalInvocationID.y * W + gl_GlobalInvocationID.x;
bool inside = (gl_GlobalInvocationID.x < W && gl_GlobalInvocationID.y < H);
bool done = !inside;

float T_fin = inside ? T_final[pix] : 0.0;
float T = T_fin;
int   last_contrib = inside ? int(n_contrib[pix]) : 0;
vec3  dL_dpix = inside ? vec3(dL_dpixels[pix*3], dL_dpixels[pix*3+1], dL_dpixels[pix*3+2]) : vec3(0);

int contributor = toDo;
vec3 accum_rec = vec3(0);
float last_alpha = 0.0;
vec3 last_color = vec3(0);

const float ddelx_dx = 0.5 * float(W);
const float ddely_dy = 0.5 * float(H);

for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // Load BACK-to-FRONT: index = range.y - 1 - progress
    barrier();
    int progress = i * BLOCK_SIZE + int(gl_LocalInvocationIndex);
    if (range_start + uint(progress) < range_end) {
        int coll_id = int(values_sorted[range_end - 1u - uint(progress)]);
        collected_id[gl_LocalInvocationIndex] = coll_id;
        collected_xy[gl_LocalInvocationIndex] = vec2(means2D[coll_id*2], means2D[coll_id*2+1]);
        collected_conic_o[gl_LocalInvocationIndex] = vec4(
            conic_opacity_packed[coll_id*4],
            conic_opacity_packed[coll_id*4+1],
            conic_opacity_packed[coll_id*4+2],
            conic_opacity_packed[coll_id*4+3]);
        for (int c = 0; c < 3; c++)
            collected_colors[c * BLOCK_SIZE + int(gl_LocalInvocationIndex)] =
                colors[coll_id * 3 + c];
    }
    barrier();

    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        contributor--;
        if (contributor >= last_contrib) continue;

        vec4 con_o = collected_conic_o[j];
        vec2 xy    = collected_xy[j];
        vec2 d     = xy - vec2(gl_GlobalInvocationID.xy);
        float power = -0.5f*(con_o.x*d.x*d.x + con_o.z*d.y*d.y) - con_o.y*d.x*d.y;
        if (power > 0.0) continue;

        float G     = exp(power);
        float alpha = min(0.99f, con_o.w * G);
        if (alpha < 1.0/255.0) continue;

        // Recover T for this step (backwards from T_fin)
        T = T / (1.0 - alpha);
        float dchannel_dcolor = alpha * T;

        float dL_dalpha = 0.0;
        int global_id = collected_id[j];
        for (int ch = 0; ch < 3; ch++) {
            float c = collected_colors[ch * BLOCK_SIZE + j];
            accum_rec[ch] = last_alpha * last_color[ch] + (1.0 - last_alpha) * accum_rec[ch];
            last_color[ch] = c;
            dL_dalpha += (c - accum_rec[ch]) * dL_dpix[ch];
            atomicAdd(dL_dcolors[global_id * 3 + ch], dchannel_dcolor * dL_dpix[ch]);
        }
        dL_dalpha *= T;
        last_alpha = alpha;

        float bg_dot = dot(bg_color, dL_dpix);
        dL_dalpha += (-T_fin / (1.0 - alpha)) * bg_dot;

        float dL_dG = con_o.w * dL_dalpha;
        float gdx = G * d.x;
        float gdy = G * d.y;
        atomicAdd(dL_dmeans2D[global_id*2  ], dL_dG * (-gdx * con_o.x - gdy * con_o.y) * ddelx_dx);
        atomicAdd(dL_dmeans2D[global_id*2+1], dL_dG * (-gdy * con_o.z - gdx * con_o.y) * ddely_dy);
        atomicAdd(dL_dconics[global_id*3  ], -0.5f * gdx * d.x * dL_dG);
        atomicAdd(dL_dconics[global_id*3+1], -0.5f * gdx * d.y * dL_dG);
        atomicAdd(dL_dconics[global_id*3+2], -0.5f * gdy * d.y * dL_dG);
        atomicAdd(dL_dopacity[global_id],    G * dL_dalpha);
    }
}
```

---

## preprocess_backward.comp — Shader Spec

**Workgroup**: `local_size_x=256, local_size_y=1, local_size_z=1`
**Dispatch**: `ceil(N/256) × 1 × 1`

**Bindings**:
```
binding 0  = positions        RO  float[N*3]
binding 1  = radii            RO  int[N]
binding 2  = cov3D            RO  float[N*6]   -- from ForwardCache
binding 3  = d_conics         RO  float[N*3]   -- from rasterizer backward (a,b,c)
binding 4  = d_opacity        RO  float[N]     -- from rasterizer backward
binding 5  = sh_coeffs        RO  float[N*max_coeffs*3]
binding 6  = scales           RO  float[N*3]   -- exp-activated
binding 7  = rotations        RO  float[N*4]   -- normalized quaternion (r,x,y,z)
binding 8  = d_rgb            RO  float[N*3]   -- from rasterizer backward
binding 9  = d_means2D        RO  float[N*2]   -- from rasterizer backward
binding 10 = p_view           RO  float[N*3]   -- view-space pos from ForwardCache
binding 11 = p_hom_w          RO  float[N]     -- clip w from ForwardCache
binding 12 = d_means3D        RW  float[N*3]   -- output (write, not atomicAdd)
binding 13 = d_sh             RW  float[N*max_coeffs*3]  -- output
binding 14 = d_scales         RW  float[N*3]   -- output
binding 15 = d_rotations      RW  float[N*4]   -- output
binding 16 = PreprocessBackwardUBO (std140)
```

**PreprocessBackwardUBO** (std140):
```glsl
layout(std140, binding = 16) uniform PreprocessBackwardUBO {
    mat4  view_matrix;       // column-major float[16]
    uint  num_gaussians;
    uint  sh_degree;
    uint  sh_coeffs_per_g;  // (sh_degree+1)^2
    float scale_modifier;
    float h_x;   // focal_x = W / (2*tan_fovx)
    float h_y;   // focal_y = H / (2*tan_fovy)
    float tan_fovx;
    float tan_fovy;
    vec3  cam_pos;
    float _pad;
    uint  training;  // 1=training (no clamp), 0=inference
    uint  _pad2[3];
};
```

**Algorithm** — per Gaussian i (if radii[i] > 0):

Part A — `computeCov2DCUDA` equivalent:
```
t = p_view[i] (already clamped during forward)
J = jacobian(t, h_x, h_y)
W_mat = upper-left 3x3 of view_matrix
T = W_mat * J
Vrk = upper-triangle of cov3D[i] → 3x3 symmetric matrix
cov2D = transpose(T) * transpose(Vrk) * T

// c_xx/xy/yy from cov2D, then add h_var=0.3 to diagonal
denom = (c_xx + h_var)*(c_yy + h_var) - c_xy^2
denom2inv = 1 / (denom*denom + 1e-7)

// dL/d_c from dL/d_conic (inverse cov2D):
dL_dc_xx = denom2inv * (...)
dL_dc_yy = ...
dL_dc_xy = ...

// dL/dT (gradient through cov2D = T^T * Vrk^T * T)
// dL/dJ then dL/dt_x, dL/dt_y, dL/dt_z
// Transform back to world: dL_dmeans3D += view_T * dL_dt
// dL/dVrk → dL_dcov3D[i*6..i*6+5]
// dL_dcov3D → computeCov3D → dL_dscales, dL_drots
```

Part B — `preprocessCUDA` equivalent (SH backward):
```
// Reconstruct view dir from position and cam_pos
// Apply clamp mask (same forward eval to detect clamping)
// Compute SH basis gradients for requested degree
// d_sh[i*max_coeffs..] += basis * d_rgb_eff
// View-direction gradient → position contribution → accumulate to d_means3D[i]

// Also: projection Jacobian for d_means2D → d_means3D:
// m_w = 1/(proj * pos).w
// dL_dmeans3D[i] += proj_jacobian_T * d_means2D[i]  (same as preprocessCUDA)
```

---

## Tasks

### T22: RasterizeBackwardPass + RasterizerBackwardVulkan (Layer-1)
**Spec**: §6 (Rasterizer backward, eval_3D=false)
**Effort**: 6 points
**Deliverables**:
- `src/vulkan/shaders/rasterize_backward.comp`
- `include/vulkan/backward_bindings.h` (rasterize_backward_bind:: namespace)
- `include/vulkan/rasterize_backward_pass.h` + `src/vulkan/rasterize_backward_pass.cpp`
- `include/vulkan/rasterizer_backward_vulkan.h` + `src/vulkan/rasterizer_backward_vulkan.cpp`
- `tests/test_rasterize_backward_pass_vk.cpp` — tiny fixture unit test
- `tests/test_rasterizer_backward_vulkan.cpp` — golden comparison against CPU backward

**Test spec**:
```
RasterizeBackwardPass.TinyFixture:
  - N=4, W=64, H=64 (2x2 tile grid, 4 tiles)
  - Construct minimal sorted list manually (a few pairs)
  - Feed known T_final, n_contrib, dL_dpixels
  - Check dL_dcolors, dL_dopacity are nonzero and finite
  - GTEST_SKIP if no Vulkan device

RasterizerBackwardVulkan.MatchesCPU_TinyGolden:
  - Load full tiny fixture (N=103, W=64, H=64)
  - Run CPU rasterizer forward → ForwardCache
  - Load backward_dL_dout_color.npy as dL_dpixels
  - Run Vulkan backward and CPU backward
  - Assert max abs diff < 1e-4 on dL_dmeans2D, dL_dconics, dL_dopacity, dL_dcolors
  - GTEST_SKIP if no Vulkan device
```

**CMakeLists**: Add both test sources to `gs3d_vk_tests`.

---

### T23: PreprocessBackwardPass + PreprocessorBackwardVulkan (Layer-1)
**Spec**: §2 (Preprocess backward, eval_3D=false)
**Effort**: 8 points
**Deliverables**:
- `src/vulkan/shaders/preprocess_backward.comp`
- Add `preprocess_backward_bind::` namespace to `include/vulkan/backward_bindings.h`
- `include/vulkan/preprocess_backward_pass.h` + `src/vulkan/preprocess_backward_pass.cpp`
- `include/vulkan/preprocessor_backward_vulkan.h` + `src/vulkan/preprocessor_backward_vulkan.cpp`
- `tests/test_preprocess_backward_pass_vk.cpp` — tiny fixture unit test
- `tests/test_preprocessor_backward_vulkan.cpp` — golden comparison against CPU backward

**Test spec**:
```
PreprocessBackwardPass.TinyFixture:
  - N=4 Gaussians, sh_degree=0
  - Construct minimal ForwardCache (cov3D, p_view, p_hom_w all plausible)
  - Feed d_conics, d_rgb, d_means2D
  - Check d_sh, d_scales, d_rotations, d_means3D are finite and nonzero
  - GTEST_SKIP if no Vulkan device

PreprocessorBackwardVulkan.MatchesCPU_TinyGolden:
  - Load full tiny fixture (N=103, sh_degree=3)
  - Run CPU forward (Preprocessor + Rasterizer) to populate ForwardCache
  - Run CPU rasterizer backward to get RasterGradOutput
  - Run Vulkan preprocess backward and CPU preprocess backward
  - Assert max abs diff < 1e-4 on d_means3D, d_cov3D, d_sh, d_scales, d_rotations
  - GTEST_SKIP if no Vulkan device
```

**ForwardCache population**: The test populates ForwardCache from the CPU
forward run (same pattern as existing `test_preprocessor_backward.cpp`).
`PreprocessorBackwardVulkan::backward()` uploads cov3D, p_view, p_hom_w from
the ForwardCache to GPU SSBOs before dispatching.

**CMakeLists**: Add both test sources to `gs3d_vk_tests`.

---

### T24: Integration Test + Docs + Tag
**Effort**: 2 points
**Deliverables**:
- `tests/test_backward_pipeline_vk.cpp` — full end-to-end:
  - Vulkan forward (from ForwardPipeline test) → download ForwardCache
  - Vulkan rasterize backward → Vulkan preprocess backward
  - Compare final d_sh, d_scales, d_rotations, d_means3D against CUDA golden npy
  - GTEST_SKIP if no Vulkan device
- `dev_notes/sp3_backward_notes.md` — completion notes
- Git tag `sp3-backward-ready`

---

## Implementation Notes

### GLSL Float Atomics
```glsl
#extension GL_EXT_shader_atomic_float : require
// ...
layout(std430, set = 0, binding = 8) buffer DLMeans2D { float dL_dmeans2D[]; };
// Usage:
atomicAdd(dL_dmeans2D[global_id * 2], value_x);
```

### Shared Memory Tiling in rasterize_backward.comp
BLOCK_SIZE = 16 × 16 = 256 threads. Shared arrays:
```glsl
shared int   s_id     [256];
shared vec2  s_xy     [256];
shared vec4  s_conic_o[256];
shared float s_colors [768];  // 3 * 256
```
Total: ~9KB shared memory. Well within the 48KB minimum Vulkan guarantee.

### h_var (regularization filter) in preprocess_backward.comp
```cpp
constexpr float h_var = 0.3f;  // matches forward preprocess.comp and CUDA
```
Used identically to the CUDA computeCov2DCUDA kernel: add to c_xx and c_yy
AFTER reading the cov2D value.

### ForwardCache → GPU Buffers
`PreprocessorBackwardVulkan` uploads from ForwardCache:
- `cache.cov3D` [N*6]
- `cache.p_view` [N*3]
- `cache.p_hom_w` [N]
- `cache.pre->radii` [N]

Note: `RasterizerBackwardVulkan` uploads from:
- `cache.T_final` [H*W]
- `cache.n_contrib` [H*W]

### Clamping mask in preprocess_backward.comp
The SH clamp mask requires re-running the SH forward eval to determine which
channels were clamped. This is the same approach as `computeColorFromSH`
backward in CUDA — no extra storage, just recompute from the SH coefficients
and view direction.

### Tolerance: 1e-4 (relaxed from SP-2's 1e-6)
Backward gradients accumulate floating-point error from atomic operations.
Multiple atomicAdds from different threads introduce order-dependent rounding.
Use 1e-4 absolute tolerance for backward vs CPU comparison (CUDA golden is
a secondary check, not the primary correctness criterion).

---

## Constraints (from CLAUDE.md STOP Gates)
- Gate #1: Research before editing backward code. This plan IS the research.
- Gate #5: Read spec §2 (Preprocessing) and §6 (Rasterizer) before coding T23/T22.
- Gate #6: All new test files require dual-review (+1 self +2 external) after creation.
- Gate #8: Pre-commit checklist before each commit.

---

## Out of Scope
- `eval_3D=true` backward (AAA-Gaussians 3D eval — complex, next sprint)
- Layer-2 record() API for backward passes (not needed until E2E GPU training)
- Gradient checking against finite differences (useful but deferred)
