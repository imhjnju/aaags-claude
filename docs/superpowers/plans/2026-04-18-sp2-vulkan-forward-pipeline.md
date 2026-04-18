# SP-2 Vulkan Forward Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the non-spec-compliant `PreprocessorVK` (0/10 matches spec §4) and build from scratch the full SP-2 forward pipeline — 6 passes (PreprocessPass, PrefixScanPass, ScatterPass, RadixSortPass, TileRangePass, RasterizePass) + 4 adapter classes (PreprocessorVulkan, TileBinnerVulkan, SorterVulkan, RasterizerVulkan) — all validated against CUDA golden artifacts from SP-0's tiny fixture.

**Architecture:** Pass-first per spec §4. Each pass = one compute shader with fixed binding table + push constants + specialization constants. Adapter class inherits existing `Preprocessor`/`TileBinner`/`Sorter`/`Rasterizer` abstract interface, exposes sync `process/bin/sort/rasterize` (layer 1) + Vulkan-only `record(cmd, ...)` (layer 2). Inter-pass buffer schema (packed `conic_opacity[N,4]` float4, uint64 sort keys, etc.) matches CUDA golden exactly to allow bit-level validation.

**Tech Stack:** C++17, Vulkan 1.1 compute shaders (GLSL 450), xxd-embedded SPIR-V (SP-1), SP-0 `npy_reader.h` + `compare.h` + `manifest.h` for golden validation, CMake 3.28.

---

## Pre-flight checklist (verify before T1)

- [x] SP-0 tiny fixture exists: `harmonyos_3dgs/tests/golden/tiny/step000001/cam0000/` has 25 `.npy` + `manifest.json`
- [x] SP-0 C++ infrastructure: `tests/golden/npy_reader.h`, `compare.h`, `manifest.h` all pass tests
- [x] SP-1 Vulkan infrastructure: `VulkanContext::capabilities()`, byte-stream `VulkanShader`, `VulkanComputePipeline::{dispatch_sync, record}`, `insert_compute_barrier`, `gs3d_embed_spirv` CMake helper
- [x] DensityController UB fix on master (prevents flakiness during SP-2 runs)
- [x] master at `95699c8` (tagged `sp1-infrastructure-ready`), 187/187 tests pass

---

## Interface Inventory

### Existing (to be consumed by SP-2)

| Interface | Location | Usage |
|-----------|----------|-------|
| `Preprocessor` abstract class | `harmonyos_3dgs/include/preprocessor.h` | Base for `PreprocessorVulkan` |
| `TileBinner` abstract | `harmonyos_3dgs/include/tile_binner.h` | Base for `TileBinnerVulkan` |
| `Sorter` abstract | `harmonyos_3dgs/include/sorter.h` | Base for `SorterVulkan` |
| `Rasterizer` abstract | `harmonyos_3dgs/include/rasterizer.h` | Base for `RasterizerVulkan` |
| `PreprocessOutput` struct | `harmonyos_3dgs/include/types.h:69` | Public output (separate conics + opacities_2d) |
| `BinningOutput` struct | `harmonyos_3dgs/include/types.h:85` | Public sort/binning output |
| `ForwardCache` struct | `harmonyos_3dgs/include/types.h:97` | T_final + n_contrib cache |
| `GaussianData`, `Camera`, `RenderConfig` | `harmonyos_3dgs/include/types.h` | Input types |
| `VulkanContext` + capabilities | `harmonyos_3dgs/include/vulkan/vk_context.h` | SP-1 |
| `VulkanBuffer` (host-visible) | `harmonyos_3dgs/include/vulkan/vk_buffer.h` | SP-1 |
| `VulkanShader(ctx, uint8_t*, size_t)` | `harmonyos_3dgs/include/vulkan/vk_shader.h` | SP-1 byte-stream overload |
| `VulkanComputePipeline` + dispatch | `harmonyos_3dgs/include/vulkan/vk_pipeline.h` | SP-1 two-layer |
| `insert_compute_barrier(cmd)` | `harmonyos_3dgs/include/vulkan/vk_pipeline.h` | SP-1 helper |
| `load_npy` / `compare_f32`/`u32`/`u64` / `load_manifest` / `find_artifact` | `harmonyos_3dgs/tests/golden/*.h` | SP-0 |
| `gs3d_embed_spirv(name, var)` CMake function | `harmonyos_3dgs/CMakeLists.txt` | SP-1 |
| `gs3d_compile_glsl(name, var)` CMake | same | Existing |

### Draft (created in this plan)

| Interface | Planned location | Created in Task |
|-----------|------------------|-----------------|
| Binding layout header | `harmonyos_3dgs/include/vulkan/preprocess_bindings.h` | Task 2 |
| `CameraUBO` struct (std140) | `harmonyos_3dgs/include/vulkan/vk_camera_ubo.h` | Task 2 |
| `preprocess.comp` shader | `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp` | Task 4 (replaces deleted existing) |
| `PreprocessPass` class | `harmonyos_3dgs/src/vulkan/preprocess_pass.{h,cpp}` | Task 5 |
| `PreprocessorVulkan : Preprocessor` | `harmonyos_3dgs/include/vulkan/preprocessor_vulkan.h`, `src/vulkan/preprocessor_vulkan.cpp` | Task 6 |
| `prefix_sum.comp` | `harmonyos_3dgs/src/vulkan/shaders/prefix_sum.comp` | Task 8 |
| `scatter.comp` | `harmonyos_3dgs/src/vulkan/shaders/scatter.comp` | Task 9 |
| `PrefixScanPass`, `ScatterPass` classes | `harmonyos_3dgs/src/vulkan/{prefix_scan_pass,scatter_pass}.{h,cpp}` | Task 10 |
| `TileBinnerVulkan : TileBinner` | `harmonyos_3dgs/include/vulkan/tile_binner_vulkan.h`, `src/vulkan/tile_binner_vulkan.cpp` | Task 10 |
| `radix_sort_count.comp`, `radix_sort_scatter.comp` | `harmonyos_3dgs/src/vulkan/shaders/` | Task 12 |
| `tile_range.comp` | same | Task 13 |
| `RadixSortPass`, `TileRangePass`, `SorterVulkan : Sorter` | `harmonyos_3dgs/{include/vulkan,src/vulkan}/` | Task 14 |
| `rasterize.comp` | `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` | Task 16 |
| `RasterizePass`, `RasterizerVulkan : Rasterizer` | `harmonyos_3dgs/{include/vulkan,src/vulkan}/` | Task 17 |
| Integration chain test | `harmonyos_3dgs/tests/test_forward_pipeline_vk.cpp` | Task 19 |

### Deleted (in Task 1)

- `harmonyos_3dgs/include/vulkan/preprocessor_vk.h`
- `harmonyos_3dgs/src/vulkan/preprocessor_vk.cpp`
- `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp` (old version; new one written in Task 4)
- `harmonyos_3dgs/tests/test_preprocessor_vk.cpp` (will be replaced by `test_preprocess_pass_vk.cpp` in Task 7)

---

## File Structure

```
harmonyos_3dgs/
├── include/vulkan/
│   ├── preprocess_bindings.h          [T2] per-pass binding indices + push constants + spec constants
│   ├── vk_camera_ubo.h                 [T2] CameraUBO std140 struct + static_assert
│   ├── preprocessor_vulkan.h           [T6] PreprocessorVulkan : Preprocessor
│   ├── tile_binner_vulkan.h            [T10] TileBinnerVulkan : TileBinner
│   ├── sorter_vulkan.h                 [T14] SorterVulkan : Sorter
│   ├── rasterizer_vulkan.h             [T17] RasterizerVulkan : Rasterizer
│   └── preprocess_pass.h, tile_binner_passes.h, sort_passes.h, rasterize_pass.h  [T5/T10/T14/T17]
│
├── src/vulkan/
│   ├── preprocess_pass.cpp             [T5]
│   ├── preprocessor_vulkan.cpp         [T6]
│   ├── prefix_scan_pass.cpp            [T10]
│   ├── scatter_pass.cpp                [T10]
│   ├── tile_binner_vulkan.cpp          [T10]
│   ├── radix_sort_pass.cpp             [T14]
│   ├── tile_range_pass.cpp             [T14]
│   ├── sorter_vulkan.cpp               [T14]
│   ├── rasterize_pass.cpp              [T17]
│   ├── rasterizer_vulkan.cpp           [T17]
│   └── shaders/
│       ├── preprocess.comp             [T4] (replaces deleted old)
│       ├── prefix_sum.comp             [T8]
│       ├── scatter.comp                [T9]
│       ├── radix_sort_count.comp       [T12]
│       ├── radix_sort_scatter.comp     [T12]
│       ├── tile_range.comp             [T13]
│       └── rasterize.comp              [T16]
│
├── tests/
│   ├── test_preprocess_pass_vk.cpp     [T7] replaces deleted test_preprocessor_vk.cpp
│   ├── test_prefix_scan_pass_vk.cpp    [T10]
│   ├── test_scatter_pass_vk.cpp        [T10]
│   ├── test_radix_sort_pass_vk.cpp     [T14]
│   ├── test_tile_range_pass_vk.cpp     [T14]
│   ├── test_rasterize_pass_vk.cpp      [T18]
│   └── test_forward_pipeline_vk.cpp    [T19]
│
├── CMakeLists.txt                      [updated in multiple tasks]
└── dev_notes/sp2_forward_notes.md       [T21]
```

---

## Task 1: Create SP-2 worktree + delete old PreprocessorVK + scaffold

**Files:**
- Create worktree at `.claude/worktrees/sp2/` on new branch `sp2-forward-pipeline` from master `95699c8`
- Delete: `harmonyos_3dgs/include/vulkan/preprocessor_vk.h`, `src/vulkan/preprocessor_vk.cpp`, `src/vulkan/shaders/preprocess.comp`, `tests/test_preprocessor_vk.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt` (drop deleted sources)

- [ ] **Step 1: Create worktree**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git worktree add -b sp2-forward-pipeline .claude/worktrees/sp2 master
cd .claude/worktrees/sp2
git log --oneline -3  # confirm head is 95699c8 (Merge SP-1)
```

- [ ] **Step 2: Delete files**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git rm harmonyos_3dgs/include/vulkan/preprocessor_vk.h
git rm harmonyos_3dgs/src/vulkan/preprocessor_vk.cpp
git rm harmonyos_3dgs/src/vulkan/shaders/preprocess.comp
git rm harmonyos_3dgs/tests/test_preprocessor_vk.cpp
```

- [ ] **Step 3: Update CMakeLists.txt**

Edit `harmonyos_3dgs/CMakeLists.txt`:
- Remove `src/vulkan/preprocessor_vk.cpp` from `gs3d_vk_core` sources
- Remove `gs3d_compile_glsl(preprocess _spv_preprocess)` line and any dependencies on `${_spv_preprocess}`
- Remove `tests/test_preprocessor_vk.cpp` from `gs3d_vk_tests` sources
- Update `add_custom_target(gs3d_vulkan_shaders DEPENDS ...)` to remove `${_spv_preprocess}`

After editing, the Vulkan section should only compile `add_one.comp` and `hello.comp`. `gs3d_vk_tests` should have `test_vk_compute.cpp`, `test_vk_capabilities.cpp`, `test_vk_device_selection.cpp`, `test_vk_pipeline_dispatch.cpp`, `test_vk_hello.cpp`.

- [ ] **Step 4: Build and verify**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -5
cmake --build build 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -5
```

Expected: build succeeds, tests drop from 187 to 178 (lost 9 PreprocessorVK Slice tests), all 178 pass.

- [ ] **Step 5: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add -A
git -c commit.gpgsign=false commit -m "sp2: remove non-spec-compliant PreprocessorVK (0/10 spec §4 match)

Old PreprocessorVK (vk_core + shader + test) had 9 passing tests against
CPU oracle, but did not match spec §4 requirements (binding layout,
camera UBO, packed conic_opacity, specialization constants, hard-error
guards, record() API). Deleting in favor of clean rewrite per spec §4.

Tests drop from 187 to 178; rebuilt from scratch in T4-T7."
```

---

## Task 2: Binding layout header + CameraUBO struct

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/preprocess_bindings.h`
- Create: `harmonyos_3dgs/include/vulkan/vk_camera_ubo.h`

**Purpose:** Single source of truth for binding indices, push constant structs, specialization constant IDs across C++ pipeline construction and GLSL shaders. CameraUBO struct matches std140 with static_assert-verified layout.

- [ ] **Step 1: Create `preprocess_bindings.h`**

```cpp
// Single source of truth for SP-2 forward pipeline binding indices,
// push constants, and specialization constants. Mirrored exactly in
// each shader's `layout(binding=...)` declarations.
#pragma once
#include <cstdint>

// --- preprocess.comp bindings (spec §4.8.1) ---
namespace preprocess_bind {
constexpr uint32_t POSITIONS           = 0;
constexpr uint32_t SCALES              = 1;
constexpr uint32_t ROTATIONS           = 2;
constexpr uint32_t OPACITIES           = 3;
constexpr uint32_t SH                  = 4;
constexpr uint32_t FILTER_3D           = 5;
constexpr uint32_t MEANS2D             = 6;
constexpr uint32_t DEPTHS              = 7;
constexpr uint32_t CONIC_OPACITY_PACKED = 8;
constexpr uint32_t RGB                 = 9;
constexpr uint32_t RADII               = 10;
constexpr uint32_t TILES_TOUCHED       = 11;
constexpr uint32_t CAMERA_UBO          = 12;
}  // namespace preprocess_bind

// Push constants (24 bytes, spec §4.8.1)
struct PreprocessPushConstants {
    uint32_t num_gaussians;
    uint32_t sh_degree;
    uint32_t sh_coeffs_per_g;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    float    scale_modifier;
};
static_assert(sizeof(PreprocessPushConstants) == 24,
              "PreprocessPushConstants must be exactly 24 bytes per spec §4.8.1");

// Specialization constant IDs (spec §4.5)
namespace preprocess_spec {
constexpr uint32_t TRAINING = 0;  // spec_training: 1=training (no clamp), 0=inference (clamp)
constexpr uint32_t EVAL_3D  = 1;  // spec_eval_3D: always 0 in SP-2
}  // namespace preprocess_spec

// --- prefix_sum.comp bindings (spec §4.8.2) ---
namespace prefix_sum_bind {
constexpr uint32_t INPUT_ARRAY     = 0;
constexpr uint32_t OUTPUT_ARRAY    = 1;
constexpr uint32_t WORKGROUP_SUMS  = 2;
}
struct PrefixSumPushConstants {
    uint32_t num_elements;
    uint32_t phase;       // 0=local scan, 1=workgroup-sum scan, 2=add-back
    uint32_t stride;
    uint32_t _pad;
};
static_assert(sizeof(PrefixSumPushConstants) == 16, "");

// --- scatter.comp bindings (spec §4.8.3) ---
namespace scatter_bind {
constexpr uint32_t MEANS2D         = 0;
constexpr uint32_t DEPTHS          = 1;
constexpr uint32_t RADII           = 2;
constexpr uint32_t POINT_OFFSETS   = 3;
constexpr uint32_t TILES_TOUCHED   = 4;
constexpr uint32_t KEYS_UNSORTED   = 5;
constexpr uint32_t VALUES_UNSORTED = 6;
}
struct ScatterPushConstants {
    uint32_t num_gaussians;
    uint32_t num_tiles_x;
    uint32_t tile_w;
    uint32_t tile_h;
};
static_assert(sizeof(ScatterPushConstants) == 16, "");

// --- radix_sort_count.comp bindings (spec §4.8.4) ---
namespace radix_count_bind {
constexpr uint32_t KEYS_IN   = 0;
constexpr uint32_t HISTOGRAMS = 1;
}
// --- radix_sort_scatter.comp bindings (spec §4.8.5) ---
namespace radix_scatter_bind {
constexpr uint32_t KEYS_IN        = 0;
constexpr uint32_t VALUES_IN      = 1;
constexpr uint32_t BUCKET_OFFSETS = 2;
constexpr uint32_t KEYS_OUT       = 3;
constexpr uint32_t VALUES_OUT     = 4;
}
struct RadixSortPushConstants {
    uint32_t num_elements;
    uint32_t current_bit;   // 0, 4, 8, ..., 60
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(RadixSortPushConstants) == 16, "");

// --- tile_range.comp bindings (spec §4.8.6) ---
namespace tile_range_bind {
constexpr uint32_t KEYS_SORTED = 0;
constexpr uint32_t TILE_RANGES = 1;
}
struct TileRangePushConstants {
    uint32_t num_elements;
    uint32_t num_tiles;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(TileRangePushConstants) == 16, "");

// --- rasterize.comp bindings (spec §4.8.7) ---
namespace rasterize_bind {
constexpr uint32_t VALUES_SORTED      = 0;
constexpr uint32_t TILE_RANGES        = 1;
constexpr uint32_t MEANS2D            = 2;
constexpr uint32_t CONIC_OPACITY_PACKED = 3;
constexpr uint32_t RGB                = 4;
constexpr uint32_t OUT_IMAGE          = 5;
constexpr uint32_t TRANSMITTANCE      = 6;
constexpr uint32_t N_CONTRIB          = 7;
constexpr uint32_t RASTER_UBO         = 8;
}
struct RasterizePushConstants {
    uint32_t num_gaussians;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
};
static_assert(sizeof(RasterizePushConstants) == 20, "");
```

- [ ] **Step 2: Create `vk_camera_ubo.h`**

```cpp
// Host-side CameraUBO struct matching shader's std140 layout exactly.
// Spec §4.6: 224 bytes, std140, vec3 padded to vec4.
#pragma once
#include <cstdint>
#include <cstddef>

struct alignas(16) CameraUBO {
    float viewmatrix[16];         // 64B, column-major
    float projmatrix[16];         // 64B
    float inv_viewprojmatrix[16]; // 64B
    float campos_pad[4];          // 16B: xyz=campos, w=0
    float fov_size[4];            // 16B: x=tan_fovx, y=tan_fovy, z=width, w=height
};
static_assert(sizeof(CameraUBO) == 224, "CameraUBO std140 layout mismatch (spec §4.6)");
static_assert(offsetof(CameraUBO, projmatrix)         == 64,  "");
static_assert(offsetof(CameraUBO, inv_viewprojmatrix) == 128, "");
static_assert(offsetof(CameraUBO, campos_pad)         == 192, "");
static_assert(offsetof(CameraUBO, fov_size)           == 208, "");
```

- [ ] **Step 3: Syntax-check compile**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cat > /tmp/sp2_t2_check.cpp <<'EOF'
#include "vulkan/preprocess_bindings.h"
#include "vulkan/vk_camera_ubo.h"
int main() {
    PreprocessPushConstants p{};
    CameraUBO c{};
    (void)p; (void)c;
    return 0;
}
EOF
g++ -c -std=c++17 -I include -o /tmp/sp2_t2.o /tmp/sp2_t2_check.cpp && rm -f /tmp/sp2_t2.o /tmp/sp2_t2_check.cpp
echo "compile OK"
```

Expected: `compile OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/include/vulkan/preprocess_bindings.h \
        harmonyos_3dgs/include/vulkan/vk_camera_ubo.h
git -c commit.gpgsign=false commit -m "sp2: binding layout header + CameraUBO std140 struct"
```

---

## Task 3: PreprocessorVulkan adapter stub (TDD red)

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/preprocessor_vulkan.h`

**Purpose:** Declare the class structure without implementation. Lets downstream tests compile while implementation catches up in T4-T6.

> **Note (2026-04-18):** Plan originally specified `void process(..., PreprocessOutput& out)`. Corrected to `PreprocessOutput process(...)` to match `Preprocessor` base class and all existing implementations. T6 code block updated accordingly.

- [ ] **Step 1: Write header**

```cpp
// PreprocessorVulkan: adapter class implementing abstract `Preprocessor`
// interface over the SP-2 PreprocessPass (compute shader). Spec §4.1.
//
// Layer 1 (public): `process()` override — sync call for tests + non-chained use.
// Layer 2 (Vulkan-only): `record(cmd, ...)` — record into external command buffer.
//
// SP-2 hard-error constraints per spec §4.4:
//   - cfg.eval_3D must be false
//   - cfg.tile_w == 16 && cfg.tile_h == 16
//   - cfg.antialiasing must be false
#pragma once
#include "preprocessor.h"
#include "vulkan/vk_context.h"
#include <vulkan/vulkan.h>
#include <memory>

class PreprocessPass;  // fwd decl, defined in preprocess_pass.h

class PreprocessorVulkan : public Preprocessor {
public:
    explicit PreprocessorVulkan(VulkanContext& ctx);
    ~PreprocessorVulkan() override;

    // Layer 1: sync API matching Preprocessor contract.
    PreprocessOutput process(const GaussianData& g, const Camera& cam,
                             const RenderConfig& cfg, FrameAllocator& alloc,
                             ForwardCache* cache = nullptr) override;

    // Layer 2: Vulkan-only — record into external command buffer.
    // Buffers must be pre-bound via bind_buffers().
    void record(VkCommandBuffer cmd, uint32_t num_gaussians,
                uint32_t sh_degree, uint32_t sh_coeffs_per_g,
                uint32_t num_tiles_x, uint32_t num_tiles_y,
                float scale_modifier);

    // Getters for chained-pipeline handoff (used by TileBinnerVulkan etc).
    VkBuffer means2D_buffer()            const;
    VkBuffer depths_buffer()             const;
    VkBuffer conic_opacity_packed_buffer() const;
    VkBuffer rgb_buffer()                const;
    VkBuffer radii_buffer()              const;
    VkBuffer tiles_touched_buffer()      const;

private:
    VulkanContext& ctx_;
    std::unique_ptr<PreprocessPass> pass_;
    // ... private buffer handles filled in T5-T6
};
```

- [ ] **Step 2: Syntax check**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cat > /tmp/sp2_t3_check.cpp <<'EOF'
#include "vulkan/preprocessor_vulkan.h"
// forward declaration usage test: no body needed
EOF
g++ -c -std=c++17 -I include -o /tmp/sp2_t3.o /tmp/sp2_t3_check.cpp && rm -f /tmp/sp2_t3.o /tmp/sp2_t3_check.cpp
echo "compile OK"
```

Expected: `compile OK`. Note: no .cpp yet, no link error expected because no one instantiates it.

- [ ] **Step 3: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/include/vulkan/preprocessor_vulkan.h
git -c commit.gpgsign=false commit -m "sp2: PreprocessorVulkan adapter class header stub"
```

---

## Task 4: `preprocess.comp` shader per spec §4

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp`
- Modify: `harmonyos_3dgs/CMakeLists.txt` (add compile + embed)

**Spec references:**
- §4.8.1: 13 bindings, push constants (24B), specialization constants
- §4.5: `spec_training=1` → no SH color clamp (training mode)
- §4.4: reject eval_3D, non-16×16 tiles, antialiasing (via push constant + spec constant signaling; not runtime check inside shader)

**Algorithm (per-Gaussian, one thread per Gaussian):**
1. Bounds check: `if (gl_GlobalInvocationID.x >= pc.num_gaussians) return;`
2. Read raw params (positions/scales/rotations/opacities/sh/filter_3D) for this gaussian
3. World → view: `p_view = viewmatrix * vec4(position, 1.0)` → near-plane cull if `p_view.z < 0.2` → write `radii[i]=0, tiles_touched[i]=0` and return
4. View → clip → NDC → pixel: `pix = (ndc.xy * 0.5 + 0.5) * image_size`
5. Compute cov3D from scale+rotation: `L = R·diag(exp(scale))`, `Cov3D = L·Lᵀ` (6 unique upper-triangle values)
6. Compute cov2D via projection Jacobian: `J = [[focal_x/z, 0, -focal_x·x/z²], [0, focal_y/z, -focal_y·y/z²]]`; `T = J·W` (where W is upper-left 3x3 of view), `Cov2D = T·Cov3D·Tᵀ`
7. Dilate cov2D: `cov2D[0][0] += 0.3`, `cov2D[1][1] += 0.3` (AAA low-pass filter; filter_3D further scaling deferred to 2D Phase per spec §4.4 which fixes eval_3D=false, so just use the 0.3 dilation for now)
8. Compute determinant; if det ≤ 0 mark culled (`radii[i]=0`, return)
9. Compute conic = inverse of cov2D: `{a, b, c} = {cov22/det, -cov12/det, cov11/det}`
10. Compute opacity_activated = sigmoid(raw_opacity)
11. Pack: `conic_opacity_packed[i*4..i*4+3] = {a, b, c, opacity_activated}`
12. Compute radius from eigenvalues: `lambda1,2 = trace/2 ± sqrt((trace/2)² - det)`, radius_pix = ceil(3 × sqrt(max(lambda1, lambda2)))
13. Compute AABB in tile space; `tiles_touched[i] = (max_tile_x - min_tile_x + 1) * (max_tile_y - min_tile_y + 1)`
14. Write `means2D[i]`, `depths[i] = p_view.z`, `radii[i]`, `tiles_touched[i]`
15. Evaluate SH → RGB: `rgb[i*3..i*3+2] = computeColorFromSH(sh, active_degree, view_dir)`. If `spec_training=1`, do NOT clamp rgb to [0,1]; if 0, clamp.

Reference the CUDA `preprocessCUDA` in `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` for exact numerical ordering.

- [ ] **Step 1: Write shader**

Create `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp`. Target ~200 lines. Use:
- `#version 450`
- `layout(local_size_x = 256) in;`
- Bindings 0-12 matching `preprocess_bindings.h`
- `layout(constant_id=0) const uint spec_training = 1;`
- `layout(constant_id=1) const uint spec_eval_3D = 0;`
- `layout(push_constant) uniform PC { ... }` matching `PreprocessPushConstants`
- `layout(std140, binding=12) uniform CameraUBO { mat4 viewmatrix; mat4 projmatrix; mat4 inv_viewprojmatrix; vec4 campos_pad; vec4 fov_size; };`
- Helper functions inline (no external shader includes): `build_cov3D_from_scale_rot`, `project_to_2D`, `inverse_2x2`, `sh_to_rgb_deg3`
- SH evaluation matches `computeColorFromSH` in `AAA-Gaussians/.../auxiliary.h`; copy the constants (SH_C0, SH_C1, SH_C2[], SH_C3[]) from `harmonyos_3dgs/src/sh_eval.cpp` which has the CPU reference

Watch for pitfalls from `PORTING_PITFALLS.md`:
- Matrix column-major (GLM style): `mat4 * vec4` gives column-vector transform
- Quaternion convention (r, x, y, z) — match `harmonyos_3dgs/src/math_utils.cpp::build_rotation`
- Pixel center offset: use `(ndc.xy * 0.5 + 0.5) * vec2(width, height)` — spec §4 SP-2 is 2D mode so no -0.5 adjustment
- Early near-plane test at `p_view.z < 0.2`

- [ ] **Step 2: Wire into CMakeLists.txt**

In `harmonyos_3dgs/CMakeLists.txt`, add after existing `gs3d_compile_glsl(hello _spv_hello)`:
```cmake
gs3d_compile_glsl(preprocess _spv_preprocess)
gs3d_embed_spirv(preprocess _hdr_preprocess)
```
Update `add_custom_target(gs3d_vulkan_shaders DEPENDS ...)` to include `${_spv_preprocess} ${_hdr_preprocess}`.

- [ ] **Step 3: Build shader only**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cmake --build build --target gs3d_vulkan_shaders 2>&1 | tail -5
ls build/shaders/preprocess.spv build/shaders/preprocess_spv.h
```

Expected: both files exist; if glslangValidator reports errors, fix shader syntax.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/src/vulkan/shaders/preprocess.comp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp2: preprocess.comp per spec §4.8.1 (13 bindings + packed conic + training SH)"
```

---

## Task 5: `PreprocessPass` class (wraps pipeline + descriptor set + dispatch)

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/preprocess_pass.h`
- Create: `harmonyos_3dgs/src/vulkan/preprocess_pass.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

**Purpose:** `PreprocessPass` owns the `VulkanShader` + `VulkanComputePipeline` for preprocess.comp. Takes buffer handles externally (caller manages lifetime). Provides `dispatch_sync(num_gaussians, push_constants, camera_ubo_buffer)` and `record(cmd, ...)`.

- [ ] **Step 1: Write header**

Create `harmonyos_3dgs/include/vulkan/preprocess_pass.h`:
```cpp
#pragma once
#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/preprocess_bindings.h"
#include <vulkan/vulkan.h>
#include <memory>

class PreprocessPass {
public:
    // Buffer handles struct; caller owns lifetime.
    struct Buffers {
        VkBuffer positions, scales, rotations, opacities, sh, filter_3D;  // inputs
        VkBuffer means2D, depths, conic_opacity_packed, rgb;              // outputs
        VkBuffer radii, tiles_touched;                                    // outputs
        VkBuffer camera_ubo;                                              // UBO
    };

    // `spec_training`: 1 = training (no SH clamp), 0 = inference.
    // `spec_eval_3D`:  always 0 in SP-2.
    PreprocessPass(VulkanContext& ctx, uint32_t spec_training, uint32_t spec_eval_3D);
    ~PreprocessPass();

    // Bind buffers and update descriptor set.
    void bind_buffers(const Buffers& b);

    // Layer 1: sync dispatch.
    void dispatch_sync(const PreprocessPushConstants& pc);

    // Layer 2: record into external command buffer.
    void record(VkCommandBuffer cmd, const PreprocessPushConstants& pc);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    uint32_t num_elements_for_dispatch_ = 0;
};
```

- [ ] **Step 2: Write implementation**

Create `harmonyos_3dgs/src/vulkan/preprocess_pass.cpp`:
- Include `preprocess_spv.h` (xxd-generated header from T4)
- Constructor: build VulkanShader from `preprocess_spv[]`/`preprocess_spv_len` using byte-stream overload; set up specialization constants; create VulkanComputePipeline with 12 SSBOs + 1 UBO = 13 bindings, `push_constant_bytes = sizeof(PreprocessPushConstants)`
- `bind_buffers`: allocate descriptor set and vkUpdateDescriptorSets on all 13 bindings
- `dispatch_sync`: compute groups = `(num_gaussians + 255) / 256`, call `pipeline_->dispatch_sync(descriptor_set_, groups, 1, 1, &pc, sizeof(pc))`
- `record`: same but `pipeline_->record(cmd, descriptor_set_, ...)`

Key detail: VulkanComputePipeline constructor currently takes `num_ssbo_bindings` as the count. Binding 12 is a UBO, not SSBO. Either extend `VulkanComputePipeline` to support mixed bindings, OR manually set up `VkDescriptorSetLayout` inside `PreprocessPass` (easier — do this to avoid touching SP-1 code). Use `vkCreateDescriptorSetLayout`, `vkCreatePipelineLayout`, `vkCreateComputePipelines` directly, bypassing `VulkanComputePipeline` for this 13-binding case.

Alternative: Extend `VulkanComputePipeline` with a new constructor accepting `std::vector<VkDescriptorType>` listing each binding's type. Prefer this — cleaner for downstream passes. Write that extension as part of this task.

- [ ] **Step 3: Extend VulkanComputePipeline for mixed binding types**

In `include/vulkan/vk_pipeline.h`, add new constructor alongside existing:
```cpp
/// Constructor supporting mixed storage + uniform buffer bindings.
/// binding_types.size() = total bindings at set=0, in binding-index order.
/// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER supported.
VulkanComputePipeline(VulkanContext& ctx,
                      const VulkanShader& shader,
                      const std::vector<VkDescriptorType>& binding_types,
                      uint32_t push_constant_bytes = 0,
                      uint32_t max_descriptor_sets = 4,
                      const VkSpecializationInfo* spec_info = nullptr);
```

In `src/vulkan/vk_pipeline.cpp`, implement the new constructor. Keep existing constructor as thin wrapper:
```cpp
VulkanComputePipeline::VulkanComputePipeline(VulkanContext& ctx,
                                              const VulkanShader& shader,
                                              uint32_t num_ssbo_bindings,
                                              uint32_t push_constant_bytes,
                                              uint32_t max_descriptor_sets)
  : VulkanComputePipeline(ctx, shader,
        std::vector<VkDescriptorType>(num_ssbo_bindings, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
        push_constant_bytes, max_descriptor_sets, nullptr) {}
```

Main constructor: iterate binding_types, for each create a `VkDescriptorSetLayoutBinding` with the correct type. Allocate descriptor pool with `VkDescriptorPoolSize` entries aggregated per type. Handle spec_info pass-through.

Also extend `allocateDescriptorSet` to accept mixed `std::vector<VkDescriptorBufferInfo>` so UBO binding can be passed. Actually simpler: split into two helpers — one for SSBO (existing), one for UBO, called in sequence by `PreprocessPass::bind_buffers`.

Simplest: make `allocateDescriptorSet(const std::vector<VkBuffer>&)` still work (assume SSBO), and add:
```cpp
/// Update the most recently allocated descriptor set's binding `binding` to
/// reference `buffer` as a uniform buffer.
void update_ubo(VkDescriptorSet ds, uint32_t binding, VkBuffer buffer, VkDeviceSize range);
```

- [ ] **Step 4: Register PreprocessPass in CMake**

In `harmonyos_3dgs/CMakeLists.txt`, add to `gs3d_vk_core` sources:
```cmake
src/vulkan/preprocess_pass.cpp
```

- [ ] **Step 5: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
```

Expected: clean build, no tests yet (T7 adds tests).

- [ ] **Step 6: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/include/vulkan/preprocess_pass.h \
        harmonyos_3dgs/src/vulkan/preprocess_pass.cpp \
        harmonyos_3dgs/include/vulkan/vk_pipeline.h \
        harmonyos_3dgs/src/vulkan/vk_pipeline.cpp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp2: PreprocessPass class + VulkanComputePipeline mixed-binding ctor"
```

---

## Task 6: `PreprocessorVulkan` implementation (adapter over PreprocessPass)

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

- [ ] **Step 1: Implement `process()`**

Create `harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp`:
```cpp
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/preprocess_pass.h"
#include "vulkan/vk_camera_ubo.h"
#include "vulkan/vk_buffer.h"
#include <stdexcept>
#include <cstring>
#include <cmath>

PreprocessorVulkan::PreprocessorVulkan(VulkanContext& ctx) : ctx_(ctx) {
    pass_ = std::make_unique<PreprocessPass>(ctx, /*spec_training=*/1, /*spec_eval_3D=*/0);
}
PreprocessorVulkan::~PreprocessorVulkan() = default;

PreprocessOutput PreprocessorVulkan::process(const GaussianData& g, const Camera& cam,
                                              const RenderConfig& cfg, FrameAllocator& alloc,
                                              ForwardCache* cache) {
    // SP-2 hard errors per spec §4.4
    if (cfg.eval_3D) throw std::runtime_error("PreprocessorVulkan: eval_3D=true not supported in SP-2");
    if (cfg.tile_w != 16 || cfg.tile_h != 16) throw std::runtime_error("PreprocessorVulkan: only 16x16 tiles supported");
    if (cfg.antialiasing) throw std::runtime_error("PreprocessorVulkan: antialiasing flag not supported");

    PreprocessOutput out{};
    const int N = g.count;
    const int M = g.max_coeffs;

    // Allocate device-local buffers: inputs (6) + outputs (6) + CameraUBO (1)
    // For SP-2 Phase 1 we use host-visible (VulkanBuffer) for all buffers.
    // Next phase may upgrade to device-local + staging.
    auto make = [&](size_t bytes, VkBufferUsageFlags usage) {
        return std::make_shared<VulkanBuffer>(ctx_, bytes, usage);
    };
    auto pos_buf   = make(N*3*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto scl_buf   = make(N*3*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rot_buf   = make(N*4*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto op_buf    = make(N*sizeof(float),    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto sh_buf    = make(N*M*3*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto f3_buf    = make(N*sizeof(float),    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto m2d_buf   = make(N*2*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dep_buf   = make(N*sizeof(float),    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cop_buf   = make(N*4*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rgb_buf   = make(N*3*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rad_buf   = make(N*sizeof(int32_t),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tt_buf    = make(N*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cam_buf   = make(sizeof(CameraUBO),  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Upload inputs
    pos_buf->upload(g.positions, N*3*sizeof(float));
    scl_buf->upload(g.scales,    N*3*sizeof(float));
    rot_buf->upload(g.rotations, N*4*sizeof(float));
    op_buf ->upload(g.opacities, N*sizeof(float));
    sh_buf ->upload(g.sh_coeffs, N*M*3*sizeof(float));
    if (g.filter_3D) f3_buf->upload(g.filter_3D, N*sizeof(float));
    else { std::vector<float> zeros(N, 0.01f); f3_buf->upload(zeros.data(), N*sizeof(float)); }

    // Build CameraUBO
    CameraUBO c{};
    std::memcpy(c.viewmatrix, cam.view_matrix, 64);
    std::memcpy(c.projmatrix, cam.viewproj_matrix, 64);
    // inv_viewprojmatrix: invert 4x4 viewproj. Use math_utils helper or glm-free inversion.
    // (Fill in using existing invert_4x4 helper from harmonyos_3dgs/src/math_utils.cpp,
    //  or add one if missing.)
    invert_mat4(cam.viewproj_matrix, c.inv_viewprojmatrix);
    c.campos_pad[0] = cam.cam_pos[0]; c.campos_pad[1] = cam.cam_pos[1]; c.campos_pad[2] = cam.cam_pos[2]; c.campos_pad[3] = 0;
    c.fov_size[0]  = cam.tan_fovx; c.fov_size[1] = cam.tan_fovy;
    c.fov_size[2]  = (float)cam.width; c.fov_size[3] = (float)cam.height;
    cam_buf->upload(&c, sizeof(CameraUBO));

    // Bind + dispatch
    PreprocessPass::Buffers b{pos_buf->handle(), scl_buf->handle(), rot_buf->handle(),
                               op_buf->handle(), sh_buf->handle(), f3_buf->handle(),
                               m2d_buf->handle(), dep_buf->handle(), cop_buf->handle(),
                               rgb_buf->handle(), rad_buf->handle(), tt_buf->handle(),
                               cam_buf->handle()};
    pass_->bind_buffers(b);

    PreprocessPushConstants pc{
        .num_gaussians = (uint32_t)N,
        .sh_degree = (uint32_t)cfg.sh_degree,
        .sh_coeffs_per_g = (uint32_t)M,
        .num_tiles_x = (uint32_t)((cam.width + 15) / 16),
        .num_tiles_y = (uint32_t)((cam.height + 15) / 16),
        .scale_modifier = cfg.scale_modifier};
    pass_->dispatch_sync(pc);

    // Download outputs and deinterleave conic_opacity → conics + opacities_2d
    out.means2D       = alloc.allocate_array<float>(N*2);  m2d_buf->download(out.means2D, N*2*sizeof(float));
    out.depths        = alloc.allocate_array<float>(N);    dep_buf->download(out.depths, N*sizeof(float));
    out.rgb           = alloc.allocate_array<float>(N*3);  rgb_buf->download(out.rgb, N*3*sizeof(float));
    out.radii         = alloc.allocate_array<int>(N);      rad_buf->download(out.radii, N*sizeof(int32_t));
    out.tiles_touched = alloc.allocate_array<int>(N);      tt_buf->download(out.tiles_touched, N*sizeof(uint32_t));

    // Deinterleave packed conic_opacity[N,4] → conics[N,3] + opacities_2d[N]
    std::vector<float> packed(N*4);
    cop_buf->download(packed.data(), N*4*sizeof(float));
    out.conics       = alloc.allocate_array<float>(N*3);
    out.opacities_2d = alloc.allocate_array<float>(N);
    for (int i = 0; i < N; i++) {
        out.conics[i*3+0]   = packed[i*4+0];
        out.conics[i*3+1]   = packed[i*4+1];
        out.conics[i*3+2]   = packed[i*4+2];
        out.opacities_2d[i] = packed[i*4+3];
    }
    out.eval_3D = false;
    // gauss2screen, cov3D_inv, mean_offset all nullptr (not eval_3D)
    return out;
}

// ... record() and buffer getters similar, deferred to later if chained integration needs them.
// For T6, record() throws "not implemented yet" — T19 integration fills in.
void PreprocessorVulkan::record(VkCommandBuffer, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, float) {
    throw std::runtime_error("PreprocessorVulkan::record: deferred to T19");
}

VkBuffer PreprocessorVulkan::means2D_buffer() const { /* TODO T19 */ return VK_NULL_HANDLE; }
// ... other getters similar stubs
```

Add `invert_mat4` helper to `harmonyos_3dgs/include/math_utils.h` if not present (inline function using cofactor expansion; reference AAA-Gaussians inverse or Gauss-Jordan).

- [ ] **Step 2: Register in CMake**

In `harmonyos_3dgs/CMakeLists.txt`, add to `gs3d_vk_core` sources:
```cmake
src/vulkan/preprocessor_vulkan.cpp
```

- [ ] **Step 3: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cmake --build build 2>&1 | tail -10
```

Expected: clean build; no test yet.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp \
        harmonyos_3dgs/include/math_utils.h \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp2: PreprocessorVulkan::process() implementation + deinterleave"
```

---

## Task 7: `test_preprocess_pass_vk.cpp` — validate against CUDA golden (tiny fixture)

**Files:**
- Create: `harmonyos_3dgs/tests/test_preprocess_pass_vk.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

**Strategy:** Load SP-0 tiny fixture inputs (positions, scales, rotations, opacities, sh, filter_3D + camera reconstruction). Run `PreprocessorVulkan::process()`. Compare outputs against golden `preprocess_means2D.npy`, `preprocess_depths.npy`, `preprocess_conic_opacity.npy`, `preprocess_rgb.npy`, `preprocess_radii.npy`, `preprocess_tiles_touched.npy`. Dual-threshold per spec §1.2: abs<1e-5, rel<1e-4.

**Challenge:** tiny fixture camera is stored implicitly via `build_tiny_scene()`. Need to either (a) save camera to `.npy` at SP-0 dump time, or (b) rebuild camera in C++ matching `tools/dump_fixtures.py` exactly.

**Solution**: (b) — write `build_tiny_camera()` in C++ matching Python. The scene params (means3D/scales/rotations/opacities/sh/filter_3D) are already in the fixture's input-side npys... wait, they're NOT in the manifest (SP-0 only dumps outputs + RNG artifacts).

**Real solution**: add a new SP-0 sub-task to dump the INPUT tensors (or) extend the tiny fixture manifest. For now, add a Python-side one-off:

Actually simpler: write `build_tiny_scene_cpp.h` — a C++ replica of `build_tiny_scene()` producing identical bit-for-bit inputs (PRNG reimplementation). PyTorch Generator with seed=42 uses Philox. Replicating Philox in C++ is work.

**Simplest viable plan**: Dump the input tensors alongside the golden outputs. One-time sub-task.

### Sub-task 7.0: Dump input tensors to tiny fixture

Extend `tools/dump_tiny.py` to also dump `input_positions`, `input_scales`, `input_rotations`, `input_opacities`, `input_sh`, `input_filter_3D`, and camera components (viewmatrix, projmatrix, inv_viewprojmatrix, campos, tan_fovx, tan_fovy, H, W, sh_degree). These go into the same `step000001/cam0000/` directory with `input_*` prefix. Re-run to regenerate the fixture.

- [ ] **Step 1: Extend `tools/dump_tiny.py`**

Add after line `out["preprocess_radii"] = radii`:
```python
# Dump input tensors too, for C++ test reconstruction.
out["input_positions"]  = s["means3D"]
out["input_scales"]     = s["scales"]
out["input_rotations"]  = s["rotations"]
out["input_opacities"]  = s["opacities"]
out["input_sh"]         = s["sh"]
out["input_filter_3D"]  = s["filter_3D"]
out["input_viewmatrix"]     = s["viewmatrix"]
out["input_projmatrix"]     = s["projmatrix"]
out["input_inv_viewprojmatrix"] = s["inv_viewprojmatrix"]
out["input_campos"]         = s["campos"]
out["input_fov_size"]       = torch.tensor([s["tan_fovx"], s["tan_fovy"],
                                            float(s["W"]), float(s["H"])], device="cuda")
out["input_meta"]           = torch.tensor([float(s["sh_degree"]), float(s["sh_coeffs_per_g"]),
                                            float(s["H"]), float(s["W"])], device="cuda")
```

Regenerate:
```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
conda run -n aaa-gs python -m tools.dump_tool --fixture=tiny \
    --output=harmonyos_3dgs/tests/golden/tiny
ls harmonyos_3dgs/tests/golden/tiny/step000001/cam0000/input_*.npy
```

Expected: 12 new `input_*.npy` files.

- [ ] **Step 2: Write C++ test**

Create `harmonyos_3dgs/tests/test_preprocess_pass_vk.cpp`:
```cpp
#include <gtest/gtest.h>
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/vk_context.h"
#include "types.h"
#include "golden/npy_reader.h"
#include "golden/compare.h"
#include "golden/manifest.h"
#include <vector>

namespace {
std::string tiny_cam_dir() {
    return std::string(TEST_DATA_DIR) + "/../golden/tiny/step000001/cam0000";
}
}

TEST(PreprocessPass, MatchesCUDAGolden_Tiny) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    // Load input tensors from tiny fixture
    auto pos = load_npy(tiny_cam_dir() + "/input_positions.npy");
    auto scl = load_npy(tiny_cam_dir() + "/input_scales.npy");
    auto rot = load_npy(tiny_cam_dir() + "/input_rotations.npy");
    auto opa = load_npy(tiny_cam_dir() + "/input_opacities.npy");
    auto sh  = load_npy(tiny_cam_dir() + "/input_sh.npy");
    auto f3d = load_npy(tiny_cam_dir() + "/input_filter_3D.npy");
    auto vm  = load_npy(tiny_cam_dir() + "/input_viewmatrix.npy");
    auto pm  = load_npy(tiny_cam_dir() + "/input_projmatrix.npy");
    auto ivpm= load_npy(tiny_cam_dir() + "/input_inv_viewprojmatrix.npy");
    auto cp  = load_npy(tiny_cam_dir() + "/input_campos.npy");
    auto fs  = load_npy(tiny_cam_dir() + "/input_fov_size.npy");
    auto mt  = load_npy(tiny_cam_dir() + "/input_meta.npy");

    const int N = (int)pos.shape[0];
    const int sh_degree = (int)mt.f32()[0];
    const int M = (int)mt.f32()[1];
    const int H = (int)mt.f32()[2];
    const int W = (int)mt.f32()[3];

    GaussianData g{};
    g.count = N; g.sh_degree = sh_degree; g.max_coeffs = M;
    g.positions = pos.f32(); g.scales = scl.f32();
    g.rotations = rot.f32(); g.opacities = opa.f32();
    g.sh_coeffs = sh.f32();  g.filter_3D = f3d.f32();

    Camera cam{};
    std::memcpy(cam.view_matrix, vm.f32(), 64);
    std::memcpy(cam.viewproj_matrix, pm.f32(), 64);  // "projmatrix" here is actually full viewproj per build_tiny_scene
    cam.cam_pos[0] = cp.f32()[0]; cam.cam_pos[1] = cp.f32()[1]; cam.cam_pos[2] = cp.f32()[2];
    cam.tan_fovx = fs.f32()[0]; cam.tan_fovy = fs.f32()[1];
    cam.width = W; cam.height = H;

    RenderConfig cfg{};
    cfg.sh_degree = sh_degree; cfg.training = true; cfg.eval_3D = false;
    cfg.tile_w = 16; cfg.tile_h = 16; cfg.antialiasing = false;
    cfg.scale_modifier = 1.0f;

    FrameAllocator alloc(32 * 1024 * 1024);
    PreprocessOutput out{};
    PreprocessorVulkan preproc(ctx);
    preproc.process(g, cam, cfg, alloc, nullptr, out);

    // Load golden outputs
    auto g_means2D = load_npy(tiny_cam_dir() + "/preprocess_means2D.npy");
    auto g_depths  = load_npy(tiny_cam_dir() + "/preprocess_depths.npy");
    auto g_conic_op= load_npy(tiny_cam_dir() + "/preprocess_conic_opacity.npy");  // [N,4] packed
    auto g_rgb     = load_npy(tiny_cam_dir() + "/preprocess_rgb.npy");
    auto g_radii   = load_npy(tiny_cam_dir() + "/preprocess_radii.npy");
    auto g_tt      = load_npy(tiny_cam_dir() + "/preprocess_tiles_touched.npy");

    // Compare outputs (dual threshold)
    std::vector<float> m2d(out.means2D, out.means2D + N*2);
    std::vector<float> g_m2d(g_means2D.f32(), g_means2D.f32() + N*2);
    auto r1 = compare_f32(m2d, g_m2d, /*abs=*/1e-5f, /*rel=*/1e-4f);
    EXPECT_TRUE(r1.passed) << "means2D mismatch: max_abs=" << r1.max_abs_err
                           << " max_rel=" << r1.max_rel_err
                           << " first_bad=" << r1.first_bad_index;

    std::vector<float> dp(out.depths, out.depths + N);
    std::vector<float> g_dp(g_depths.f32(), g_depths.f32() + N);
    auto r2 = compare_f32(dp, g_dp, 1e-5f, 1e-4f);
    EXPECT_TRUE(r2.passed) << "depths mismatch";

    // Re-pack our conics + opacities_2d and compare to golden packed conic_opacity
    std::vector<float> packed(N*4);
    for (int i = 0; i < N; i++) {
        packed[i*4+0] = out.conics[i*3+0];
        packed[i*4+1] = out.conics[i*3+1];
        packed[i*4+2] = out.conics[i*3+2];
        packed[i*4+3] = out.opacities_2d[i];
    }
    std::vector<float> g_co(g_conic_op.f32(), g_conic_op.f32() + N*4);
    auto r3 = compare_f32(packed, g_co, 1e-5f, 1e-4f);
    EXPECT_TRUE(r3.passed) << "conic_opacity mismatch";

    std::vector<float> rgb(out.rgb, out.rgb + N*3);
    std::vector<float> g_rgb_v(g_rgb.f32(), g_rgb.f32() + N*3);
    auto r4 = compare_f32(rgb, g_rgb_v, 1e-5f, 1e-4f);
    EXPECT_TRUE(r4.passed) << "rgb mismatch";

    // radii + tiles_touched: exact integer match
    std::vector<uint32_t> radii_u(N);
    for (int i = 0; i < N; i++) radii_u[i] = (uint32_t)out.radii[i];
    std::vector<uint32_t> g_radii_u(g_radii.i32(), g_radii.i32() + N);
    EXPECT_TRUE(compare_u32(radii_u, g_radii_u)) << "radii mismatch";

    std::vector<uint32_t> tt_u(N);
    for (int i = 0; i < N; i++) tt_u[i] = (uint32_t)out.tiles_touched[i];
    std::vector<uint32_t> g_tt_u(g_tt.u32(), g_tt.u32() + N);
    EXPECT_TRUE(compare_u32(tt_u, g_tt_u)) << "tiles_touched mismatch";
}

TEST(PreprocessPass, RejectsEval3D) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVulkan preproc(ctx);
    GaussianData g{}; Camera cam{}; cam.width = 64; cam.height = 64;
    RenderConfig cfg{}; cfg.eval_3D = true; cfg.tile_w = 16; cfg.tile_h = 16;
    FrameAllocator alloc(1024); PreprocessOutput out{};
    EXPECT_THROW(preproc.process(g, cam, cfg, alloc, nullptr, out), std::runtime_error);
}

TEST(PreprocessPass, RejectsNon16Tile) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVulkan preproc(ctx);
    GaussianData g{}; Camera cam{}; cam.width = 64; cam.height = 64;
    RenderConfig cfg{}; cfg.tile_w = 8; cfg.tile_h = 16;
    FrameAllocator alloc(1024); PreprocessOutput out{};
    EXPECT_THROW(preproc.process(g, cam, cfg, alloc, nullptr, out), std::runtime_error);
}
```

- [ ] **Step 3: Register in CMakeLists**

Add `tests/test_preprocess_pass_vk.cpp` to `gs3d_vk_tests` sources.

- [ ] **Step 4: Build and run**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R PreprocessPass --output-on-failure
```

Expected: 3 tests pass. If `MatchesCUDAGolden_Tiny` fails, diff Vulkan output vs golden element-wise to find first divergence point (likely an algorithmic error in preprocess.comp — possibly matrix convention, SH eval, or cov projection). Fix iteratively.

- [ ] **Step 5: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/tests/test_preprocess_pass_vk.cpp \
        tools/dump_tiny.py \
        harmonyos_3dgs/tests/golden/tiny/step000001/cam0000/input_*.npy \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp2: PreprocessPass validated against CUDA golden tiny fixture

Extends dump_tiny.py to also dump 12 input tensors (positions, scales,
rotations, opacities, sh, filter_3D + camera). Adds test_preprocess_pass_vk.cpp
with 3 tests: CUDA golden match (dual-threshold) + 2 hard-error rejections."
```

---

## Task 8: `prefix_sum.comp` (3-level Blelloch scan)

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/prefix_sum.comp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

**Spec §4.8.2**: generic scan (used for both tiles_touched sum AND radix sort histograms). Three phases controlled by `phase` push constant:
- phase=0: local scan within each 256-thread workgroup, write group sum to `workgroup_sums[group_id]`
- phase=1: scan `workgroup_sums` (single-workgroup case since <= 1024 workgroups for ~262K elements; for >262K, extend to scan workgroup-of-workgroup sums)
- phase=2: add back accumulated offsets (workgroup_sums[group_id-1]) to local scan results

For SP-2 we only need to handle up to ~N (fixtures have N≤20, basketball ~100k). Single-level + add-back (2 dispatches) suffices for <=65K. Blelloch up to 3 levels for larger.

**Algorithm** (Blelloch scan — exclusive):
```
shared float shared[256];
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint local = gl_LocalInvocationID.x;
    if (pc.phase == 0) {
        // Load
        shared[local] = (i < pc.num_elements) ? input_array[i] : 0;
        barrier();
        // Up-sweep
        for (uint d = 1; d < 256; d *= 2) {
            if (local % (2*d) == 0 && local+2*d-1 < 256)
                shared[local+2*d-1] += shared[local+d-1];
            barrier();
        }
        // Last thread stores total (workgroup sum) and clears for down-sweep
        if (local == 255) { workgroup_sums[gl_WorkGroupID.x] = shared[255]; shared[255] = 0; }
        barrier();
        // Down-sweep
        for (uint d = 128; d >= 1; d /= 2) {
            if (local % (2*d) == 0 && local+2*d-1 < 256) {
                float t = shared[local+d-1];
                shared[local+d-1] = shared[local+2*d-1];
                shared[local+2*d-1] += t;
            }
            barrier();
        }
        if (i < pc.num_elements) output_array[i] = shared[local];
    }
    // phase 1, 2 similar patterns — see full shader code below
}
```

**Full shader** — write out ~100 lines matching spec §4.8.2 binding and push constant. Test in T10.

- [ ] **Step 1: Write shader**

Create `harmonyos_3dgs/src/vulkan/shaders/prefix_sum.comp`. 100-150 lines. Use `std430` for input/output, `uint` data type. Handle all 3 phases. For single-workgroup (<=256 elements) case, phase=0 is sufficient.

- [ ] **Step 2: Wire into CMakeLists**

```cmake
gs3d_compile_glsl(prefix_sum _spv_prefix_sum)
gs3d_embed_spirv(prefix_sum _hdr_prefix_sum)
```
Update `gs3d_vulkan_shaders` DEPENDS.

- [ ] **Step 3: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
cmake --build build --target gs3d_vulkan_shaders 2>&1 | tail -3
ls build/shaders/prefix_sum.spv build/shaders/prefix_sum_spv.h
```

Expected: both files created.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add harmonyos_3dgs/src/vulkan/shaders/prefix_sum.comp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp2: prefix_sum.comp (3-level Blelloch scan)"
```

---

## Task 9: `scatter.comp` (duplicateWithKeys)

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/scatter.comp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

**Spec §4.8.3**: one thread per Gaussian. For each Gaussian i with `radii[i] > 0`, compute tile AABB from `means2D[i]` and `radii[i]`. For each touched tile (tx, ty), write `(tile_key, gaussian_id)` pair at `point_offsets[i] + local_offset`. Tile key = `(tile_y << 32) | (uint32_t)bit_cast(depth)` where depth bits are ordered for radix sort.

Actually, spec §4.8.3 just says `keys_unsorted[R]` uint64 and `values_unsorted[R]` uint32. Key format: high 32 bits = tile_id = `ty * num_tiles_x + tx`, low 32 bits = depth (reinterpreted as sortable uint32 via float bit pattern).

- [ ] **Step 1: Write shader**

Create `harmonyos_3dgs/src/vulkan/shaders/scatter.comp`:
```glsl
#version 450
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer M { float means2D[]; };     // [N*2]
layout(std430, binding = 1) readonly buffer D { float depths[]; };       // [N]
layout(std430, binding = 2) readonly buffer R { int   radii[]; };        // [N]
layout(std430, binding = 3) readonly buffer O { uint  point_offsets[]; }; // [N]
layout(std430, binding = 4) readonly buffer T { uint  tiles_touched[]; };// [N]
layout(std430, binding = 5) writeonly buffer KU { uint64_t keys_unsorted[]; };   // [R]
layout(std430, binding = 6) writeonly buffer VU { uint     values_unsorted[]; }; // [R]

layout(push_constant) uniform PC {
    uint num_gaussians;
    uint num_tiles_x;
    uint tile_w;   // 16
    uint tile_h;   // 16
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_gaussians) return;
    if (radii[i] <= 0 || tiles_touched[i] == 0u) return;

    float cx = means2D[i*2];
    float cy = means2D[i*2+1];
    int r = radii[i];
    int min_tx = max(0, int((cx - r) / pc.tile_w));
    int max_tx = min(int(pc.num_tiles_x) - 1, int((cx + r) / pc.tile_w));
    int min_ty = max(0, int((cy - r) / pc.tile_h));
    int max_ty = int((cy + r) / pc.tile_h);

    // Sortable uint32 bits from float depth: sign-flip trick ensures IEEE-754
    // floats sort correctly as unsigned ints when they're all non-negative.
    // For non-negative depths, bit pattern preserves ordering directly.
    // For any depth (pos/neg), use: u = (depth < 0) ? ~floatBitsToUint(depth) : floatBitsToUint(depth) ^ 0x80000000u
    uint depth_bits = floatBitsToUint(depths[i]);
    // Our depths are all positive (passed near-plane), so simple cast suffices.

    uint offset = point_offsets[i];
    uint local = 0;
    for (int ty = min_ty; ty <= max_ty; ty++) {
        for (int tx = min_tx; tx <= max_tx; tx++) {
            uint tile_id = uint(ty) * pc.num_tiles_x + uint(tx);
            uint64_t key = (uint64_t(tile_id) << 32) | uint64_t(depth_bits);
            keys_unsorted[offset + local] = key;
            values_unsorted[offset + local] = i;
            local++;
        }
    }
}
```

Check: on HarmonyOS / Maleoon, `GL_ARB_gpu_shader_int64` availability — verified via SP-1 capability probe; Tegra Thor supports it. If not on target, uint64 can be emulated with two uint32 — deferred to HarmonyOS deployment phase.

- [ ] **Step 2: Wire into CMakeLists + build + commit** (same pattern as T8)

---

## Task 10: `PrefixScanPass` + `ScatterPass` + `TileBinnerVulkan` + golden tests

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/tile_binner_passes.h` (PrefixScanPass, ScatterPass classes)
- Create: `harmonyos_3dgs/src/vulkan/prefix_scan_pass.cpp`, `scatter_pass.cpp`
- Create: `harmonyos_3dgs/include/vulkan/tile_binner_vulkan.h` + .cpp
- Create: `harmonyos_3dgs/tests/test_prefix_scan_pass_vk.cpp`, `test_scatter_pass_vk.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

**Pattern:** each pass class is analogous to `PreprocessPass` (T5). `TileBinnerVulkan` adapter:
- Owns private `PrefixScanPass` + `ScatterPass` instances
- `bin(PreprocessOutput, N, camera, cfg, alloc) → BinningOutput` public method: runs scan (on tiles_touched→point_offsets), then scatter (generates keys_unsorted + values_unsorted)
- Internally allocates `point_offsets` buffer (private, not in public BinningOutput per spec §4.6)

**Tests:**
- `test_prefix_scan_pass_vk.cpp`: generic scan on synthetic uint arrays — small (64), boundary (256), medium (4096). Compare against CPU exclusive scan.
- `test_scatter_pass_vk.cpp`: load golden `preprocess_tiles_touched`, `preprocess_point_offsets`, `preprocess_means2D`, `preprocess_depths`, `preprocess_radii` from tiny fixture. Run scatter. Compare output `keys_unsorted` + `values_unsorted` to golden `sort_keys_unsorted.npy` + `sort_values_unsorted.npy`.

- [ ] **Step 1**: Write headers + implementations (~200 lines each)
- [ ] **Step 2**: Register in CMake
- [ ] **Step 3**: Write tests
- [ ] **Step 4**: Build + run
- [ ] **Step 5**: Commit

Acceptance: tiny fixture `sort_keys_unsorted.npy` [R=103] uint64 and `sort_values_unsorted.npy` [R=103] uint32 match Vulkan output byte-for-byte (compare_u64 / compare_u32).

---

## Task 11: `TileBinnerVulkan` full integration with golden

**Files:**
- Create: `harmonyos_3dgs/tests/test_tile_binner_vulkan.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

Tests the full `TileBinnerVulkan::bin()` public method end-to-end against tiny fixture golden. Inputs: `preprocess_*.npy` (use those already on disk, don't regenerate). Outputs: compare against `sort_keys_unsorted.npy`, `sort_values_unsorted.npy`, and internal `preprocess_point_offsets.npy` (accessible via a debug accessor on TileBinnerVulkan or by dispatching the `PrefixScanPass` directly in the test).

- [ ] **Step 1**: Write end-to-end test
- [ ] **Step 2**: Build + run, expect PASS
- [ ] **Step 3**: Commit

---

## Task 12: `radix_sort_count.comp` + `radix_sort_scatter.comp`

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/radix_sort_count.comp`
- Create: `harmonyos_3dgs/src/vulkan/shaders/radix_sort_scatter.comp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

**Spec §4.8.4, §4.8.5**: 4-bit radix, 16 passes total. Each pass:
1. `radix_sort_count.comp` — local histogram per workgroup (16 buckets), write to `histograms[wg_id * 16 + bucket]`
2. `prefix_sum.comp` (reused from T8, phase 0+1+2) — exclusive scan over histograms
3. `radix_sort_scatter.comp` — stable scatter: each workgroup reads keys, computes bucket, writes to `keys_out[bucket_offsets[wg_id*16+bucket]++]`

Stable scatter requires per-workgroup local rank within bucket. Use `subgroupInclusiveAdd` or local-memory prefix sum.

**For SP-2 Phase 1**, simplest viable implementation: single-workgroup radix (works for R ≤ 256 — tiny fixture has R=103). Per-workgroup local scan for in-workgroup rank + global offset = per-pass position. Scales to larger R in a follow-up; basketball has R≈10K-100K (exceeds single workgroup, needs multi-workgroup stable radix — deferred to Phase 2 if tiny golden passes).

- [ ] **Step 1**: Write both shaders (~150 lines each)
- [ ] **Step 2**: Wire CMake + build
- [ ] **Step 3**: Commit

---

## Task 13: `tile_range.comp`

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/tile_range.comp`
- Modify: CMakeLists.txt

**Spec §4.8.6**: one thread per element in sorted keys. For each position i, extract `tile_id = keys_sorted[i] >> 32`. If `i == 0 || (keys_sorted[i-1] >> 32) != tile_id`, this is the start of tile `tile_id`. If `i == R-1 || (keys_sorted[i+1] >> 32) != tile_id`, this is the end. Write `tile_ranges[tile_id * 2 + 0/1]` accordingly.

- [ ] Straightforward ~50 lines. Standard task pattern.

---

## Task 14: `RadixSortPass` + `TileRangePass` + `SorterVulkan` + golden tests

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/sort_passes.h` (RadixSortPass + TileRangePass)
- Create: `harmonyos_3dgs/src/vulkan/radix_sort_pass.cpp`, `tile_range_pass.cpp`
- Create: `harmonyos_3dgs/include/vulkan/sorter_vulkan.h` + .cpp
- Create: `harmonyos_3dgs/tests/test_radix_sort_pass_vk.cpp`, `test_tile_range_pass_vk.cpp`, `test_sorter_vulkan.cpp`
- Modify: CMakeLists.txt

**`RadixSortPass`**: internally orchestrates 16 iterations of (count → scan → scatter) using `PrefixScanPass` from T10 for the scan step. Ping-pongs between two key/value buffer pairs.

**`SorterVulkan : Sorter`**: 
- `sort(BinningOutput& bin, FrameAllocator&)` public method
- Internally runs RadixSortPass then TileRangePass
- In-place: fills `keys_sorted`/`values_sorted` and `tile_ranges` fields of BinningOutput

**Tests**:
- `test_radix_sort_pass_vk`: load `sort_keys_unsorted.npy` + `sort_values_unsorted.npy`, sort, compare against `sort_keys_sorted.npy` + `sort_values_sorted.npy` byte-for-byte (exact match — CUB is deterministic per SP-0 T22).
- `test_tile_range_pass_vk`: load `sort_keys_sorted.npy`, compute ranges, compare against `sort_tile_ranges.npy`.
- `test_sorter_vulkan`: end-to-end: load unsorted, sort+range in one `sort()` call, compare all 3 outputs.

- [ ] 5 sub-steps (same pattern as T10)

---

## Task 15: Sort integration validation

**Files:**
- Modify: `test_sorter_vulkan.cpp` (add chained integration test)

Run `TileBinnerVulkan::bin()` + `SorterVulkan::sort()` sequentially, starting from golden preprocess outputs. Verify final `keys_sorted`, `values_sorted`, `tile_ranges` match golden. Also test `record()` + barriers chain path.

- [ ] Single integration test. Quick.

---

## Task 16: `rasterize.comp` (per-tile alpha blending)

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp`
- Modify: CMakeLists.txt

**Spec §4.8.7**: 16×16 workgroup = 1 tile per workgroup. Each thread = 1 pixel. Shared memory (per workgroup): batch-load 256 Gaussian records from sorted list. Inner loop: iterate sorted gaussians in this tile's range, compute alpha from conic+opacity at pixel, blend front-to-back. Track per-pixel T (transmittance). Write `out_image[3, H, W]` + `transmittance[H*W]` + `n_contrib[H*W]`.

**Shared memory layout per spec §4.7 (separate arrays, no structs)**:
```glsl
shared float s_mean_x[256], s_mean_y[256];
shared float s_conic_a[256], s_conic_b[256], s_conic_c[256];
shared float s_opacity[256];
shared float s_rgb_r[256], s_rgb_g[256], s_rgb_b[256];
shared uint  s_gid[256];
// Total: 10.24 KB (< 16 KB minimum)
```

**Algorithm**:
```
tile_id = gl_WorkGroupID.y * num_tiles_x + gl_WorkGroupID.x
range = tile_ranges[tile_id]  // .x=start, .y=end
pixel_x = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x
pixel_y = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y
valid_pixel = (pixel_x < W && pixel_y < H)
T = 1.0; C = vec3(0); n = 0
for batch in range [start .. end] step 256:
    // Load up to 256 gaussians cooperatively
    if (gl_LocalInvocationIndex + batch_start < range.y):
        gid = values_sorted[batch_start + local_idx]
        s_mean_x[local_idx] = means2D[gid*2]
        ... (populate all shared arrays)
    barrier()
    // Per-pixel inner loop
    if (valid_pixel && T > 1e-4):
        batch_size = min(256, range.y - batch_start)
        for k in 0..batch_size:
            dx = pixel_x - s_mean_x[k]; dy = pixel_y - s_mean_y[k]
            power = -0.5 * (s_conic_a[k]*dx*dx + s_conic_c[k]*dy*dy) - s_conic_b[k]*dx*dy
            if power > 0 skip
            alpha = min(0.99, s_opacity[k] * exp(power))
            if alpha < 1e-4 skip
            new_T = T * (1 - alpha)
            if new_T < 1e-4 { T = new_T; break }
            C += s_rgb_r/g/b[k] * alpha * T
            T = new_T
            n++
    barrier()
// Write outputs
if (valid_pixel):
    px = pixel_y * W + pixel_x
    out_image[0 * H*W + px] = C.r  // CHW layout
    out_image[1 * H*W + px] = C.g
    out_image[2 * H*W + px] = C.b
    transmittance[px] = T
    n_contrib[px] = n
```

- [ ] Write shader ~150 lines. Complex but bounded.
- [ ] Wire CMake + build + commit

---

## Task 17: `RasterizePass` + `RasterizerVulkan` adapter

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/rasterize_pass.h` + .cpp
- Create: `harmonyos_3dgs/include/vulkan/rasterizer_vulkan.h` + .cpp
- Modify: CMakeLists.txt

Standard pattern. `RasterizerVulkan::rasterize(PreprocessOutput, BinningOutput, Camera, RenderConfig, float* output_image, ..., ForwardCache*, FrameAllocator*) override`.

- [ ] 5 sub-steps same as T5+T6.

---

## Task 18: `test_rasterize_pass_vk.cpp` — validate against golden image

**Files:**
- Create: `harmonyos_3dgs/tests/test_rasterize_pass_vk.cpp`
- Modify: CMakeLists.txt

Load all golden inputs: `preprocess_means2D/conic_opacity/rgb`, `sort_tile_ranges/values_sorted`, camera. Run `RasterizerVulkan::rasterize()`. Compare:
- `rasterize_image.npy` [3, H, W] → abs<1e-5, rel<1e-4 (≈ PSNR > 100dB)
- `rasterize_transmittance.npy` [H*W] → abs<1e-5
- `rasterize_n_contrib.npy` [H*W] uint32 → exact match

- [ ] 5 sub-steps. If image mismatch, debug shared memory aliasing, alpha blending order, batch boundaries.

---

## Task 19: `test_forward_pipeline_vk.cpp` — full chain integration via `record()`

**Files:**
- Create: `harmonyos_3dgs/tests/test_forward_pipeline_vk.cpp`
- Modify: CMakeLists.txt

End-to-end test using the layer-2 `record()` API: one `VkCommandBuffer`, record preprocess → barrier → tilebin → barrier → sort → barrier → rasterize, submit once. Compare final image against `rasterize_image.npy`. Must also implement the `record()` stubs on each adapter class that were deferred from T6, T10, T14, T17.

- [ ] **Step 1**: Implement `record()` on all 4 adapters (fill in the stubs)
- [ ] **Step 2**: Write integration test
- [ ] **Step 3**: Build + run
- [ ] **Step 4**: Commit

---

## Task 20: Full validation + regression

**Files:** none (verification only unless fixes needed).

- [ ] **Step 1**: Clean build VULKAN ON

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=ON -DENABLE_OPENCL=OFF 2>&1 | tail -5
cmake --build build 2>&1 | tail -5
```

- [ ] **Step 2**: Full ctest

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -15
```

Expected test count:
- 165 baseline (SP-0) + 10 SP-1 + ~15 new SP-2 = ~190 tests. Fewer than SP-1's 187 because we deleted 9 old PreprocessorVK tests and added ~17 new. Target: ≥ 192.

New SP-2 test families:
- `PreprocessPass.*` (3)
- `PrefixScanPass.*` (3)
- `ScatterPass.*` (2)
- `TileBinnerVulkan.*` (1)
- `RadixSortPass.*` (2)
- `TileRangePass.*` (1)
- `SorterVulkan.*` (2)
- `RasterizePass.*` (3)
- `ForwardPipeline.*` (1)
- Total: ~18

All must pass. If any fail, debug and commit fixes.

- [ ] **Step 3**: VULKAN OFF regression

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=OFF -DENABLE_OPENCL=OFF 2>&1 | tail -3
cmake --build build 2>&1 | tail -3
ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

Expected: 165 tests pass, no Vulkan. Same as SP-1 OFF baseline.

- [ ] **Step 4**: Return to ON

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_OPENCL=OFF 2>&1 | tail -3
cmake --build build 2>&1 | tail -3
```

---

## Task 21: Close notes + tag `sp2-forward-ready`

**Files:**
- Create: `dev_notes/sp2_forward_notes.md`

- [ ] **Step 1: Write notes**

```markdown
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
  `process/bin/sort/rasterize` overrides + Vulkan-only `record(cmd, ...)`.
- Single source of truth: `preprocess_bindings.h` (binding indices, push
  constants, specialization constants), `vk_camera_ubo.h` (std140 struct).
- Hard-error guards per spec §4.4: `eval_3D`, non-16×16 tiles, `antialiasing`
  all reject at entry.
- ~18 new tests, all validated against SP-0 tiny fixture CUDA golden.
- Extended `dump_tiny.py` to dump input tensors (positions, scales, ...,
  camera components) for C++-side test reconstruction.
- Extended `VulkanComputePipeline` with mixed binding-types constructor
  (SSBO + UBO) and `update_ubo()` helper.

## Test counts

- CPU + SP-0: 165
- SP-1 Vulkan infra: 10
- SP-2 new: 18
- **Total: ~193 tests**

## Handoff to SP-3 (backward pipeline)

- Packed `conic_opacity_packed` layout established. SP-3 `rasterize_backward`
  will consume the same packed buffer (no deinterleave needed on backward path).
- `record()` + `insert_compute_barrier(cmd)` pattern proven in forward chain.
  SP-3 chains onto the same command buffer style.
- `has_shader_atomic_float` = true on NVIDIA Thor (per SP-1 capability probe),
  so SP-3's rasterize_backward will use native atomicAdd(float) path.
- CUDA golden backward tensors already dumped by SP-0 tiny fixture:
  `backward_d_means2D/d_colors/d_conic/d_opacity/d_means3D/d_cov3D/d_sh/d_scales/d_rotations.npy`.

## Known limits (deferred)

- `rasterize.comp` currently assumes all Gaussians fit in one 256-batch inside
  shared memory load. For very dense tiles (many thousands of Gaussians in
  one 16x16 tile) batched loop handles it correctly but large batches stress
  GPU scheduler. Basketball numbers not yet stress-tested.
- Radix sort Phase 1 uses a simpler single-workgroup scatter; multi-workgroup
  stable scatter for R > 65536 deferred. Tiny fixture R=103 well within limits.
- `PreprocessorVulkan` uses host-visible (Phase 1 simple) buffers per SP-1
  VulkanBuffer. Device-local + staging optimization deferred to performance
  phase after SP-5 convergence validation.
```

- [ ] **Step 2: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp2
git add dev_notes/sp2_forward_notes.md
git -c commit.gpgsign=false commit -m "sp2: close notes + SP-3 handoff checklist"
```

- [ ] **Step 3: Tag**

```bash
git tag -a sp2-forward-ready -m "SP-2 forward pipeline complete: 6 passes + 4 adapter classes per spec §4, validated against SP-0 CUDA golden tiny fixture. ~193 tests pass."
git tag -l sp2-forward-ready -n
```

---

## SP-2 Exit Criteria

- [ ] All 21 tasks completed
- [ ] ~193 tests passing (165 baseline + 10 SP-1 + ~18 SP-2)
- [ ] Tiny fixture CUDA golden matches at every pass boundary (dual-threshold PASS on means2D/depths/conic_opacity/rgb; exact match on radii/tiles_touched/sort keys/values/tile_ranges)
- [ ] `rasterize_image` matches golden with abs<1e-5 (≈ PSNR > 100dB)
- [ ] `ENABLE_VULKAN=OFF` regression: 165 tests pass with no Vulkan targets
- [ ] `record()` chain integration test passes: full forward pipeline in one command buffer
- [ ] `PreprocessorVulkan::process()` rejects eval_3D, non-16x16 tiles, antialiasing
- [ ] `dev_notes/sp2_forward_notes.md` committed
- [ ] Tag `sp2-forward-ready` exists
