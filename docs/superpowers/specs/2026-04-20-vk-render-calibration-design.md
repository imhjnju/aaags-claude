# Vulkan Render Calibration — Phase 1 Design

**Date:** 2026-04-20
**Goal:** Create `gs3d_vk_render` CLI tool and a CPU-vs-Vulkan comparison test to validate the Vulkan forward pipeline produces correct rendering output.
**Scope:** Phase 1 — `eval_3D=false` only. Phase 2 (eval_3D shaders + Python golden) is a separate design.

---

## 1. Motivation

`render_single.py` implements the correct AAA-GS rendering result using the Python/CUDA pipeline. Before we can calibrate `gs3d_vk_render` against it (Phase 2, which requires `eval_3D=true` in Vulkan shaders), we first need to prove the Vulkan forward pipeline wiring is correct by comparing against the CPU backend under `eval_3D=false`.

The Vulkan pipeline (PreprocessorVulkan, TileBinnerVulkan, SorterVulkan, RasterizerVulkan) implements the same abstract interfaces as the CPU backend. Both plug into the `Renderer` class. If given identical inputs and `eval_3D=false`, they should produce near bit-exact output (PSNR > 60 dB).

## 2. Components

### 2.1 Camera Utilities Extraction

**Files:** `include/camera_utils.h`, `src/camera_utils.cpp`

Extract the following from `src/main.cpp` static functions into a shared utility:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `buildLookAt` | `(const float eye[3], const float center[3], const float up[3], float view_matrix[16])` | Look-at view matrix, column-major |
| `buildPerspective` | `(float tan_fovx, float tan_fovy, float znear, float zfar, float proj[16])` | Perspective projection matrix, column-major |
| `mat4Mul` | `(const float A[16], const float B[16], float out[16])` | 4x4 column-major matrix multiply |
| `loadCameraJson` | `(const char* json_path, int cam_id)` → `Camera` | Parse cameras.json, build Camera struct |
| `autoCamera` | `(const GaussianData& g, int width, int height)` → `Camera` | Bounding-box auto camera |

Dependencies: `types.h` only (for `Camera`, `GaussianData`).

`src/main.cpp` is modified to `#include "camera_utils.h"` and remove its static copies.

`camera_utils.cpp` is added to the `gs3d_core` static library in CMakeLists.txt.

### 2.2 Vulkan Render CLI (`gs3d_vk_render`)

**File:** `src/vulkan/vk_render_main.cpp`
**CMake target:** `gs3d_vk_render` (already declared at CMakeLists.txt:261)

```
Usage: gs3d_vk_render <model.ply> <cameras.json> [cam_id]
```

Flow:
1. Parse args: PLY path (required), cameras.json (required), cam_id (default 0)
2. `VulkanContext ctx; ctx.init()` — hard-error if no Vulkan device
3. `loadPly(argv[1])` → `LoadedModel`
4. `loadCameraJson(argv[2], cam_id)` → `Camera`
5. Configure `RenderConfig`:
   - `eval_3D = false` (Phase 1)
   - `bg_color` from `BG_WHITE` env var (0 or 1)
   - `sh_degree` from model
   - `antialiasing = false`
6. Create `Renderer` with Vulkan backends:
   ```cpp
   Renderer(
       std::make_unique<PreprocessorVulkan>(ctx),
       std::make_unique<TileBinnerVulkan>(ctx),
       std::make_unique<SorterVulkan>(ctx),
       std::make_unique<RasterizerVulkan>(ctx))
   ```
7. `renderer->render(model.data, cam, config, image.data())`
8. `writePPM("output_vk.ppm", image.data(), W, H)` + print timing

No multi-angle mode, no auto-camera, no OpenCL — focused tool for calibration.

### 2.3 CPU vs Vulkan Comparison Test

**File:** `tests/test_vk_vs_cpu_render.cpp`
**Added to:** `gs3d_vk_tests` executable in CMakeLists.txt

**Test case:** `VkVsCpuRender.FullFramePSNR`

Flow:
1. Skip (not fail) if `basket-aaa.ply` is missing or `VulkanContext::init()` fails
2. Load `basket-aaa.ply` via `loadPly()`
3. Load camera 0 from `cameras.json` via `loadCameraJson()`
4. `RenderConfig`: `eval_3D=false`, black bg, `sh_degree` from model
5. Render with CPU backend:
   ```cpp
   Renderer cpu_renderer(
       std::make_unique<PreprocessorCPU>(),
       std::make_unique<TileBinnerCPU>(),
       std::make_unique<SorterCPU>(),
       std::make_unique<RasterizerCPU>());
   cpu_renderer.render(model.data, cam, config, cpu_image.data());
   ```
6. Render with Vulkan backend:
   ```cpp
   Renderer vk_renderer(
       std::make_unique<PreprocessorVulkan>(ctx),
       std::make_unique<TileBinnerVulkan>(ctx),
       std::make_unique<SorterVulkan>(ctx),
       std::make_unique<RasterizerVulkan>(ctx));
   vk_renderer.render(model.data, cam, config, vk_image.data());
   ```
7. Compare all `H * W * 3` pixel values:
   - Compute MSE: `sum((cpu[i] - vk[i])^2) / (H*W*3)`
   - PSNR: `10 * log10(1.0 / MSE)` (pixel range [0,1])
   - Track max absolute per-pixel error
   - Print diagnostics: PSNR, max error, error location
8. Assertions:
   - `EXPECT_GT(psnr, 60.0)` — strict near bit-exact match
   - `EXPECT_LT(max_error, 0.001)` — no single pixel off by more than 0.1%

**Skip logic:** Uses `GTEST_SKIP()` when:
- `basket-aaa.ply` not found at expected path
- `VulkanContext::init()` returns false (no GPU)

## 3. CMake Changes

1. Add `src/camera_utils.cpp` to `gs3d_core` library sources
2. Append `tests/test_vk_vs_cpu_render.cpp` to `gs3d_vk_tests` sources
3. Add `REPO_ROOT_DIR` compile definition to `gs3d_vk_tests` pointing to `${CMAKE_CURRENT_SOURCE_DIR}/..` — needed to locate `basket-aaa.ply` (at repo root) and `cameras.json` (at `harmonyos_3dgs/cameras.json`)
4. No changes needed for `gs3d_vk_render` target (already declared, just needs source file to exist)

**Asset paths in test:**
- PLY: `REPO_ROOT_DIR "/basket-aaa.ply"`
- Cameras: `TEST_DATA_DIR "/../../cameras.json"` (or `REPO_ROOT_DIR "/harmonyos_3dgs/cameras.json"`)

## 4. Output Layout Fix (CHW→HWC in RasterizerVulkan)

**Problem:** `rasterize.comp` writes CHW (`out_image[ch*HW+px]`). The CPU rasterizer writes HWC (`out_img[px*3+ch]`). Currently `RasterizerVulkan::rasterize()` returns raw CHW, which breaks the `Rasterizer` interface contract. `VulkanTrainer` works around this with an explicit CHW→HWC copy after calling `rasterize()` (vulkan_trainer.cpp:317-325, with a TODO to fix the shader/adapter).

**Fix:** Add CHW→HWC conversion at the end of `RasterizerVulkan::rasterize()` (Layer-1 path). This ensures the `Rasterizer` interface consistently returns HWC regardless of backend. Then remove the duplicate conversion in `vulkan_trainer.cpp`.

Modified files:
- `src/vulkan/rasterizer_vulkan.cpp` — add ~6 lines CHW→HWC after download
- `src/vulkan_trainer.cpp` — remove the `chw_tmp` copy block (lines ~317-325) and its TODO

This is a bugfix, not a feature — the interface contract was always HWC.

## 5. Phase 2 (Out of Scope)

Phase 2 will be a separate design covering:
- `eval_3D=true` path in `preprocess.comp` (gauss2screen, cov3D_inv, AABB, frustum culling)
- StopThePop K-buffer rasterizer in `rasterize.comp` (depthAlongRay, maxContribRayPixel, per-pixel sorting)
- Per-tile depth keys in tile binner
- Golden comparison against `render_single.py` Python/CUDA output

## 6. Success Criteria

- `gs3d_vk_render basket-aaa.ply cameras.json 0` produces a valid PPM matching CPU output visually
- `VkVsCpuRender.FullFramePSNR` passes with PSNR > 60 dB
- Build remains green: all 243+ existing tests pass
- No new warnings
