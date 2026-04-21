# Vulkan Render Calibration — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `gs3d_vk_render` CLI tool and a CPU-vs-Vulkan full-frame comparison test proving the Vulkan forward pipeline produces correct output (PSNR > 60 dB).

**Architecture:** Plug Vulkan backends (PreprocessorVulkan, TileBinnerVulkan, SorterVulkan, RasterizerVulkan) into the existing `Renderer` class. Extract camera utilities from `main.cpp` into shared code. Fix the CHW→HWC layout mismatch in `RasterizerVulkan::rasterize()`. Test compares in-memory renders from both backends on `basket-aaa.ply` cam 0.

**Tech Stack:** C++17, CMake, GoogleTest, Vulkan compute, existing `gs3d_core`/`gs3d_vk_core` libraries.

**Spec:** `docs/superpowers/specs/2026-04-20-vk-render-calibration-design.md`

---

### Task 1: Extract camera utilities from main.cpp

**Files:**
- Create: `harmonyos_3dgs/include/camera_utils.h`
- Create: `harmonyos_3dgs/src/camera_utils.cpp`
- Modify: `harmonyos_3dgs/src/main.cpp` (lines 28–264)
- Modify: `harmonyos_3dgs/CMakeLists.txt` (line 22, gs3d_core sources)

- [ ] **Step 1: Create `camera_utils.h` header**

```cpp
// harmonyos_3dgs/include/camera_utils.h
#pragma once
#include "types.h"

// Build a look-at view matrix (column-major, matching GLM/CUDA convention).
void buildLookAt(const float eye[3], const float center[3], const float up[3],
                 float view_matrix[16]);

// Build perspective projection matrix (column-major).
void buildPerspective(float tan_fovx, float tan_fovy, float znear, float zfar,
                      float proj[16]);

// Multiply two 4x4 column-major matrices: out = A * B.
void mat4Mul(const float A[16], const float B[16], float out[16]);

// Auto-compute camera from scene bounding box (5th/95th percentile).
Camera autoCamera(const GaussianData& g, int width, int height);

// Load camera from cameras.json (3DGS format).
// cameras.json stores: position (world camera center), rotation (C2W 3x3),
// fx, fy, width, height.
Camera loadCameraJson(const char* json_path, int cam_id);
```

- [ ] **Step 2: Create `camera_utils.cpp` implementation**

Move the bodies of `buildLookAt` (main.cpp:28–63), `buildPerspective` (main.cpp:66–74), `mat4Mul` (main.cpp:77–85), `autoCamera` (main.cpp:88–142), and `loadCameraJson` (main.cpp:146–264) into `camera_utils.cpp`. Remove the `static` keyword from each function. Add `#include "camera_utils.h"` and the same includes the originals used (`<cmath>`, `<cstring>`, `<algorithm>`, `<vector>`, `<fstream>`, `<string>`, `<stdexcept>`, `<cstdio>`).

```cpp
// harmonyos_3dgs/src/camera_utils.cpp
#include "camera_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void buildLookAt(const float eye[3], const float center[3], const float up[3],
                 float view_matrix[16]) {
    // [exact body from main.cpp:29–63, unchanged]
    ...
}

void buildPerspective(float tan_fovx, float tan_fovy, float znear, float zfar,
                      float proj[16]) {
    // [exact body from main.cpp:67–74, unchanged]
    ...
}

void mat4Mul(const float A[16], const float B[16], float out[16]) {
    // [exact body from main.cpp:78–85, unchanged]
    ...
}

Camera autoCamera(const GaussianData& g, int width, int height) {
    // [exact body from main.cpp:89–142, unchanged]
    ...
}

Camera loadCameraJson(const char* json_path, int cam_id) {
    // [exact body from main.cpp:147–264, unchanged]
    ...
}
```

The implementer must copy these bodies verbatim from `main.cpp`, removing only the `static` keyword.

- [ ] **Step 3: Add `camera_utils.cpp` to `gs3d_core` in CMakeLists.txt**

In `harmonyos_3dgs/CMakeLists.txt`, add `src/camera_utils.cpp` to the `gs3d_core` library source list (after line 43, `src/train_utils.cpp`):

```cmake
    src/train_utils.cpp
    src/camera_utils.cpp
)
```

- [ ] **Step 4: Update `main.cpp` to use shared utilities**

In `harmonyos_3dgs/src/main.cpp`:
1. Add `#include "camera_utils.h"` after the existing includes (after line 25).
2. Delete lines 27–264 (the five static functions: `buildLookAt`, `buildPerspective`, `mat4Mul`, `autoCamera`, `loadCameraJson`).
3. The rest of `main()` (line 266 onward) calls these functions by the same names — no call-site changes needed.

- [ ] **Step 5: Build and run existing tests**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build
ctest --test-dir build --output-on-failure -R "RendererE2E|PlyLoader"
```

Expected: build succeeds, existing tests pass. The `gs3d_render` binary still works because `camera_utils` is linked via `gs3d_core`.

- [ ] **Step 6: Commit**

```bash
git add harmonyos_3dgs/include/camera_utils.h harmonyos_3dgs/src/camera_utils.cpp \
        harmonyos_3dgs/src/main.cpp harmonyos_3dgs/CMakeLists.txt
git commit -m "refactor: extract camera utilities from main.cpp into shared camera_utils"
```

---

### Task 2: Fix CHW→HWC layout in RasterizerVulkan::rasterize()

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/rasterizer_vulkan.cpp` (after line 204)
- Modify: `harmonyos_3dgs/src/vulkan_trainer.cpp` (lines 303–329)

- [ ] **Step 1: Add CHW→HWC conversion to RasterizerVulkan::rasterize()**

In `harmonyos_3dgs/src/vulkan/rasterizer_vulkan.cpp`, after line 204 (`img_buf->download(output_image, ...)`), before the `if (cache)` block, insert:

```cpp
    // Convert GPU CHW layout to CPU HWC layout.
    // rasterize.comp writes: out_image[ch * HW + px]  (CHW)
    // Rasterizer interface: output_image[px * 3 + ch]  (HWC)
    {
        std::vector<float> chw(static_cast<std::size_t>(HW) * 3u);
        std::memcpy(chw.data(), output_image,
                    static_cast<std::size_t>(HW) * 3u * sizeof(float));
        for (int px = 0; px < HW; ++px) {
            for (int ch = 0; ch < 3; ++ch) {
                output_image[static_cast<std::size_t>(px) * 3 + ch] =
                    chw[static_cast<std::size_t>(ch) * HW + px];
            }
        }
    }
```

Add `#include <vector>` if not already present (it is — line 37).

- [ ] **Step 2: Remove duplicate conversion from VulkanTrainer::step()**

In `harmonyos_3dgs/src/vulkan_trainer.cpp`, delete the CHW→HWC block at lines 303–329 (the `// 4b. Convert rasterizer output from CHW ...` block including the TODO comment). The conversion now happens inside `rasterize()`.

- [ ] **Step 3: Build and run training tests**

```bash
cd harmonyos_3dgs && cmake --build build
ctest --test-dir build --output-on-failure -R "TrainingStep|VkVsPyReference|Basketball"
```

Expected: all training tests still pass — the HWC output is now produced by `rasterize()` instead of `step()`, same result.

- [ ] **Step 4: Commit**

```bash
git add harmonyos_3dgs/src/vulkan/rasterizer_vulkan.cpp \
        harmonyos_3dgs/src/vulkan_trainer.cpp
git commit -m "fix: move CHW→HWC conversion into RasterizerVulkan::rasterize()

The Rasterizer interface contract is HWC (pixel-major). The Vulkan
rasterizer was returning raw CHW from the GPU, forcing VulkanTrainer
to do its own conversion. Move the conversion into the adapter so
all callers get consistent HWC output."
```

---

### Task 3: Create vk_render_main.cpp

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/vk_render_main.cpp`

- [ ] **Step 1: Write vk_render_main.cpp**

```cpp
// harmonyos_3dgs/src/vulkan/vk_render_main.cpp
//
// gs3d_vk_render — Vulkan forward render CLI for calibration.
// Usage: gs3d_vk_render <model.ply> <cameras.json> [cam_id]

#include "camera_utils.h"
#include "image_io.h"
#include "ply_loader.h"
#include "renderer.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage: %s <model.ply> <cameras.json> [cam_id]\n",
                     argv[0]);
        return 1;
    }

    const char* ply_path = argv[1];
    const char* cam_path = argv[2];
    int cam_id = (argc >= 4) ? std::atoi(argv[3]) : 0;

    // 1. Vulkan init
    VulkanContext ctx;
    if (!ctx.init()) {
        std::fprintf(stderr, "[FAIL] No Vulkan device with compute queue\n");
        return 1;
    }
    std::printf("[info] Vulkan device: %s\n", ctx.deviceName().c_str());

    // 2. Load model
    std::printf("Loading model: %s\n", ply_path);
    auto model = loadPly(ply_path);
    std::printf("  %d Gaussians, SH degree %d, filter_3D: %s\n",
                model.data.count, model.data.sh_degree,
                model.data.filter_3D ? "yes" : "no");

    // 3. Load camera
    Camera cam = loadCameraJson(cam_path, cam_id);

    // 4. Render config
    RenderConfig config{};
    const char* bg_env = std::getenv("BG_WHITE");
    float bg_val = (bg_env && bg_env[0] == '1') ? 1.0f : 0.0f;
    config.bg_color[0] = config.bg_color[1] = config.bg_color[2] = bg_val;
    config.sh_degree = model.data.sh_degree;
    config.eval_3D = false;
    config.antialiasing = false;
    std::printf("Config: eval_3D=%s, bg=%.0f, sh_degree=%d\n",
                config.eval_3D ? "ON" : "OFF", bg_val, config.sh_degree);

    // 5. Create Vulkan renderer
    size_t alloc_size = 512ULL * 1024 * 1024;
    if (model.data.count > 500000)
        alloc_size = 2ULL * 1024 * 1024 * 1024;

    auto renderer = std::make_unique<Renderer>(
        std::make_unique<PreprocessorVulkan>(ctx),
        std::make_unique<TileBinnerVulkan>(ctx),
        std::make_unique<SorterVulkan>(ctx),
        std::make_unique<RasterizerVulkan>(ctx),
        alloc_size);

    // 6. Render
    int W = cam.width, H = cam.height;
    std::vector<float> image(static_cast<size_t>(W) * H * 3, 0.0f);
    std::printf("Rendering %dx%d...\n", W, H);
    auto t0 = std::chrono::high_resolution_clock::now();
    renderer->render(model.data, cam, config, image.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Render time: %.1f ms\n", ms);

    // 7. Save
    const char* out_path = "output_vk.ppm";
    writePPM(out_path, image.data(), W, H);
    std::printf("Saved → %s\n", out_path);

    model.free();
    return 0;
}
```

- [ ] **Step 2: Build gs3d_vk_render**

```bash
cd harmonyos_3dgs && cmake --build build --target gs3d_vk_render
```

Expected: build succeeds (the CMake target at line 261 already declares this binary).

- [ ] **Step 3: Smoke test**

```bash
cd harmonyos_3dgs && ./build/gs3d_vk_render ../basket-aaa.ply cameras.json 0
```

Expected: renders to `output_vk.ppm`, prints timing, no crash. The PLY is at the repo root (`../basket-aaa.ply` relative to `harmonyos_3dgs/`).

If `basket-aaa.ply` is not available (e.g., in CI), use a tiny fixture:
```bash
./build/gs3d_vk_render tests/test_data/tiny_3gaussians.ply <path-to-cameras.json> 0
```

- [ ] **Step 4: Commit**

```bash
git add harmonyos_3dgs/src/vulkan/vk_render_main.cpp
git commit -m "feat: add gs3d_vk_render CLI for Vulkan forward rendering"
```

---

### Task 4: Create CPU-vs-Vulkan comparison test

**Files:**
- Create: `harmonyos_3dgs/tests/test_vk_vs_cpu_render.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt` (lines 266–292, gs3d_vk_tests sources; lines 303–306, compile definitions)

- [ ] **Step 1: Add test source and REPO_ROOT_DIR to CMakeLists.txt**

In `harmonyos_3dgs/CMakeLists.txt`, append `tests/test_vk_vs_cpu_render.cpp` to the `gs3d_vk_tests` source list (after line 292, `tests/test_vk_vs_py_reference.cpp`):

```cmake
                tests/test_vk_vs_py_reference.cpp
                tests/test_vk_vs_cpu_render.cpp)
```

Add `REPO_ROOT_DIR` to the compile definitions block (after line 306, `TEST_DATA_DIR=...`):

```cmake
            target_compile_definitions(gs3d_vk_tests PRIVATE
                ADD_ONE_SPV_PATH="${_spv_add_one}"
                SHADER_DIR="${_vk_shader_dir}"
                TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data"
                REPO_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
```

- [ ] **Step 2: Write the test**

```cpp
// harmonyos_3dgs/tests/test_vk_vs_cpu_render.cpp
//
// CPU-vs-Vulkan full-frame render comparison.
// Loads basket-aaa.ply + cameras.json cam 0, renders with both backends
// under eval_3D=false, asserts PSNR > 60 dB.

#include <gtest/gtest.h>

#include "camera_utils.h"
#include "ply_loader.h"
#include "renderer.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

// Compute PSNR between two float images (range [0,1]).
// Returns infinity if images are identical (MSE=0).
static double computePSNR(const float* a, const float* b, size_t count,
                          float& max_err, size_t& max_err_idx) {
    double mse = 0.0;
    max_err = 0.0f;
    max_err_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        float diff = a[i] - b[i];
        float abs_diff = std::fabs(diff);
        if (abs_diff > max_err) {
            max_err = abs_diff;
            max_err_idx = i;
        }
        mse += static_cast<double>(diff) * diff;
    }
    mse /= static_cast<double>(count);
    if (mse == 0.0) return 999.0;  // identical
    return 10.0 * std::log10(1.0 / mse);
}

TEST(VkVsCpuRender, FullFramePSNR) {
    // --- Skip conditions ---
    const std::string ply_path =
        std::string(REPO_ROOT_DIR) + "/basket-aaa.ply";
    const std::string cam_path =
        std::string(REPO_ROOT_DIR) + "/harmonyos_3dgs/cameras.json";

    {
        std::ifstream f(ply_path);
        if (!f.good())
            GTEST_SKIP() << "basket-aaa.ply not found at " << ply_path;
    }
    {
        std::ifstream f(cam_path);
        if (!f.good())
            GTEST_SKIP() << "cameras.json not found at " << cam_path;
    }

    VulkanContext ctx;
    if (!ctx.init())
        GTEST_SKIP() << "No Vulkan device available";

    // --- Load model + camera ---
    auto model = loadPly(ply_path.c_str());
    ASSERT_GT(model.data.count, 0);

    Camera cam = loadCameraJson(cam_path.c_str(), 0);
    const int W = cam.width;
    const int H = cam.height;
    ASSERT_GT(W, 0);
    ASSERT_GT(H, 0);
    const size_t num_pixels = static_cast<size_t>(W) * H * 3;

    RenderConfig config{};
    config.eval_3D = false;
    config.antialiasing = false;
    config.sh_degree = model.data.sh_degree;
    // black background
    config.bg_color[0] = config.bg_color[1] = config.bg_color[2] = 0.0f;

    size_t alloc_size = 512ULL * 1024 * 1024;
    if (model.data.count > 500000)
        alloc_size = 2ULL * 1024 * 1024 * 1024;

    // --- CPU render ---
    std::vector<float> cpu_image(num_pixels, 0.0f);
    {
        auto cpu_renderer = std::make_unique<Renderer>(
            std::make_unique<PreprocessorCPU>(),
            std::make_unique<TileBinnerCPU>(),
            std::make_unique<SorterCPU>(),
            std::make_unique<RasterizerCPU>(),
            alloc_size);
        cpu_renderer->render(model.data, cam, config, cpu_image.data());
    }

    // --- Vulkan render ---
    std::vector<float> vk_image(num_pixels, 0.0f);
    {
        auto vk_renderer = std::make_unique<Renderer>(
            std::make_unique<PreprocessorVulkan>(ctx),
            std::make_unique<TileBinnerVulkan>(ctx),
            std::make_unique<SorterVulkan>(ctx),
            std::make_unique<RasterizerVulkan>(ctx),
            alloc_size);
        vk_renderer->render(model.data, cam, config, vk_image.data());
    }

    // --- Compare ---
    float max_err = 0.0f;
    size_t max_err_idx = 0;
    double psnr = computePSNR(cpu_image.data(), vk_image.data(), num_pixels,
                               max_err, max_err_idx);

    size_t max_px = max_err_idx / 3;
    int max_ch = static_cast<int>(max_err_idx % 3);
    std::printf("  PSNR: %.2f dB\n", psnr);
    std::printf("  Max error: %.6f at pixel %zu channel %d "
                "(cpu=%.6f, vk=%.6f)\n",
                max_err, max_px, max_ch,
                cpu_image[max_err_idx], vk_image[max_err_idx]);

    EXPECT_GT(psnr, 60.0) << "PSNR too low — Vulkan output diverges from CPU";
    EXPECT_LT(max_err, 0.001f) << "Max per-pixel error too large";

    model.free();
}
```

- [ ] **Step 3: Build**

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build
```

Expected: build succeeds with the new test compiled into `gs3d_vk_tests`.

- [ ] **Step 4: Run the test**

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure -R "VkVsCpuRender"
```

Expected: `VkVsCpuRender.FullFramePSNR` passes with PSNR > 60 dB (or skips if no PLY/GPU).

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure
```

Expected: all 243+ existing tests still pass + new test passes.

- [ ] **Step 6: Commit**

```bash
git add harmonyos_3dgs/tests/test_vk_vs_cpu_render.cpp harmonyos_3dgs/CMakeLists.txt
git commit -m "test: add CPU-vs-Vulkan full-frame render comparison (PSNR > 60 dB)"
```

---

### Task 5: Final validation

- [ ] **Step 1: Visual comparison**

Run both renderers on the same input and visually compare:

```bash
cd harmonyos_3dgs
# CPU render
./build/gs3d_render ../basket-aaa.ply cameras.json 0
# Vulkan render
./build/gs3d_vk_render ../basket-aaa.ply cameras.json 0
# Both produce PPM files — compare visually or with ImageMagick:
# compare output.ppm output_vk.ppm -compose src diff.ppm
```

- [ ] **Step 2: Full test suite**

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure
```

Expected: all tests pass (243+ existing + 1 new VkVsCpuRender).

- [ ] **Step 3: Commit any final fixups if needed**

If any issues surfaced during validation, fix and commit individually.
