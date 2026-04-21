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
// Returns 999.0 if images are identical (MSE=0).
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
    if (mse == 0.0) return 999.0;
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
    // PreprocessorVulkan is hardcoded to spec_training=1 (training mode, no
    // upper SH color clamp). Setting training=true on the CPU side matches
    // this behaviour, eliminating the systematic RGB divergence that otherwise
    // dominates the PSNR (unclamped SH values can exceed 1.0 by several units).
    config.training = true;
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

    // With sort-critical computations marked 'precise' in preprocess.comp (no
    // FMA) and training=true matching Vulkan's spec_training=1, the only
    // remaining error is FP accumulation noise in the rasterizer's alpha-blend
    // loop and minor FMA differences in SH evaluation / cov2D.  On basket-aaa
    // (400K Gaussians, 720x960) this gives ~100 dB PSNR.
    // Threshold: PSNR > 60 dB is near-bit-exact; 40 dB is the absolute floor.
    EXPECT_GT(psnr, 60.0) << "PSNR too low — Vulkan output diverges from CPU";
    EXPECT_LT(max_err, 0.01f) << "Max per-pixel error unreasonably large";

    model.free();
}
