// test_vk_vs_cuda_basketball.cpp — Full-scene VK eval_3D vs CUDA golden.
//
// Loads basket-aaa.ply (400000 Gaussians, SH degree 3) @ cam 0 (720x960),
// renders via the Vulkan Renderer path (same as vk_render_main), loads the
// CUDA golden at tests/golden/basketball/cam0/cuda_image.raw, computes
// PSNR + per-channel PSNR + max/mean/p99 abs error, writes a spatial diff
// heatmap PPM to CMAKE_BINARY_DIR, and asserts PSNR >= kBaselinePSNR.
//
// Baseline starts at 5.0 dB (sentinel) and is overwritten in Task 5 after
// measurement. The canonical golden (aaa.json features enabled) differs from
// the prior VK-aligned reference by ~25 dB, so the initial VK-vs-canonical PSNR
// is expected to be substantially lower than the historical 32 dB against the
// old VK-aligned reference. Target: >=60 dB.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "camera_utils.h"
#include "image_io.h"
#include "ply_loader.h"
#include "renderer.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

namespace {

// G0 baseline locked at 24.8 dB (floor(25.308*10)/10 - 0.5).
// Measured P0=25.308 dB (3-run reproducible, delta=0.000 dB) on 2026-04-21.
// Task 9 (proper_ewa_scaling=true): P1=36.864 dB (+11.556 dB) on 2026-04-21.
// G1 baseline locked at 36.3 dB (floor(36.864*10)/10 - 0.5).
// Task 12 (per-pixel k-buffer HEAD_WINDOW=4): P2=40.266 dB (+3.402 dB) on 2026-04-21.
// G2 baseline locked at 39.7 dB (floor(40.266*10)/10 - 0.5).
// Task 11 (tile_based_culling INVALID sentinel for eval_3D): P3=40.256 dB (+3.392 dB vs P1) on 2026-04-21.
// G3 baseline locked at 39.7 dB (floor(40.256*10)/10 - 0.5 = 39.75 → 39.7; kept same as G2).
// Task 12b (hierarchical sub-tile TAIL re-sort, simplified): P4=42.623 dB (+2.367 dB) on 2026-04-21.
// G4 baseline locked at 42.1 dB (floor(42.623*10)/10 - 0.5).
// DO NOT lower this without documenting why.
constexpr float kBaselinePSNR = 42.1f;

constexpr int kW = 720;
constexpr int kH = 960;

const std::string kPlyPath =
    "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply";
const std::string kCamPath =
    "/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json";
const std::string kGoldenPath =
    std::string(TEST_DATA_DIR) + "/golden/basketball/cam0/cuda_image.raw";

struct Metrics {
    float psnr;
    float psnr_r, psnr_g, psnr_b;
    float max_abs, mean_abs, p99_abs;
    int num_bad;  // |diff| > 1e-3
};

float psnr_from_mse(float mse) {
    if (mse <= 0.0f) return 100.0f;
    return -10.0f * std::log10(mse);
}

Metrics compute_metrics_hwc(const std::vector<float>& vk,
                            const std::vector<float>& cuda) {
    EXPECT_EQ(vk.size(), cuda.size());
    const size_t N = vk.size();
    double sse = 0.0, sse_r = 0.0, sse_g = 0.0, sse_b = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    int num_bad = 0;
    std::vector<float> abs_errs;
    abs_errs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        float d = vk[i] - cuda[i];
        float ad = std::fabs(d);
        float d2 = d * d;
        sse += d2;
        sum_abs += ad;
        if (ad > max_abs) max_abs = ad;
        if (ad > 1e-3f) ++num_bad;
        abs_errs.push_back(ad);
        int ch = static_cast<int>(i % 3);
        if (ch == 0) sse_r += d2;
        else if (ch == 1) sse_g += d2;
        else sse_b += d2;
    }
    std::sort(abs_errs.begin(), abs_errs.end());
    float p99 = abs_errs[static_cast<size_t>(abs_errs.size() * 0.99)];
    const size_t per_ch = N / 3;
    Metrics m{};
    m.psnr     = psnr_from_mse(static_cast<float>(sse / N));
    m.psnr_r   = psnr_from_mse(static_cast<float>(sse_r / per_ch));
    m.psnr_g   = psnr_from_mse(static_cast<float>(sse_g / per_ch));
    m.psnr_b   = psnr_from_mse(static_cast<float>(sse_b / per_ch));
    m.max_abs  = max_abs;
    m.mean_abs = static_cast<float>(sum_abs / N);
    m.p99_abs  = p99;
    m.num_bad  = num_bad;
    return m;
}

void write_diff_heatmap(const std::vector<float>& vk,
                        const std::vector<float>& cuda,
                        const std::string& out_ppm) {
    const float scale = 1.0f / 0.05f;
    std::vector<float> heat(static_cast<size_t>(kW) * kH * 3);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            size_t base = (static_cast<size_t>(y) * kW + x) * 3;
            float dmax = 0.0f;
            for (int c = 0; c < 3; ++c) {
                dmax = std::max(dmax, std::fabs(vk[base + c] - cuda[base + c]));
            }
            float v = std::clamp(dmax * scale, 0.0f, 1.0f);
            heat[base + 0] = v;
            heat[base + 1] = v;
            heat[base + 2] = v;
        }
    }
    writePPM(out_ppm.c_str(), heat.data(), kW, kH);
}

}  // namespace

TEST(VkVsCudaBasketball, Cam0_PsnrAtLeastBaseline) {
    if (!std::filesystem::exists(kPlyPath)) {
        GTEST_SKIP() << "basket-aaa.ply not available at " << kPlyPath;
    }
    if (!std::filesystem::exists(kCamPath)) {
        GTEST_SKIP() << "cameras.json not available at " << kCamPath;
    }
    if (!std::filesystem::exists(kGoldenPath)) {
        GTEST_SKIP() << "CUDA golden not generated — run "
                        "tools/render_single.py --raw-out ... --npy-out ... "
                        "--hash-out ... Missing: "
                     << kGoldenPath;
    }

    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device.";
    }

    auto model = loadPly(kPlyPath.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width, kW);
    ASSERT_EQ(cam.height, kH);

    RenderConfig cfg{};
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.sh_degree = model.data.sh_degree;
    cfg.eval_3D = true;
    cfg.antialiasing = false;
    cfg.training = true;

    const size_t alloc = 4ULL * 1024 * 1024 * 1024;
    Renderer renderer(
        std::make_unique<PreprocessorVulkan>(ctx, /*eval_3D=*/true),
        std::make_unique<TileBinnerVulkan>(ctx),
        std::make_unique<SorterVulkan>(ctx),
        std::make_unique<RasterizerVulkan>(ctx, /*eval_3D=*/true),
        alloc);

    std::vector<float> vk_hwc(static_cast<size_t>(kW) * kH * 3, 0.0f);
    renderer.render(model.data, cam, cfg, vk_hwc.data());
    model.free();

    std::vector<float> golden_hwc(vk_hwc.size());
    std::ifstream in(kGoldenPath, std::ios::binary);
    ASSERT_TRUE(in.good()) << "Cannot open golden: " << kGoldenPath;
    in.read(reinterpret_cast<char*>(golden_hwc.data()),
            static_cast<std::streamsize>(golden_hwc.size() * sizeof(float)));
    ASSERT_EQ(in.gcount(),
              static_cast<std::streamsize>(golden_hwc.size() * sizeof(float)))
        << "Golden file size mismatch";

    Metrics m = compute_metrics_hwc(vk_hwc, golden_hwc);

    std::string diff_ppm =
        std::string(CMAKE_BINARY_DIR) + "/vk_cuda_diff_cam0.ppm";
    write_diff_heatmap(vk_hwc, golden_hwc, diff_ppm);

    std::printf("[VkVsCudaBasketball] PSNR=%.3f dB (R=%.2f G=%.2f B=%.2f)\n",
                m.psnr, m.psnr_r, m.psnr_g, m.psnr_b);
    std::printf("[VkVsCudaBasketball] max_abs=%.5f mean_abs=%.5f "
                "p99_abs=%.5f bad_pixels(>1e-3)=%d\n",
                m.max_abs, m.mean_abs, m.p99_abs, m.num_bad);
    std::printf("[VkVsCudaBasketball] diff heatmap -> %s\n", diff_ppm.c_str());

    EXPECT_GE(m.psnr, kBaselinePSNR)
        << "VK vs CUDA PSNR regressed below baseline " << kBaselinePSNR
        << " dB. Inspect " << diff_ppm;
}
