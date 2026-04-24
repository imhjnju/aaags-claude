// test_vk_vs_cuda_basketball.cpp — Full-scene VK eval_3D vs CUDA golden (CHW layout).
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

Metrics compute_metrics_chw(const std::vector<float>& vk,
                            const std::vector<float>& cuda) {
    EXPECT_EQ(vk.size(), cuda.size());
    const size_t N = vk.size();
    const size_t per_ch = N / 3;
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
        // CHW: pixels 0..per_ch-1 are R, per_ch..2*per_ch-1 are G, 2*per_ch..3*per_ch-1 are B.
        int ch = static_cast<int>(i / per_ch);
        if (ch == 0) sse_r += d2;
        else if (ch == 1) sse_g += d2;
        else sse_b += d2;
    }
    std::sort(abs_errs.begin(), abs_errs.end());
    float p99 = abs_errs[static_cast<size_t>(abs_errs.size() * 0.99)];
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
    const size_t HW = static_cast<size_t>(kH) * kW;
    // writePPM expects HWC layout, so build heat in HWC for output.
    std::vector<float> heat(static_cast<size_t>(kW) * kH * 3);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            size_t pix = static_cast<size_t>(y) * kW + x;
            float dmax = 0.0f;
            for (int c = 0; c < 3; ++c) {
                dmax = std::max(dmax, std::fabs(vk[c * HW + pix] - cuda[c * HW + pix]));
            }
            float v = std::clamp(dmax * scale, 0.0f, 1.0f);
            size_t base = pix * 3;
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

    std::vector<float> vk_chw(static_cast<size_t>(kW) * kH * 3, 0.0f);
    renderer.render(model.data, cam, cfg, vk_chw.data());
    model.free();

    std::vector<float> golden_chw(vk_chw.size());
    std::ifstream in(kGoldenPath, std::ios::binary);
    ASSERT_TRUE(in.good()) << "Cannot open golden: " << kGoldenPath;
    in.read(reinterpret_cast<char*>(golden_chw.data()),
            static_cast<std::streamsize>(golden_chw.size() * sizeof(float)));
    ASSERT_EQ(in.gcount(),
              static_cast<std::streamsize>(golden_chw.size() * sizeof(float)))
        << "Golden file size mismatch";

    // Golden raw was saved as HWC by render_single.py --raw-out. Convert to CHW.
    {
        const int HW = kH * kW;
        std::vector<float> tmp(HW * 3);
        std::memcpy(tmp.data(), golden_chw.data(), HW * 3 * sizeof(float));
        for (int px = 0; px < HW; ++px) {
            for (int ch = 0; ch < 3; ++ch) {
                golden_chw[static_cast<size_t>(ch) * HW + px] = tmp[static_cast<size_t>(px) * 3 + ch];
            }
        }
    }

    // Dump VK raw output for external comparison
    {
        std::string vk_raw = std::string(CMAKE_BINARY_DIR) + "/vk_image.raw";
        std::ofstream vo(vk_raw, std::ios::binary);
        vo.write(reinterpret_cast<const char*>(vk_chw.data()),
                 static_cast<std::streamsize>(vk_chw.size() * sizeof(float)));
        std::printf("[VkVsCudaBasketball] VK raw -> %s\n", vk_raw.c_str());
    }

    Metrics m = compute_metrics_chw(vk_chw, golden_chw);

    std::string diff_ppm =
        std::string(CMAKE_BINARY_DIR) + "/vk_cuda_diff_cam0.ppm";
    write_diff_heatmap(vk_chw, golden_chw, diff_ppm);

    std::printf("[VkVsCudaBasketball] PSNR=%.3f dB (R=%.2f G=%.2f B=%.2f)\n",
                m.psnr, m.psnr_r, m.psnr_g, m.psnr_b);
    std::printf("[VkVsCudaBasketball] max_abs=%.5f mean_abs=%.5f "
                "p99_abs=%.5f bad_pixels(>1e-3)=%d\n",
                m.max_abs, m.mean_abs, m.p99_abs, m.num_bad);
    std::printf("[VkVsCudaBasketball] diff heatmap -> %s\n", diff_ppm.c_str());

    // --- Spatial region analysis ---
    // Split into foreground (cuda pixel > 0.01) vs background, and report
    // per-region PSNR + top-20 worst pixels.
    {
        const size_t HW = static_cast<size_t>(kH) * kW;
        double sse_fg = 0.0, sse_bg = 0.0;
        int n_fg = 0, n_bg = 0;
        int vk_brighter = 0, cuda_brighter = 0;
        struct WorstPixel { int x, y; float vk_r, vk_g, vk_b; float cu_r, cu_g, cu_b; float err; };
        std::vector<WorstPixel> worst;
        worst.reserve(static_cast<size_t>(kW) * kH);

        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                size_t pix = static_cast<size_t>(y) * kW + x;
                float vr = vk_chw[0 * HW + pix], vg = vk_chw[1 * HW + pix], vb = vk_chw[2 * HW + pix];
                float cr = golden_chw[0 * HW + pix], cg = golden_chw[1 * HW + pix], cb = golden_chw[2 * HW + pix];
                float dr = vr - cr, dg = vg - cg, db = vb - cb;
                float err = std::sqrt(dr*dr + dg*dg + db*db) / std::sqrt(3.0f);
                float cuda_lum = 0.2126f*cr + 0.7152f*cg + 0.0722f*cb;
                float vk_lum   = 0.2126f*vr + 0.7152f*vg + 0.0722f*vb;
                float d2 = dr*dr + dg*dg + db*db;
                if (cuda_lum > 0.01f) { sse_fg += d2; n_fg++; } else { sse_bg += d2; n_bg++; }
                if (vk_lum > cuda_lum + 0.001f) vk_brighter++;
                else if (cuda_lum > vk_lum + 0.001f) cuda_brighter++;
                worst.push_back({x, y, vr, vg, vb, cr, cg, cb, err});
            }
        }
        std::sort(worst.begin(), worst.end(),
                  [](const WorstPixel& a, const WorstPixel& b) { return a.err > b.err; });

        float psnr_fg = n_fg > 0 ? psnr_from_mse(static_cast<float>(sse_fg / (n_fg * 3))) : 100.0f;
        float psnr_bg = n_bg > 0 ? psnr_from_mse(static_cast<float>(sse_bg / (n_bg * 3))) : 100.0f;
        std::printf("[VkVsCudaBasketball] FG pixels=%d PSNR=%.2f dB | BG pixels=%d PSNR=%.2f dB\n",
                    n_fg, psnr_fg, n_bg, psnr_bg);
        std::printf("[VkVsCudaBasketball] VK_brighter=%d CUDA_brighter=%d equal=%d\n",
                    vk_brighter, cuda_brighter, kW*kH - vk_brighter - cuda_brighter);

        std::printf("[VkVsCudaBasketball] Top-20 worst pixels (VK vs CUDA [R,G,B] RMSE):\n");
        for (int i = 0; i < 20 && i < (int)worst.size(); ++i) {
            const auto& p = worst[i];
            std::printf("  #%2d (%4d,%4d) err=%.4f  VK=[%.3f,%.3f,%.3f] CU=[%.3f,%.3f,%.3f]\n",
                        i+1, p.x, p.y, p.err, p.vk_r, p.vk_g, p.vk_b, p.cu_r, p.cu_g, p.cu_b);
        }

        // Histogram: how many pixels at each error level
        int bins[10] = {};
        for (const auto& p : worst) {
            int b = std::min(9, static_cast<int>(p.err * 20.0f));
            bins[b]++;
        }
        std::printf("[VkVsCudaBasketball] Error histogram (bin width=0.05):\n");
        for (int i = 0; i < 10; ++i) {
            std::printf("  [%.2f,%.2f): %d pixels\n", i*0.05f, (i+1)*0.05f, bins[i]);
        }
    }

    EXPECT_GE(m.psnr, kBaselinePSNR)
        << "VK vs CUDA PSNR regressed below baseline " << kBaselinePSNR
        << " dB. Inspect " << diff_ppm;
}
