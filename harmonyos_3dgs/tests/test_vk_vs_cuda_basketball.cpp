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

#include "test_data_paths.h"

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
// S10 (eval_3D near-plane cull removed — preprocess.comp:914, preprocessor_cpu.cpp:65):
//   P5=54.838 dB (+12.215 dB) on 2026-04-24. ROI (bottom-left table) PSNR
//   jumped 34.64 -> 57.95 dB (+23.31 dB); VK-brighter bias in ROI collapsed
//   from 99.3% to 5.0%. Root cause: `if (p_view.z <= 0.2) return` in VK
//   preprocessor fired unconditionally and dropped 3 near-camera Gaussians
//   (gid 67596, 347926, 61986) that CUDA renders because aaa.json leaves
//   splatting_settings.near_clipping=false (forward.cu:136 gates on that).
// G5 baseline locked at 54.3 dB (floor(54.838*10)/10 - 0.5).
// DO NOT lower this without documenting why.
constexpr float kBaselinePSNR = 54.3f;

constexpr int kW = 720;
constexpr int kH = 960;

// Resolved at first use via test_data::find_basket_aaa_ply() — searches the
// current worktree, the master root, and known sibling worktrees so the test
// works regardless of which worktree the developer has the .ply file in.
// (The 100 MB file can't be git-tracked.)
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
    if (vk.size() != cuda.size()) return {};
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
    std::string kPlyPath;
    if (!test_data::resolve_basket_aaa_ply(kPlyPath)) {
        GTEST_SKIP() << "basket-aaa.ply not found in any known location "
                        "(set $BASKET_AAA_PLY to override)";
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
        std::make_unique<PreprocessorVulkan>(ctx, /*eval_3D=*/true, /*proper_ewa=*/true),
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

// Diagnostic test: render the table-subset PLY (only Gaussians projecting into
// the bottom-left 160x240 ROI) and compare against a CUDA golden rendered from
// the same subset. This isolates VK vs CUDA divergence contributed by Gaussians
// inside the ROI vs leakage from Gaussians elsewhere. No PSNR assertion —
// always SUCCEED() after printing so ctest stays green; this is a data
// gathering harness, not a gate.
TEST(VkVsCudaBasketball, Cam0_TableSubset) {
    // Subset artifacts are produced by tools/extract_gaussians_by_roi.py
    // + tools/render_single.py and live under tools/out/ of the current worktree.
    // Use REPO_ROOT_DIR so the test follows the developer between worktrees
    // (was hardcoded to white-table, breaking other worktrees).
    const std::string kSubsetPlyPath =
        std::string(REPO_ROOT_DIR) + "/tools/out/basket-aaa-table.ply";
    const std::string kSubsetGoldenPath =
        std::string(REPO_ROOT_DIR) + "/tools/out/basket-aaa-table_cuda.raw";

    // ROI: bottom-left 160x240 window in image space.
    constexpr struct { int x_min, y_min, x_max, y_max; } kRoi{0, 720, 160, 960};

    if (!std::filesystem::exists(kSubsetPlyPath)) {
        GTEST_SKIP() << "Subset PLY not available at " << kSubsetPlyPath
                     << " — generate via tools/extract_table_subset.py";
    }
    if (!std::filesystem::exists(kCamPath)) {
        GTEST_SKIP() << "cameras.json not available at " << kCamPath;
    }
    if (!std::filesystem::exists(kSubsetGoldenPath)) {
        GTEST_SKIP() << "Subset CUDA golden not generated — run "
                        "tools/render_single.py --ply "
                     << kSubsetPlyPath << " --raw-out " << kSubsetGoldenPath;
    }

    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device.";
    }

    auto model = loadPly(kSubsetPlyPath.c_str());
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
        std::make_unique<PreprocessorVulkan>(ctx, /*eval_3D=*/true, /*proper_ewa=*/true),
        std::make_unique<TileBinnerVulkan>(ctx),
        std::make_unique<SorterVulkan>(ctx),
        std::make_unique<RasterizerVulkan>(ctx, /*eval_3D=*/true),
        alloc);

    std::vector<float> vk_chw(static_cast<size_t>(kW) * kH * 3, 0.0f);
    renderer.render(model.data, cam, cfg, vk_chw.data());
    model.free();

    std::vector<float> golden_chw(vk_chw.size());
    std::ifstream in(kSubsetGoldenPath, std::ios::binary);
    ASSERT_TRUE(in.good()) << "Cannot open subset golden: " << kSubsetGoldenPath;
    in.read(reinterpret_cast<char*>(golden_chw.data()),
            static_cast<std::streamsize>(golden_chw.size() * sizeof(float)));
    ASSERT_EQ(in.gcount(),
              static_cast<std::streamsize>(golden_chw.size() * sizeof(float)))
        << "Subset golden file size mismatch";

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
        std::string vk_raw = std::string(CMAKE_BINARY_DIR) + "/vk_image_subset.raw";
        std::ofstream vo(vk_raw, std::ios::binary);
        vo.write(reinterpret_cast<const char*>(vk_chw.data()),
                 static_cast<std::streamsize>(vk_chw.size() * sizeof(float)));
        std::printf("[TableSubset] VK raw -> %s\n", vk_raw.c_str());
    }

    Metrics m = compute_metrics_chw(vk_chw, golden_chw);

    std::string diff_ppm =
        std::string(CMAKE_BINARY_DIR) + "/vk_cuda_diff_cam0_subset.ppm";
    write_diff_heatmap(vk_chw, golden_chw, diff_ppm);

    std::printf("[TableSubset] PSNR=%.3f dB (R=%.2f G=%.2f B=%.2f)\n",
                m.psnr, m.psnr_r, m.psnr_g, m.psnr_b);
    std::printf("[TableSubset] max_abs=%.5f mean_abs=%.5f "
                "p99_abs=%.5f bad_pixels(>1e-3)=%d\n",
                m.max_abs, m.mean_abs, m.p99_abs, m.num_bad);
    std::printf("[TableSubset] diff heatmap -> %s\n", diff_ppm.c_str());

    // --- Full-image diagnostics (mirrors the baseline test's block) ---
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
        std::printf("[TableSubset FULL] FG pixels=%d PSNR=%.2f dB | BG pixels=%d PSNR=%.2f dB\n",
                    n_fg, psnr_fg, n_bg, psnr_bg);
        std::printf("[TableSubset FULL] VK_brighter=%d CUDA_brighter=%d equal=%d\n",
                    vk_brighter, cuda_brighter, kW*kH - vk_brighter - cuda_brighter);

        std::printf("[TableSubset FULL] Top-20 worst pixels (VK vs CUDA [R,G,B] RMSE):\n");
        for (int i = 0; i < 20 && i < (int)worst.size(); ++i) {
            const auto& p = worst[i];
            std::printf("  #%2d (%4d,%4d) err=%.4f  VK=[%.3f,%.3f,%.3f] CU=[%.3f,%.3f,%.3f]\n",
                        i+1, p.x, p.y, p.err, p.vk_r, p.vk_g, p.vk_b, p.cu_r, p.cu_g, p.cu_b);
        }

        int bins[10] = {};
        for (const auto& p : worst) {
            int b = std::min(9, static_cast<int>(p.err * 20.0f));
            bins[b]++;
        }
        std::printf("[TableSubset FULL] Error histogram (bin width=0.05):\n");
        for (int i = 0; i < 10; ++i) {
            std::printf("  [%.2f,%.2f): %d pixels\n", i*0.05f, (i+1)*0.05f, bins[i]);
        }
    }

    // --- ROI-restricted diagnostics (x in [x_min,x_max), y in [y_min,y_max)) ---
    {
        const size_t HW = static_cast<size_t>(kH) * kW;
        double sse_roi = 0.0;
        int n_roi = 0;
        int vk_brighter = 0, cuda_brighter = 0, equal = 0;
        struct WorstPixel { int x, y; float vk_r, vk_g, vk_b; float cu_r, cu_g, cu_b; float err; };
        std::vector<WorstPixel> worst;
        const int roi_w = kRoi.x_max - kRoi.x_min;
        const int roi_h = kRoi.y_max - kRoi.y_min;
        worst.reserve(static_cast<size_t>(roi_w) * roi_h);

        for (int y = kRoi.y_min; y < kRoi.y_max; ++y) {
            for (int x = kRoi.x_min; x < kRoi.x_max; ++x) {
                size_t pix = static_cast<size_t>(y) * kW + x;
                float vr = vk_chw[0 * HW + pix], vg = vk_chw[1 * HW + pix], vb = vk_chw[2 * HW + pix];
                float cr = golden_chw[0 * HW + pix], cg = golden_chw[1 * HW + pix], cb = golden_chw[2 * HW + pix];
                float dr = vr - cr, dg = vg - cg, db = vb - cb;
                float err = std::sqrt(dr*dr + dg*dg + db*db) / std::sqrt(3.0f);
                float cuda_lum = 0.2126f*cr + 0.7152f*cg + 0.0722f*cb;
                float vk_lum   = 0.2126f*vr + 0.7152f*vg + 0.0722f*vb;
                float d2 = dr*dr + dg*dg + db*db;
                sse_roi += d2;
                n_roi++;
                if (vk_lum > cuda_lum + 0.001f) vk_brighter++;
                else if (cuda_lum > vk_lum + 0.001f) cuda_brighter++;
                else equal++;
                worst.push_back({x, y, vr, vg, vb, cr, cg, cb, err});
            }
        }
        std::sort(worst.begin(), worst.end(),
                  [](const WorstPixel& a, const WorstPixel& b) { return a.err > b.err; });

        float psnr_roi = n_roi > 0 ? psnr_from_mse(static_cast<float>(sse_roi / (n_roi * 3))) : 100.0f;
        std::printf("[TableSubset ROI] ROI=x[%d,%d) y[%d,%d) pixels=%d PSNR=%.2f dB\n",
                    kRoi.x_min, kRoi.x_max, kRoi.y_min, kRoi.y_max, n_roi, psnr_roi);
        std::printf("[TableSubset ROI] VK_brighter=%d CUDA_brighter=%d equal=%d\n",
                    vk_brighter, cuda_brighter, equal);

        std::printf("[TableSubset ROI] Top-20 worst pixels (VK vs CUDA [R,G,B] RMSE):\n");
        for (int i = 0; i < 20 && i < (int)worst.size(); ++i) {
            const auto& p = worst[i];
            std::printf("  #%2d (%4d,%4d) err=%.4f  VK=[%.3f,%.3f,%.3f] CU=[%.3f,%.3f,%.3f]\n",
                        i+1, p.x, p.y, p.err, p.vk_r, p.vk_g, p.vk_b, p.cu_r, p.cu_g, p.cu_b);
        }

        int bins[10] = {};
        for (const auto& p : worst) {
            int b = std::min(9, static_cast<int>(p.err * 20.0f));
            bins[b]++;
        }
        std::printf("[TableSubset ROI] Error histogram (bin width=0.05):\n");
        for (int i = 0; i < 10; ++i) {
            std::printf("  [%.2f,%.2f): %d pixels\n", i*0.05f, (i+1)*0.05f, bins[i]);
        }
    }

    // Intentionally no PSNR assertion — this is a diagnostic harness for the
    // VK-vs-CUDA parity investigation, not a regression gate. Keep ctest green
    // so reports surface via stdout (search for "[TableSubset ...]").
    SUCCEED();
}

// Diagnostic test: dump VK's per-pixel n_contrib buffer (how many Gaussians
// were blended into each pixel's accumulator) for basket-aaa.ply / cam0.
// Bypasses Renderer to call the 4 pipeline stages manually so we can thread a
// ForwardCache through rasterize() and capture T_final + n_contrib. Writes
// vk_n_contrib_cam0.raw (int32 little-endian, kW*kH elements) next to
// vk_image.raw in CMAKE_BINARY_DIR so it can be diffed against CUDA later.
TEST(VkVsCudaBasketball, Cam0_NContribDump) {
    std::string kPlyPath;
    if (!test_data::resolve_basket_aaa_ply(kPlyPath)) {
        GTEST_SKIP() << "basket-aaa.ply not found in any known location "
                        "(set $BASKET_AAA_PLY to override)";
    }
    if (!std::filesystem::exists(kCamPath)) {
        GTEST_SKIP() << "cameras.json not available at " << kCamPath;
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

    // Build the 4 pipeline stages as local unique_ptrs so we can keep a typed
    // handle on the rasterizer (RasterizerVulkan*) for potential
    // download_cache() fallback. The layer-1 rasterize() path already
    // auto-populates cache->T_final / cache->n_contrib when passed a non-null
    // cache with non-null CPU pointers (see rasterizer_vulkan.cpp:269-281).
    // basketball cam0 golden was regenerated with aaa.json features
    // (proper_ewa_scaling=True) per commit 605bce9; matching here ensures
    // the spec constant drives the shader to perform dilation. Same
    // precedent as test_preprocess_pass_vk.cpp:148 — wiring must mirror
    // the golden's actual generation config. Applies to all 3 PreprocessorVulkan
    // call sites in this file.
    auto preprocessor = std::make_unique<PreprocessorVulkan>(ctx, /*eval_3D=*/true, /*proper_ewa=*/true);
    auto tile_binner  = std::make_unique<TileBinnerVulkan>(ctx);
    auto sorter       = std::make_unique<SorterVulkan>(ctx);
    auto rasterizer   = std::make_unique<RasterizerVulkan>(ctx, /*eval_3D=*/true);

    FrameAllocator allocator(4ULL * 1024 * 1024 * 1024);

    std::vector<float> vk_chw(static_cast<size_t>(kW) * kH * 3, 0.0f);
    std::vector<int>   n_contrib(static_cast<size_t>(kW) * kH, 0);
    std::vector<float> T_final(static_cast<size_t>(kW) * kH, 1.0f);

    ForwardCache cache{};
    cache.T_final   = T_final.data();
    cache.n_contrib = n_contrib.data();
    // Other ForwardCache fields (cov2D, cov2D_det, cov3D, p_view, p_hom_w,
    // pre, bin) are backward-only and not read by the rasterizer.

    auto pre = preprocessor->process(model.data, cam, cfg, allocator);
    // Dump per-Gaussian preprocess outputs for VK-vs-CUDA diff (A+B).
    // rgb[N*3] (post-SH, post-activations), conic(a,b,c)[N*3], opacity_2d[N] (post-dilation),
    // means2D[N*2]. CUDA equivalents come from materialize_dump's preprocess_* tensors.
    {
        const size_t N = model.data.count;
        std::string base = std::string(CMAKE_BINARY_DIR) + "/vk_pre_";
        auto dump = [&](const std::string& name, const void* data, size_t bytes) {
            std::string p = base + name + ".raw";
            std::ofstream o(p, std::ios::binary);
            o.write(reinterpret_cast<const char*>(data), bytes);
            std::printf("[NContribDump PRE] %s -> %s (%zu bytes)\n", name.c_str(), p.c_str(), bytes);
        };
        dump("rgb", pre.rgb, N * 3 * sizeof(float));
        dump("conics", pre.conics, N * 3 * sizeof(float));
        dump("opacities_2d", pre.opacities_2d, N * sizeof(float));
        dump("means2D", pre.means2D, N * 2 * sizeof(float));
        dump("depths", pre.depths, N * sizeof(float));
        if (pre.gauss2screen) {
            dump("gauss2screen", pre.gauss2screen, N * 16 * sizeof(float));
        }
    }
    auto bin = tile_binner->bin(pre, model.data.count, cam, cfg, allocator);
    if (bin.total_pairs > 0) {
        sorter->sort(bin, allocator);
    }
    rasterizer->rasterize(pre, bin, cam, cfg, vk_chw.data(),
                          /*output_depth=*/nullptr, &cache, &allocator);

    // Dump sort outputs for per-tile Gaussian sequence comparison with CUDA.
    // bin.values_sorted [total_pairs] uint32 : Gaussian id per (tile,gaussian) pair, sorted
    // bin.keys_sorted   [total_pairs] uint64 : (tile_id << 32) | depth_bits
    // bin.tile_ranges   [num_tiles*2] uint32 : start,end index into values_sorted per tile
    {
        std::string base = std::string(CMAKE_BINARY_DIR) + "/vk_bin_";
        auto dump = [&](const std::string& name, const void* d, size_t bytes) {
            std::string p = base + name + ".raw";
            std::ofstream o(p, std::ios::binary);
            o.write(reinterpret_cast<const char*>(d), bytes);
            std::printf("[NContribDump BIN] %s -> %s (%zu bytes)\n",
                        name.c_str(), p.c_str(), bytes);
        };
        if (bin.values_sorted)
            dump("values_sorted", bin.values_sorted,
                 static_cast<size_t>(bin.total_pairs) * sizeof(uint32_t));
        if (bin.keys_sorted)
            dump("keys_sorted", bin.keys_sorted,
                 static_cast<size_t>(bin.total_pairs) * sizeof(uint64_t));
        if (bin.tile_ranges)
            dump("tile_ranges", bin.tile_ranges,
                 static_cast<size_t>(bin.num_tiles) * 2u * sizeof(uint32_t));
        std::printf("[NContribDump BIN] total_pairs=%d num_tiles=%d\n",
                    bin.total_pairs, bin.num_tiles);
    }
    model.free();

    // Dump raw outputs for external cross-check.
    {
        std::string nc_raw =
            std::string(CMAKE_BINARY_DIR) + "/vk_n_contrib_cam0.raw";
        std::ofstream nco(nc_raw, std::ios::binary);
        nco.write(reinterpret_cast<const char*>(n_contrib.data()),
                  static_cast<std::streamsize>(n_contrib.size() * sizeof(int32_t)));
        std::printf("[NContribDump] n_contrib raw -> %s (int32 LE, %zu elements)\n",
                    nc_raw.c_str(), n_contrib.size());

        std::string tf_raw = std::string(CMAKE_BINARY_DIR) + "/vk_T_final_cam0.raw";
        std::ofstream tfo(tf_raw, std::ios::binary);
        tfo.write(reinterpret_cast<const char*>(T_final.data()),
                  static_cast<std::streamsize>(T_final.size() * sizeof(float)));
        std::printf("[NContribDump] T_final raw -> %s (float32 LE, %zu elements)\n",
                    tf_raw.c_str(), T_final.size());

        std::string vk_raw = std::string(CMAKE_BINARY_DIR) + "/vk_image.raw";
        std::ofstream vo(vk_raw, std::ios::binary);
        vo.write(reinterpret_cast<const char*>(vk_chw.data()),
                 static_cast<std::streamsize>(vk_chw.size() * sizeof(float)));
        std::printf("[NContribDump] VK image raw -> %s\n", vk_raw.c_str());

        if (cache.replay_order_offsets && cache.replay_order_gids) {
            std::string replay_offsets_raw = std::string(CMAKE_BINARY_DIR) + "/vk_replay_order_offsets_cam0.raw";
            std::ofstream roo(replay_offsets_raw, std::ios::binary);
            roo.write(reinterpret_cast<const char*>(cache.replay_order_offsets),
                      static_cast<std::streamsize>((n_contrib.size() + 1u) * sizeof(uint32_t)));
            std::printf("[NContribDump] replay offsets raw -> %s (%zu uint32)\n",
                        replay_offsets_raw.c_str(), n_contrib.size() + 1u);

            std::string replay_gids_raw = std::string(CMAKE_BINARY_DIR) + "/vk_replay_order_gids_cam0.raw";
            std::ofstream rgo(replay_gids_raw, std::ios::binary);
            rgo.write(reinterpret_cast<const char*>(cache.replay_order_gids),
                      static_cast<std::streamsize>(cache.replay_order_count * sizeof(uint32_t)));
            std::printf("[NContribDump] replay gids raw -> %s (%zu uint32)\n",
                        replay_gids_raw.c_str(), cache.replay_order_count);

            constexpr int kProbeX = 453;
            constexpr int kProbeY = 52;
            const size_t probe_px = static_cast<size_t>(kProbeY) * kW + kProbeX;
            const uint32_t begin = cache.replay_order_offsets[probe_px];
            const uint32_t end = cache.replay_order_offsets[probe_px + 1u];
            std::printf("[NContribDump] probe (%d,%d) replay count=%u gids:",
                        kProbeX, kProbeY, end - begin);
            for (uint32_t i = begin; i < end; ++i) {
                std::printf(" %u", cache.replay_order_gids[i]);
            }
            std::printf("\n");
        }
    }

    // ROI ≡ x in [0,160), y in [720,960). 160*240 = 38400 pixels.
    constexpr struct { int x_min, y_min, x_max, y_max; } kRoi{0, 720, 160, 960};

    auto percentiles = [](std::vector<int> v, std::initializer_list<double> qs) {
        std::sort(v.begin(), v.end());
        std::vector<int> out;
        out.reserve(qs.size());
        for (double q : qs) {
            if (v.empty()) { out.push_back(0); continue; }
            size_t idx = static_cast<size_t>(q * (v.size() - 1));
            out.push_back(v[idx]);
        }
        return out;
    };

    // --- ROI histogram + percentiles ---
    {
        const size_t HW = static_cast<size_t>(kH) * kW;
        // Bins of 5: 0-5, 5-10, ..., 95-100, 100+  => 21 bins.
        constexpr int kNumBins = 21;
        int bins[kNumBins] = {};
        std::vector<int> roi_vals;
        roi_vals.reserve(static_cast<size_t>(kRoi.x_max - kRoi.x_min) *
                         (kRoi.y_max - kRoi.y_min));
        int roi_max = 0;
        struct TopPx { int x, y, n; float vr, vg, vb; float tf; };
        std::vector<TopPx> roi_pixels;
        roi_pixels.reserve(roi_vals.capacity());

        for (int y = kRoi.y_min; y < kRoi.y_max; ++y) {
            for (int x = kRoi.x_min; x < kRoi.x_max; ++x) {
                size_t pix = static_cast<size_t>(y) * kW + x;
                int n = n_contrib[pix];
                roi_vals.push_back(n);
                if (n > roi_max) roi_max = n;
                int b = n / 5;
                if (b >= kNumBins - 1) b = kNumBins - 1;
                bins[b]++;
                float vr = vk_chw[0 * HW + pix];
                float vg = vk_chw[1 * HW + pix];
                float vb = vk_chw[2 * HW + pix];
                roi_pixels.push_back({x, y, n, vr, vg, vb, T_final[pix]});
            }
        }

        auto rp = percentiles(roi_vals, {0.05, 0.50, 0.95});
        std::printf("[NContribDump ROI] ROI=x[%d,%d) y[%d,%d) pixels=%zu\n",
                    kRoi.x_min, kRoi.x_max, kRoi.y_min, kRoi.y_max,
                    roi_vals.size());
        std::printf("[NContribDump ROI] n_contrib p5=%d p50=%d p95=%d max=%d\n",
                    rp[0], rp[1], rp[2], roi_max);
        std::printf("[NContribDump ROI] Histogram (bin width=5):\n");
        for (int i = 0; i < kNumBins; ++i) {
            if (i == kNumBins - 1) {
                std::printf("  [%3d, inf): %d pixels\n", i * 5, bins[i]);
            } else {
                std::printf("  [%3d,%3d):  %d pixels\n", i * 5, (i + 1) * 5, bins[i]);
            }
        }

        std::sort(roi_pixels.begin(), roi_pixels.end(),
                  [](const TopPx& a, const TopPx& b) { return a.n > b.n; });
        std::printf("[NContribDump ROI] Top-10 highest n_contrib pixels:\n");
        for (int i = 0; i < 10 && i < (int)roi_pixels.size(); ++i) {
            const auto& p = roi_pixels[i];
            std::printf("  #%2d (%4d,%4d) n=%5d  VK=[%.3f,%.3f,%.3f] T_final=%.4f\n",
                        i + 1, p.x, p.y, p.n, p.vr, p.vg, p.vb, p.tf);
        }
    }

    // --- Full-image percentiles for context ---
    {
        std::vector<int> full_vals(n_contrib.begin(), n_contrib.end());
        int full_max = 0;
        for (int v : full_vals) if (v > full_max) full_max = v;
        auto fp = percentiles(full_vals, {0.05, 0.50, 0.95});
        std::printf("[NContribDump FULL] pixels=%zu  p5=%d p50=%d p95=%d max=%d\n",
                    full_vals.size(), fp[0], fp[1], fp[2], full_max);
    }

    // Intentionally no assertion — diagnostic only.
    SUCCEED();
}
