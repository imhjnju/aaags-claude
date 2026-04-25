// test_dump_cascade_fixtures.cpp — Phase 1 of CUDA/VK cascade equivalence test.
//
// Runs VK Preprocess + TileBinner + Sorter on basket-aaa.ply @ cam0 and dumps
// everything a CUDA-side harness needs to re-run just the cascade sort stage
// with identical inputs.
//
// Output dir (created if missing): ${CMAKE_BINARY_DIR}/cascade_trace/
//   meta.txt              N num_tiles_x num_tiles_y W H eval_3D
//   cam.npy         [22]  f32   viewproj(16) cam_pos(3) tan_fovx tan_fovy plus W,H
//   view.npy        [16]  f32   view matrix
//   viewproj.npy    [16]  f32   viewproj matrix
//   cam_pos.npy     [3]   f32
//   gauss2screen.npy [N*16] f32  (layout = VK preprocess output, flat)
//   conic_opacity.npy [N, 4] f32  (conics.xyz | opacities_2d)
//   rgb.npy         [N, 3] f32
//   means2D.npy     [N, 2] f32
//   depths.npy      [N]    f32
//   values_sorted.npy [R] u32
//   tile_ranges.npy   [T, 2] u32
//   selected_tiles.npy [K] u32   — test picks deterministic subset

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "camera_utils.h"
#include "ply_loader.h"
#include "types.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

#include "golden/npy_writer.h"

namespace {

constexpr int kW = 720;
constexpr int kH = 960;

const std::string kPlyPath =
    "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply";
const std::string kCamPath =
    "/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json";

// Target: pick the 4 tiles whose pair count falls in [min_count, max_count].
// Deterministic given fixed scene + camera.
constexpr uint32_t kSelectMinCount = 300;
constexpr uint32_t kSelectMaxCount = 2000;
constexpr size_t   kSelectK = 4;

}  // namespace

TEST(DumpCascadeFixtures, Basket_Cam0) {
    if (!std::filesystem::exists(kPlyPath)) {
        GTEST_SKIP() << "PLY not available at " << kPlyPath;
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
    cfg.sh_degree    = model.data.sh_degree;
    cfg.eval_3D      = true;
    cfg.antialiasing = false;
    cfg.training     = true;

    const size_t alloc_cap = 4ULL * 1024 * 1024 * 1024;
    FrameAllocator alloc(alloc_cap);

    PreprocessorVulkan preproc(ctx, /*eval_3D=*/true);
    TileBinnerVulkan   binner(ctx);
    SorterVulkan       sorter(ctx);

    PreprocessOutput pre = preproc.process(model.data, cam, cfg, alloc);
    ASSERT_TRUE(pre.eval_3D);
    ASSERT_NE(pre.gauss2screen, nullptr);
    ASSERT_NE(pre.conics, nullptr);
    ASSERT_NE(pre.opacities_2d, nullptr);
    ASSERT_NE(pre.rgb, nullptr);
    ASSERT_NE(pre.means2D, nullptr);
    ASSERT_NE(pre.depths, nullptr);

    const int N = model.data.count;

    BinningOutput bin = binner.bin(pre, N, cam, cfg, alloc);
    sorter.sort(bin, alloc);
    ASSERT_NE(bin.values_sorted, nullptr);
    ASSERT_NE(bin.tile_ranges, nullptr);
    const int num_tiles = bin.num_tiles;
    const int num_tiles_x = (kW + cfg.tile_w - 1) / cfg.tile_w;
    const int num_tiles_y = (kH + cfg.tile_h - 1) / cfg.tile_h;
    ASSERT_EQ(num_tiles, num_tiles_x * num_tiles_y);

    // Output directory
    const std::string dump_dir = std::string(CMAKE_BINARY_DIR) + "/cascade_trace";
    std::filesystem::create_directories(dump_dir);

    // --- conic_opacity packed [N, 4] = (cx, cy, cz, opacity_2d) ---
    std::vector<float> conic_opacity(static_cast<size_t>(N) * 4);
    for (int i = 0; i < N; ++i) {
        conic_opacity[i * 4 + 0] = pre.conics[i * 3 + 0];
        conic_opacity[i * 4 + 1] = pre.conics[i * 3 + 1];
        conic_opacity[i * 4 + 2] = pre.conics[i * 3 + 2];
        conic_opacity[i * 4 + 3] = pre.opacities_2d[i];
    }

    // --- select K tiles deterministically by pair count ---
    std::vector<uint32_t> selected;
    selected.reserve(kSelectK);
    for (int t = 0; t < num_tiles && selected.size() < kSelectK; ++t) {
        uint32_t start = bin.tile_ranges[t * 2 + 0];
        uint32_t end   = bin.tile_ranges[t * 2 + 1];
        uint32_t count = (end >= start) ? (end - start) : 0u;
        if (count >= kSelectMinCount && count <= kSelectMaxCount) {
            selected.push_back(static_cast<uint32_t>(t));
        }
    }
    // Fallback: if not enough tiles matched, relax and pick top-K by count.
    if (selected.size() < kSelectK) {
        std::vector<std::pair<uint32_t, uint32_t>> by_count;  // (count, tile_id)
        by_count.reserve(num_tiles);
        for (int t = 0; t < num_tiles; ++t) {
            uint32_t start = bin.tile_ranges[t * 2 + 0];
            uint32_t end   = bin.tile_ranges[t * 2 + 1];
            uint32_t count = (end >= start) ? (end - start) : 0u;
            if (count > 0) by_count.emplace_back(count, static_cast<uint32_t>(t));
        }
        std::sort(by_count.begin(), by_count.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        selected.clear();
        for (size_t i = 0; i < std::min<size_t>(kSelectK, by_count.size()); ++i) {
            selected.push_back(by_count[i].second);
        }
    }
    ASSERT_FALSE(selected.empty()) << "No non-empty tiles — check PLY/camera.";

    // --- write meta.txt ---
    {
        std::ofstream meta(dump_dir + "/meta.txt");
        meta << "N "            << N            << "\n"
             << "num_tiles_x "  << num_tiles_x  << "\n"
             << "num_tiles_y "  << num_tiles_y  << "\n"
             << "W "            << kW           << "\n"
             << "H "            << kH           << "\n"
             << "tile_w "       << cfg.tile_w   << "\n"
             << "tile_h "       << cfg.tile_h   << "\n"
             << "sh_degree "    << cfg.sh_degree<< "\n"
             << "eval_3D "      << (cfg.eval_3D ? 1 : 0) << "\n"
             << "total_pairs "  << bin.total_pairs << "\n";
    }

    // --- write camera pieces ---
    save_npy_f32(dump_dir + "/view.npy",     cam.view_matrix,     {16});
    save_npy_f32(dump_dir + "/viewproj.npy", cam.viewproj_matrix, {16});
    save_npy_f32(dump_dir + "/cam_pos.npy",  cam.cam_pos,         {3});
    {
        float fov[2] = { cam.tan_fovx, cam.tan_fovy };
        save_npy_f32(dump_dir + "/tan_fov.npy", fov, {2});
    }

    // --- write per-gaussian buffers ---
    save_npy_f32(dump_dir + "/gauss2screen.npy", pre.gauss2screen,
                 {static_cast<size_t>(N), 16});
    save_npy_f32(dump_dir + "/conic_opacity.npy", conic_opacity.data(),
                 {static_cast<size_t>(N), 4});
    save_npy_f32(dump_dir + "/rgb.npy",     pre.rgb,
                 {static_cast<size_t>(N), 3});
    save_npy_f32(dump_dir + "/means2D.npy", pre.means2D,
                 {static_cast<size_t>(N), 2});
    save_npy_f32(dump_dir + "/depths.npy",  pre.depths,
                 {static_cast<size_t>(N)});

    // --- write binning results ---
    save_npy_u32(dump_dir + "/values_sorted.npy", bin.values_sorted,
                 {static_cast<size_t>(bin.total_pairs)});
    save_npy_u32(dump_dir + "/tile_ranges.npy", bin.tile_ranges,
                 {static_cast<size_t>(num_tiles), 2});
    save_npy_u32(dump_dir + "/selected_tiles.npy", selected.data(),
                 {selected.size()});

    // --- round-trip sanity: load back one .npy and compare bytes ---
    {
        NpyArray g = load_npy(dump_dir + "/gauss2screen.npy");
        ASSERT_EQ(g.dtype, NpyDtype::float32);
        ASSERT_EQ(g.numel(), static_cast<size_t>(N) * 16);
        ASSERT_EQ(std::memcmp(g.f32(), pre.gauss2screen,
                              g.numel() * sizeof(float)), 0);
    }

    // --- report ---
    std::printf("[DumpCascadeFixtures] Dumped to %s\n", dump_dir.c_str());
    std::printf("[DumpCascadeFixtures]   N=%d  tiles=%d (%dx%d)  total_pairs=%d\n",
                N, num_tiles, num_tiles_x, num_tiles_y, bin.total_pairs);
    std::printf("[DumpCascadeFixtures]   selected %zu tile(s):", selected.size());
    for (auto t : selected) {
        uint32_t count = bin.tile_ranges[t * 2 + 1] - bin.tile_ranges[t * 2 + 0];
        std::printf(" %u(cnt=%u)", t, count);
    }
    std::printf("\n");

    model.free();
}
