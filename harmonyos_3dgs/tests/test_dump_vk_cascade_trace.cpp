// test_dump_vk_cascade_trace.cpp — Phase 4 / Milestone A: produce 17 VK-side
// cascade-trace NPYs for the basket-aaa fixture so
// test_cascade_equivalence.cpp's Vk_vs_Cuda test can run.
//
// Mirrors the shape/dtype contract of CUDA DeviceCascadeTraceView (see
// AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/
// stopthepop/cascade_trace.h):
//
//   slot_lookup.npy          [num_tiles]           i32
//   tail_depths.npy          [K, 512, 16, 64]      f32
//   tail_ids.npy             [K, 512, 16, 64]      i32
//   tail_wcur.npy            [K]                   u32
//   mid_depths.npy           [K, 1024, 16, 4, 8]   f32
//   mid_ids.npy              [K, 1024, 16, 4, 8]   i32
//   mid_wcur.npy             [K]                   u32
//   head_ins_depth.npy       [K, 256, 4096]        f32
//   head_ins_alpha.npy       [K, 256, 4096]        f32
//   head_ins_gid.npy         [K, 256, 4096]        i32
//   head_ins_cursor.npy      [K, 256]              u32
//   head_blend_depth.npy     [K, 256, 4096]        f32
//   head_blend_alpha.npy     [K, 256, 4096]        f32
//   head_blend_T.npy         [K, 256, 4096]        f32
//   head_blend_gid.npy       [K, 256, 4096]        i32
//   head_blend_cursor.npy    [K, 256]              u32
//
// Milestone A does NOT implement any trace writes in the shader, so all
// buffers are expected to be zero. Milestone B..E will populate them.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "camera_utils.h"
#include "ply_loader.h"
#include "types.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

#include "golden/npy_reader.h"
#include "golden/npy_writer.h"

namespace {

constexpr int kW = 720;
constexpr int kH = 960;

const std::string kPlyPath =
    "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply";
const std::string kCamPath =
    "/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json";

}  // namespace

TEST(DumpVkCascadeTrace, Basket_Cam0) {
    if (!std::filesystem::exists(kPlyPath)) {
        GTEST_SKIP() << "PLY not available at " << kPlyPath;
    }
    if (!std::filesystem::exists(kCamPath)) {
        GTEST_SKIP() << "cameras.json not available at " << kCamPath;
    }

    // Selected tile set must match test_dump_cascade_fixtures.cpp so the VK
    // trace lines up with the CUDA gold. We read selected_tiles.npy (written
    // by that test) from the parent cascade_trace dir. If missing, skip —
    // running test_dump_cascade_fixtures first is a prerequisite.
    const std::string parent_dump = std::string(CMAKE_BINARY_DIR) + "/cascade_trace";
    const std::string sel_path    = parent_dump + "/selected_tiles.npy";
    if (!std::filesystem::exists(sel_path)) {
        GTEST_SKIP() << "Run DumpCascadeFixtures.Basket_Cam0 first — "
                     << sel_path << " missing.";
    }
    NpyArray sel_arr = load_npy(sel_path);
    ASSERT_EQ(sel_arr.dtype, NpyDtype::uint32);
    ASSERT_EQ(sel_arr.shape.size(), 1u);
    std::vector<uint32_t> selected_tiles(sel_arr.numel());
    std::memcpy(selected_tiles.data(), sel_arr.u32(),
                sel_arr.numel() * sizeof(uint32_t));
    ASSERT_FALSE(selected_tiles.empty());

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
    RasterizerVulkan   rasterizer(ctx, /*eval_3D=*/true);

    PreprocessOutput pre = preproc.process(model.data, cam, cfg, alloc);
    ASSERT_TRUE(pre.eval_3D);
    const int N = model.data.count;

    BinningOutput bin = binner.bin(pre, N, cam, cfg, alloc);
    sorter.sort(bin, alloc);

    // Run the traced rasterize. Image/cache outputs are ignored; only the
    // TraceDump matters for Milestone A.
    const int HW = kW * kH;
    std::vector<float> out_img(static_cast<size_t>(HW) * 3u, 0.f);
    auto dump = rasterizer.rasterize_traced(pre, bin, cam, cfg,
                                            selected_tiles,
                                            out_img.data(),
                                            /*cache=*/nullptr);

    const uint32_t K         = dump.K;
    const uint32_t num_tiles = dump.num_tiles;
    ASSERT_EQ(K, selected_tiles.size());

    const std::string vk_dir = parent_dump + "/vk";
    std::filesystem::create_directories(vk_dir);

    // Shape constants — must match CascadeTraceConsts and the comparator.
    constexpr size_t MAX_TAIL  = 512;
    constexpr size_t MAX_MID   = 1024;
    constexpr size_t MAX_HEAD  = 4096;
    constexpr size_t SUBTILES  = 16;
    constexpr size_t QUADS     = 4;
    constexpr size_t TAIL_SLOT = 64;
    constexpr size_t MID_SLOT  = 8;
    constexpr size_t PIX       = 256;

    save_npy_i32(vk_dir + "/slot_lookup.npy",
                 dump.slot_lookup.data(), {num_tiles});

    save_npy_f32(vk_dir + "/tail_depths.npy", dump.tail_depths.data(),
                 {K, MAX_TAIL, SUBTILES, TAIL_SLOT});
    save_npy_i32(vk_dir + "/tail_ids.npy",    dump.tail_ids.data(),
                 {K, MAX_TAIL, SUBTILES, TAIL_SLOT});
    save_npy_u32(vk_dir + "/tail_wcur.npy",   dump.tail_wcur.data(),
                 {K});

    save_npy_f32(vk_dir + "/mid_depths.npy", dump.mid_depths.data(),
                 {K, MAX_MID, SUBTILES, QUADS, MID_SLOT});
    save_npy_i32(vk_dir + "/mid_ids.npy",    dump.mid_ids.data(),
                 {K, MAX_MID, SUBTILES, QUADS, MID_SLOT});
    save_npy_u32(vk_dir + "/mid_wcur.npy",   dump.mid_wcur.data(),
                 {K});

    save_npy_f32(vk_dir + "/head_ins_depth.npy",  dump.head_ins_depth.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_f32(vk_dir + "/head_ins_alpha.npy",  dump.head_ins_alpha.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_i32(vk_dir + "/head_ins_gid.npy",    dump.head_ins_gid.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_u32(vk_dir + "/head_ins_cursor.npy", dump.head_ins_cursor.data(),
                 {K, PIX});

    save_npy_f32(vk_dir + "/head_blend_depth.npy",  dump.head_blend_depth.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_f32(vk_dir + "/head_blend_alpha.npy",  dump.head_blend_alpha.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_f32(vk_dir + "/head_blend_T.npy",      dump.head_blend_T.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_i32(vk_dir + "/head_blend_gid.npy",    dump.head_blend_gid.data(),
                 {K, PIX, MAX_HEAD});
    save_npy_u32(vk_dir + "/head_blend_cursor.npy", dump.head_blend_cursor.data(),
                 {K, PIX});

    // Round-trip sanity — ensure writer produced a valid NPY we can re-read.
    {
        NpyArray tr = load_npy(vk_dir + "/tail_depths.npy");
        ASSERT_EQ(tr.dtype, NpyDtype::float32);
        ASSERT_EQ(tr.numel(), size_t(K) * MAX_TAIL * SUBTILES * TAIL_SLOT);
    }

    std::printf("[DumpVkCascadeTrace] Dumped K=%u trace buffers to %s\n",
                K, vk_dir.c_str());
    std::printf("[DumpVkCascadeTrace] num_tiles=%u selected=", num_tiles);
    for (auto t : selected_tiles) std::printf(" %u", t);
    std::printf("\n");

    model.free();
}
