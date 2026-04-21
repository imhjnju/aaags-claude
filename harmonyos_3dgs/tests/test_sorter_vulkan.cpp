// test_sorter_vulkan.cpp -- SP-2 T15: full bin+sort integration test for
// TileBinnerVulkan::bin() -> SorterVulkan::sort() chain.
//
// Validates the composed pipeline:
//   PreprocessOutput -> TileBinnerVulkan::bin() -> BinningOutput (unsorted)
//                    -> SorterVulkan::sort()    -> BinningOutput (sorted)
//
// Structural invariants checked (no byte-for-byte golden comparison — sort
// golden npy files don't exist):
//   1. bin() succeeds with total_pairs > 0 (non-trivial scene).
//   2. sort() populates keys_sorted, values_sorted, tile_ranges (non-null).
//   3. keys_sorted is monotonically non-decreasing.
//   4. All values_sorted are valid Gaussian indices in [0, N).
//   5. tile_ranges[t*2] <= tile_ranges[t*2+1] for every tile t.
//   6. tile_ranges cover exactly R elements with no gaps or overlaps.
//   7. For each tile t, every key in its range has the correct tile_id in the
//      high 32 bits (keys encode tile_id:depth).

#include "types.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/vk_context.h"

#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

}  // namespace

// -----------------------------------------------------------------------------
// End-to-end integration test: TileBinnerVulkan::bin() + SorterVulkan::sort().
// -----------------------------------------------------------------------------
TEST(SorterVulkan, BinAndSort_TinyFixture) {
    const std::string root = tiny_cam0_dir();

    // --- Load preprocess golden inputs. -------------------------------------
    auto m2d_npy   = load_npy(root + "/preprocess_means2D.npy");        // [N, 2] f32
    auto dep_npy   = load_npy(root + "/preprocess_depths.npy");         // [N]    f32
    auto rad_npy   = load_npy(root + "/preprocess_radii.npy");          // [N]    i32
    auto tt_npy    = load_npy(root + "/preprocess_tiles_touched.npy");  // [N]    i32/u32
    auto conic_opa = load_npy(root + "/preprocess_conic_opacity.npy");  // [N, 4] f32
    auto rgb_npy   = load_npy(root + "/preprocess_rgb.npy");            // [N, 3] f32
    auto meta_npy  = load_npy(root + "/input_meta.npy");                // [4]    f32

    const int N = static_cast<int>(tt_npy.numel());
    ASSERT_GT(N, 0) << "Tiny fixture is empty — test is vacuous.";
    ASSERT_EQ(m2d_npy.numel(),   static_cast<size_t>(N) * 2);
    ASSERT_EQ(dep_npy.numel(),   static_cast<size_t>(N));
    ASSERT_EQ(rad_npy.numel(),   static_cast<size_t>(N));
    ASSERT_EQ(conic_opa.numel(), static_cast<size_t>(N) * 4);
    ASSERT_EQ(rgb_npy.numel(),   static_cast<size_t>(N) * 3);

    // meta = [sh_degree, sh_coeffs_per_g, H, W]
    ASSERT_EQ(meta_npy.shape.size(), 1u);
    ASSERT_EQ(meta_npy.shape[0], 4u);
    const int H = static_cast<int>(meta_npy.f32()[2]);
    const int W = static_cast<int>(meta_npy.f32()[3]);

    // --- Copy NpyArray payloads into vectors that outlive bin(). -----------
    std::vector<float> means2d_vec(m2d_npy.f32(), m2d_npy.f32() + N * 2);
    std::vector<float> depths_vec (dep_npy.f32(), dep_npy.f32() + N);
    std::vector<float> rgb_vec    (rgb_npy.f32(), rgb_npy.f32() + N * 3);

    // radii: preprocess_radii.npy is int32.
    std::vector<int> radii_vec(N);
    std::memcpy(radii_vec.data(), rad_npy.i32(), static_cast<size_t>(N) * sizeof(int32_t));

    // tiles_touched: golden dtype may be i32 or u32. PreprocessOutput wants
    // int*. Both representations share the bit pattern for non-negative tile
    // counts (the preprocess shader writes non-negative `int`).
    std::vector<int> tt_vec(N);
    if (tt_npy.dtype == NpyDtype::uint32) {
        std::memcpy(tt_vec.data(), tt_npy.u32(), static_cast<size_t>(N) * sizeof(uint32_t));
    } else {
        ASSERT_EQ(tt_npy.dtype, NpyDtype::int32);
        std::memcpy(tt_vec.data(), tt_npy.i32(), static_cast<size_t>(N) * sizeof(int32_t));
        for (int i = 0; i < N; ++i) {
            ASSERT_GE(tt_vec[i], 0) << "tiles_touched[" << i << "] negative";
        }
    }

    // Deinterleave conic_opacity [N,4]={a,b,c,opa} -> conics[N,3] + opacities[N].
    std::vector<float> conics_vec(static_cast<size_t>(N) * 3);
    std::vector<float> opacities_vec(N);
    const float* co = conic_opa.f32();
    for (int i = 0; i < N; ++i) {
        conics_vec[i * 3 + 0] = co[i * 4 + 0];
        conics_vec[i * 3 + 1] = co[i * 4 + 1];
        conics_vec[i * 3 + 2] = co[i * 4 + 2];
        opacities_vec[i]      = co[i * 4 + 3];
    }

    // --- Build PreprocessOutput (host-side, no device_data). ---------------
    PreprocessOutput pre{};
    pre.means2D       = means2d_vec.data();
    pre.depths        = depths_vec.data();
    pre.conics        = conics_vec.data();
    pre.opacities_2d  = opacities_vec.data();
    pre.rgb           = rgb_vec.data();
    pre.radii         = radii_vec.data();
    pre.tiles_touched = tt_vec.data();
    pre.eval_3D       = false;

    Camera cam{};
    cam.width  = W;
    cam.height = H;

    RenderConfig cfg{};
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    const int num_tiles_x = (W + cfg.tile_w - 1) / cfg.tile_w;
    const int num_tiles_y = (H + cfg.tile_h - 1) / cfg.tile_h;
    const int num_tiles   = num_tiles_x * num_tiles_y;

    // --- Run Vulkan pipeline: bin then sort. ---------------------------------
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute-capable device?";

    FrameAllocator alloc(128u * 1024u * 1024u);  // 128 MB arena
    TileBinnerVulkan binner(ctx);
    SorterVulkan     sorter(ctx);

    BinningOutput result = binner.bin(pre, N, cam, cfg, alloc);

    // Invariant: bin() produces non-empty output for non-trivial scene.
    ASSERT_GT(result.total_pairs, 0)
        << "Empty binning output — degenerate scene? Check fixture.";
    ASSERT_NE(result.keys_unsorted,   nullptr);
    ASSERT_NE(result.values_unsorted, nullptr);
    EXPECT_EQ(result.num_tiles, num_tiles);

    sorter.sort(result, alloc);

    const int R = result.total_pairs;

    // --- Invariant: sort() populates all three output arrays. ---------------
    ASSERT_NE(result.keys_sorted,   nullptr);
    ASSERT_NE(result.values_sorted, nullptr);
    ASSERT_NE(result.tile_ranges,   nullptr);

    // --- Invariant: keys_sorted is monotonically non-decreasing. -----------
    for (int i = 1; i < R; ++i) {
        EXPECT_LE(result.keys_sorted[i - 1], result.keys_sorted[i])
            << "keys_sorted not monotonic at i=" << i;
    }

    // --- Invariant: all values are valid Gaussian indices in [0, N). --------
    for (int i = 0; i < R; ++i) {
        EXPECT_LT(result.values_sorted[i], static_cast<uint32_t>(N))
            << "values_sorted[" << i << "] = " << result.values_sorted[i]
            << " out of range [0, N=" << N << ")";
    }

    // --- Invariant: tile_ranges internal consistency. -----------------------
    for (int t = 0; t < num_tiles; ++t) {
        const uint32_t start = result.tile_ranges[t * 2 + 0];
        const uint32_t end   = result.tile_ranges[t * 2 + 1];
        EXPECT_LE(start, end)
            << "tile " << t << " has start=" << start << " > end=" << end;
        EXPECT_LE(end, static_cast<uint32_t>(R))
            << "tile " << t << " end=" << end << " > R=" << R;
    }

    // --- Invariant: tile_ranges covers exactly R elements (no gaps/overlaps). --
    uint32_t total_covered = 0;
    for (int t = 0; t < num_tiles; ++t) {
        total_covered += result.tile_ranges[t * 2 + 1] - result.tile_ranges[t * 2 + 0];
    }
    EXPECT_EQ(total_covered, static_cast<uint32_t>(R))
        << "tile_ranges cover " << total_covered << " elements, expected R=" << R;

    // --- Invariant: every key in tile t's range encodes tile_id == t. -------
    // Keys are 64-bit: high 32 bits = tile_id, low 32 bits = depth bits.
    for (int t = 0; t < num_tiles; ++t) {
        const uint32_t start = result.tile_ranges[t * 2 + 0];
        const uint32_t end   = result.tile_ranges[t * 2 + 1];
        for (uint32_t j = start; j < end; ++j) {
            const uint32_t tile_id = static_cast<uint32_t>(result.keys_sorted[j] >> 32);
            EXPECT_EQ(tile_id, static_cast<uint32_t>(t))
                << "key at position " << j << " has tile_id=" << tile_id
                << " but is in tile " << t << "'s range [" << start << ", " << end << ")";
        }
    }
}
