// test_tile_binner_vulkan.cpp -- SP-2 T11 end-to-end integration test for
// TileBinnerVulkan::bin().
//
// Validates that the adapter correctly composes PrefixScanPass + ScatterPass,
// exposing the PreprocessOutput -> BinningOutput contract documented in
// tile_binner.h:
//   - total_pairs   == sum(tiles_touched)
//   - keys_unsorted / values_unsorted written into FrameAllocator-backed memory
//   - num_tiles     == ceil(W/tile_w) * ceil(H/tile_h)
//   - keys_sorted / values_sorted / tile_ranges left null for SorterVulkan (T14)
//
// Structural validation only — NO byte-for-byte comparison to the CUDA golden.
// Rationale (spec §4.8.3 NUMERICAL CONTRACT, and inline comments in
// test_scatter_pass_vk.cpp lines 150-162):
//   - TileBinnerVulkan produces EXCLUSIVE point_offsets internally.
//     preprocess_point_offsets.npy is INCLUSIVE (CUDA's duplicateWithKeys
//     mutates offsets in-place via atomics). Comparing byte-for-byte would
//     produce spurious mismatches.
//   - sort_keys_unsorted.npy / sort_values_unsorted.npy do exist in the tiny
//     fixture, but CUDA's float-radius rect may differ from our int-ceiled
//     radius rect by sub-pixel edge cases — that cross-check already happens
//     in test_scatter_pass_vk.cpp. This test gates the COMPOSITION, not the
//     numerics of the individual shaders.
//
// Two tests:
//   Bin_TinyFixture_ValidKeys -- loads tiny-fixture preprocess golden, builds
//       a PreprocessOutput in host memory, runs bin(), validates structural
//       invariants (R, index ranges, tile-id ranges, depth positivity, slot ==
//       cap).
//   Bin_EmptyScene            -- all tiles_touched = 0 -> total_pairs = 0 with
//       null pair arrays (empty-scene fast path in tile_binner_vulkan.cpp:62).

#include "types.h"
#include "vulkan/tile_binner_vulkan.h"
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
// End-to-end composition test: PrefixScanPass + ScatterPass via bin().
// -----------------------------------------------------------------------------
TEST(TileBinner, Bin_TinyFixture_ValidKeys) {
    const std::string root = tiny_cam0_dir();

    // --- Load inputs from preprocess golden. ------------------------------
    auto m2d_npy     = load_npy(root + "/preprocess_means2D.npy");        // [N, 2] f32
    auto dep_npy     = load_npy(root + "/preprocess_depths.npy");         // [N]    f32
    auto rad_npy     = load_npy(root + "/preprocess_radii.npy");          // [N]    i32
    auto tt_npy      = load_npy(root + "/preprocess_tiles_touched.npy");  // [N]    i32/u32
    auto conic_opa   = load_npy(root + "/preprocess_conic_opacity.npy");  // [N, 4] f32
    auto rgb_npy     = load_npy(root + "/preprocess_rgb.npy");            // [N, 3] f32
    auto meta_npy    = load_npy(root + "/input_meta.npy");                // [4] f32

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
    // PreprocessOutput holds raw pointers. The backing vectors must live
    // until bin() returns.
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
    // This matches PreprocessOutput's split layout (conics + opacities_2d).
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

    // --- Build Camera (only width/height are consulted by bin() for the
    //     tile-grid computation; the rest is ignored because the 2D
    //     projection is already baked into PreprocessOutput). --------------
    Camera cam{};
    cam.width  = W;
    cam.height = H;

    // --- RenderConfig: bin() reads tile_w/tile_h for grid dimensions. ------
    RenderConfig cfg{};
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    // Expected grid geometry, independently of bin().
    const int num_tiles_x = (W + cfg.tile_w - 1) / cfg.tile_w;
    const int num_tiles_y = (H + cfg.tile_h - 1) / cfg.tile_h;
    const int num_tiles   = num_tiles_x * num_tiles_y;

    // --- Run Vulkan tile-binner. ------------------------------------------
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute-capable device?";

    FrameAllocator alloc(64u * 1024u * 1024u);  // 64 MB arena
    TileBinnerVulkan binner(ctx);
    BinningOutput result = binner.bin(pre, N, cam, cfg, alloc);

    // --- Invariant 1: total_pairs == sum(tiles_touched). ------------------
    int64_t expected_R = 0;
    for (int i = 0; i < N; ++i) expected_R += tt_vec[i];
    EXPECT_EQ(static_cast<int64_t>(result.total_pairs), expected_R)
        << "total_pairs should equal sum(tiles_touched).";

    // num_tiles must match the grid we computed independently.
    EXPECT_EQ(result.num_tiles, num_tiles)
        << "num_tiles should equal ceil(W/tw) * ceil(H/th).";

    // Downstream (SorterVulkan T14) fields left null by T10 contract.
    EXPECT_EQ(result.keys_sorted,   nullptr);
    EXPECT_EQ(result.values_sorted, nullptr);
    EXPECT_EQ(result.tile_ranges,   nullptr);

    // If the fixture somehow scatters zero pairs, the rest of the checks are
    // vacuous. Assert > 0 so we know the test is exercising the scatter path.
    ASSERT_GT(result.total_pairs, 0)
        << "Tiny fixture produced zero scatter pairs — test would be vacuous.";
    ASSERT_NE(result.keys_unsorted,   nullptr);
    ASSERT_NE(result.values_unsorted, nullptr);

    const int R = result.total_pairs;

    // --- Invariant 2: every value is a valid Gaussian index in [0, N). ----
    for (int j = 0; j < R; ++j) {
        ASSERT_LT(result.values_unsorted[j], static_cast<uint32_t>(N))
            << "values_unsorted[" << j << "] = " << result.values_unsorted[j]
            << " is not in [0, N=" << N << ").";
    }

    // --- Invariant 3: every tile_id (high 32 bits) is in [0, num_tiles). --
    // --- Invariant 4: every depth (low 32 bits, reinterpret as float) > 0. -
    // Gaussians with depth <= 0.2 (near-plane cull) have radii=0 and are
    // skipped by scatter.comp, so every written pair must have positive depth.
    for (int j = 0; j < R; ++j) {
        const uint64_t k = result.keys_unsorted[j];
        const uint32_t tile_id    = static_cast<uint32_t>(k >> 32);
        const uint32_t depth_bits = static_cast<uint32_t>(k & 0xFFFFFFFFull);
        float depth;
        std::memcpy(&depth, &depth_bits, sizeof(float));

        ASSERT_LT(tile_id, static_cast<uint32_t>(num_tiles))
            << "pair " << j << " tile_id=" << tile_id
            << " >= num_tiles=" << num_tiles;
        // NaN and non-positive both fail `> 0` — catches culled Gaussians
        // erroneously scattered plus any bit-pattern corruption.
        ASSERT_GT(depth, 0.0f)
            << "pair " << j << " depth=" << depth;
    }

    // --- Invariant 5: slot == cap. For each Gaussian i, exactly
    //     tiles_touched[i] pairs in the output must carry value == i. -----
    // This is the end-to-end cap correctness assertion from scatter.comp's
    // NUMERICAL CONTRACT — same shape as test_scatter_pass_vk's invariant
    // (e), but exercised through the full bin() composition.
    std::vector<int> counts(N, 0);
    for (int j = 0; j < R; ++j) {
        counts[result.values_unsorted[j]]++;
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(counts[i], tt_vec[i])
            << "Gaussian " << i << " appears in " << counts[i]
            << " pairs, expected tiles_touched=" << tt_vec[i];
    }
}

// -----------------------------------------------------------------------------
// Empty-scene fast path: all Gaussians culled (radii=0, tiles_touched=0).
// -----------------------------------------------------------------------------
TEST(TileBinner, Bin_EmptyScene) {
    // Minimal sentinel scene where every Gaussian has zero radius and zero
    // tiles_touched. tile_binner_vulkan.cpp:128 takes the R == 0 branch after
    // the prefix scan and returns null pair arrays.
    constexpr int N = 8;
    std::vector<float> means2d(N * 2, 0.0f);
    std::vector<float> depths (N,       1.0f);
    std::vector<float> conics (N * 3,   0.0f);
    std::vector<float> opacities(N,     0.0f);
    std::vector<float> rgb    (N * 3,   0.0f);
    std::vector<int>   radii  (N,       0);
    std::vector<int>   tt     (N,       0);

    PreprocessOutput pre{};
    pre.means2D       = means2d.data();
    pre.depths        = depths.data();
    pre.conics        = conics.data();
    pre.opacities_2d  = opacities.data();
    pre.rgb           = rgb.data();
    pre.radii         = radii.data();
    pre.tiles_touched = tt.data();
    pre.eval_3D       = false;

    Camera cam{};
    cam.width  = 64;
    cam.height = 64;

    RenderConfig cfg{};
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute-capable device?";

    FrameAllocator alloc(4u * 1024u * 1024u);
    TileBinnerVulkan binner(ctx);
    BinningOutput result = binner.bin(pre, N, cam, cfg, alloc);

    EXPECT_EQ(result.total_pairs, 0);
    EXPECT_EQ(result.keys_unsorted,   nullptr);
    EXPECT_EQ(result.values_unsorted, nullptr);
    EXPECT_EQ(result.keys_sorted,     nullptr);
    EXPECT_EQ(result.values_sorted,   nullptr);
    EXPECT_EQ(result.tile_ranges,     nullptr);

    // Grid geometry is still reported for downstream consumers.
    const int expect_num_tiles = ((cam.width  + 15) / 16) *
                                 ((cam.height + 15) / 16);
    EXPECT_EQ(result.num_tiles, expect_num_tiles);
}
