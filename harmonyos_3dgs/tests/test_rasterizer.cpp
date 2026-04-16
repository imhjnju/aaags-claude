#include <gtest/gtest.h>
#include "types.h"
#include "cpu/rasterizer_cpu.h"
#include <cmath>
#include <cstring>
#include <vector>

TEST(RasterizerCPU, BackgroundOnly) {
    // No Gaussians → output should be pure background color
    FrameAllocator alloc(1024 * 1024);
    Camera cam{};
    cam.width = 32; cam.height = 32;
    RenderConfig cfg{};
    cfg.bg_color[0] = 1.0f; cfg.bg_color[1] = 0.0f; cfg.bg_color[2] = 0.0f;

    PreprocessOutput pre{};
    BinningOutput bin{};
    bin.total_pairs = 0;
    bin.num_tiles = 4; // 2x2 tiles for 32x32 at 16x16
    bin.tile_ranges = alloc.allocate_array<uint32_t>(8);
    memset(bin.tile_ranges, 0, 8 * sizeof(uint32_t));

    std::vector<float> img(32 * 32 * 3, 0.0f);
    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, img.data());

    // All pixels should be red background
    for (int i = 0; i < 32*32; i++) {
        EXPECT_FLOAT_EQ(img[i*3+0], 1.0f) << "pixel " << i;
        EXPECT_FLOAT_EQ(img[i*3+1], 0.0f) << "pixel " << i;
        EXPECT_FLOAT_EQ(img[i*3+2], 0.0f) << "pixel " << i;
    }
}

TEST(RasterizerCPU, SingleGaussian_CenterContribution) {
    // One Gaussian at center of 32x32 image
    FrameAllocator alloc(1024 * 1024);
    Camera cam{};
    cam.width = 32; cam.height = 32;
    RenderConfig cfg{};
    cfg.bg_color[0] = 0; cfg.bg_color[1] = 0; cfg.bg_color[2] = 0;

    // Preprocess output for 1 Gaussian
    float means2d[2] = {16.0f, 16.0f};  // Center pixel
    float depths[1] = {5.0f};
    // Conic for a circular Gaussian with sigma=3: inv_cov = 1/(sigma^2) = 1/9
    // conic = (a, b, c) = (1/9, 0, 1/9) ≈ (0.111, 0, 0.111)
    float conics[3] = {0.111f, 0.0f, 0.111f};
    float opacities[1] = {0.95f};
    float rgb[3] = {0.8f, 0.5f, 0.2f};
    int radii[1] = {10};
    int tiles_touched[1] = {1};

    PreprocessOutput pre{};
    pre.means2D = means2d; pre.depths = depths; pre.conics = conics;
    pre.opacities_2d = opacities; pre.rgb = rgb;
    pre.radii = radii; pre.tiles_touched = tiles_touched;

    // Binning: 1 pair in tile (1,1) — the center tile of a 2x2 grid
    BinningOutput bin{};
    bin.total_pairs = 1;
    bin.num_tiles = 4;
    uint64_t keys_sorted[1] = {(uint64_t)(1*2+1) << 32}; // tile (1,1)
    uint32_t values_sorted[1] = {0};
    bin.keys_sorted = keys_sorted;
    bin.values_sorted = values_sorted;
    bin.tile_ranges = alloc.allocate_array<uint32_t>(8);
    memset(bin.tile_ranges, 0, 8 * sizeof(uint32_t));
    // tile_id = 1*2+1 = 3, so tile_ranges[3*2] = 0, tile_ranges[3*2+1] = 1
    bin.tile_ranges[6] = 0;
    bin.tile_ranges[7] = 1;

    std::vector<float> img(32 * 32 * 3, 0.0f);
    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, img.data());

    // Center pixel (16,16) should have color contribution
    int cx = 16, cy = 16;
    float r = img[(cy*32+cx)*3+0];
    float g = img[(cy*32+cx)*3+1];
    float b = img[(cy*32+cx)*3+2];
    EXPECT_GT(r, 0.0f);
    EXPECT_GT(g, 0.0f);
    EXPECT_GT(b, 0.0f);
    // Color should be close to rgb * alpha (at center, power=0, alpha=0.95)
    EXPECT_NEAR(r, 0.8f * 0.95f, 0.1f);
}

TEST(RasterizerCPU, OutputFormat_HWC) {
    // Verify output is HWC interleaved, not CHW planar
    FrameAllocator alloc(1024 * 1024);
    Camera cam{};
    cam.width = 4; cam.height = 4;
    RenderConfig cfg{};
    cfg.bg_color[0] = 0.1f; cfg.bg_color[1] = 0.2f; cfg.bg_color[2] = 0.3f;

    PreprocessOutput pre{};
    BinningOutput bin{};
    bin.total_pairs = 0;
    bin.num_tiles = 1;
    bin.tile_ranges = alloc.allocate_array<uint32_t>(2);
    memset(bin.tile_ranges, 0, 2 * sizeof(uint32_t));

    std::vector<float> img(4 * 4 * 3, 0.0f);
    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, img.data());

    // First pixel: img[0]=R, img[1]=G, img[2]=B (HWC)
    EXPECT_FLOAT_EQ(img[0], 0.1f);  // R
    EXPECT_FLOAT_EQ(img[1], 0.2f);  // G
    EXPECT_FLOAT_EQ(img[2], 0.3f);  // B
    // Second pixel: img[3]=R, img[4]=G, img[5]=B
    EXPECT_FLOAT_EQ(img[3], 0.1f);
    EXPECT_FLOAT_EQ(img[4], 0.2f);
    EXPECT_FLOAT_EQ(img[5], 0.3f);
}

TEST(RasterizerCPU, DepthOutput) {
    FrameAllocator alloc(1024 * 1024);
    Camera cam{}; cam.width = 32; cam.height = 32;
    RenderConfig cfg{};

    // Same setup as SingleGaussian test
    float means2d[2] = {16.0f, 16.0f};
    float depths[1] = {5.0f};
    float conics[3] = {0.111f, 0.0f, 0.111f};
    float opacities[1] = {0.95f};
    float rgb[3] = {0.8f, 0.5f, 0.2f};
    int radii[1] = {10}; int tiles_touched[1] = {1};

    PreprocessOutput pre{};
    pre.means2D = means2d; pre.depths = depths; pre.conics = conics;
    pre.opacities_2d = opacities; pre.rgb = rgb;
    pre.radii = radii; pre.tiles_touched = tiles_touched;

    BinningOutput bin{};
    bin.total_pairs = 1; bin.num_tiles = 4;
    uint64_t keys_sorted[1] = {(uint64_t)(3) << 32};
    uint32_t values_sorted[1] = {0};
    bin.keys_sorted = keys_sorted; bin.values_sorted = values_sorted;
    bin.tile_ranges = alloc.allocate_array<uint32_t>(8);
    memset(bin.tile_ranges, 0, 8 * sizeof(uint32_t));
    bin.tile_ranges[6] = 0; bin.tile_ranges[7] = 1;

    std::vector<float> img(32*32*3, 0.0f);
    std::vector<float> depth(32*32, 0.0f);
    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, img.data(), depth.data());

    // Center pixel should have non-zero inverse depth
    EXPECT_GT(depth[16*32+16], 0.0f);
    // inv_depth ≈ (1/5) * alpha * T = 0.2 * 0.95 * 1.0 = 0.19
    EXPECT_NEAR(depth[16*32+16], 0.2f * 0.95f, 0.05f);
}
