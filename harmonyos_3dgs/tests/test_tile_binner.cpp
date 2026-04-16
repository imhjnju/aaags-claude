#include <gtest/gtest.h>
#include "types.h"
#include "math_utils.h"
#include "cpu/tile_binner_cpu.h"
#include <cstring>

TEST(TileBinnerCPU, SingleGaussian_SingleTile) {
    FrameAllocator alloc(1024 * 1024);
    PreprocessOutput pre{};
    float means2d[2] = {8.0f, 8.0f};
    float depths[1] = {5.0f};
    int radii[1] = {5};
    int tiles[1] = {1};
    pre.means2D = means2d; pre.depths = depths;
    pre.radii = radii; pre.tiles_touched = tiles;
    // Other fields not used by binner but set to avoid null
    float conics[3] = {0}; float opa[1] = {0}; float rgb[3] = {0};
    pre.conics = conics; pre.opacities_2d = opa; pre.rgb = rgb;

    Camera cam{}; cam.width = 320; cam.height = 240;
    RenderConfig config{};

    TileBinnerCPU binner;
    auto out = binner.bin(pre, 1, cam, config, alloc);
    EXPECT_EQ(out.total_pairs, 1);
    EXPECT_EQ(out.keys_unsorted[0] >> 32, 0u);  // Tile (0,0)
    EXPECT_EQ(out.values_unsorted[0], 0u);
}

TEST(TileBinnerCPU, ZeroRadius_NoPairs) {
    FrameAllocator alloc(1024 * 1024);
    PreprocessOutput pre{};
    float means2d[2] = {8.0f, 8.0f};
    float depths[1] = {5.0f};
    int radii[1] = {0};  // culled
    int tiles[1] = {0};
    pre.means2D = means2d; pre.depths = depths;
    pre.radii = radii; pre.tiles_touched = tiles;
    float conics[3] = {0}; float opa[1] = {0}; float rgb[3] = {0};
    pre.conics = conics; pre.opacities_2d = opa; pre.rgb = rgb;

    Camera cam{}; cam.width = 320; cam.height = 240;
    RenderConfig config{};

    TileBinnerCPU binner;
    auto out = binner.bin(pre, 1, cam, config, alloc);
    EXPECT_EQ(out.total_pairs, 0);
}

TEST(TileBinnerCPU, PrefixSum) {
    FrameAllocator alloc(4 * 1024 * 1024);
    PreprocessOutput pre{};
    // 3 Gaussians touching 2, 3, 1 tiles respectively
    float means2d[6] = {8,8, 24,8, 8,24};  // pixel positions
    float depths[3] = {5.0f, 3.0f, 7.0f};
    int radii[3] = {5, 20, 5};
    int tiles[3] = {2, 3, 1};  // total = 6
    pre.means2D = means2d; pre.depths = depths;
    pre.radii = radii; pre.tiles_touched = tiles;
    float conics[9] = {0}; float opa[3] = {0}; float rgb[9] = {0};
    pre.conics = conics; pre.opacities_2d = opa; pre.rgb = rgb;

    Camera cam{}; cam.width = 320; cam.height = 240;
    RenderConfig config{};

    TileBinnerCPU binner;
    auto out = binner.bin(pre, 3, cam, config, alloc);
    EXPECT_EQ(out.total_pairs, 6);
}

TEST(TileBinnerCPU, KeyFormat_DepthEncoding) {
    FrameAllocator alloc(1024 * 1024);
    PreprocessOutput pre{};
    float means2d[2] = {8.0f, 8.0f};
    float depth_val = 5.0f;
    float depths[1] = {depth_val};
    int radii[1] = {5};
    int tiles[1] = {1};
    pre.means2D = means2d; pre.depths = depths;
    pre.radii = radii; pre.tiles_touched = tiles;
    float conics[3] = {0}; float opa[1] = {0}; float rgb[3] = {0};
    pre.conics = conics; pre.opacities_2d = opa; pre.rgb = rgb;

    Camera cam{}; cam.width = 320; cam.height = 240;
    RenderConfig config{};

    TileBinnerCPU binner;
    auto out = binner.bin(pre, 1, cam, config, alloc);
    // Extract depth from key
    uint32_t depth_bits = (uint32_t)(out.keys_unsorted[0] & 0xFFFFFFFF);
    float reconstructed;
    memcpy(&reconstructed, &depth_bits, sizeof(float));
    EXPECT_FLOAT_EQ(reconstructed, depth_val);
}

TEST(TileBinnerCPU, EmptyInput) {
    FrameAllocator alloc(1024 * 1024);
    PreprocessOutput pre{};
    Camera cam{}; cam.width = 320; cam.height = 240;
    RenderConfig config{};

    TileBinnerCPU binner;
    auto out = binner.bin(pre, 0, cam, config, alloc);
    EXPECT_EQ(out.total_pairs, 0);
}
