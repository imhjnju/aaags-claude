// test_tile_range_pass_vk.cpp -- SP-2 T14 TDD gate for TileRangePass.
//
// Validates tile_range.comp (spec §4.8.6) by manually constructing a sorted
// key array with known tile-id groupings and comparing the output
// tile_ranges against a host-side oracle.
//
// Key layout: high 32 bits = tile_id, low 32 bits = depth (arbitrary for
// this test since tile_range.comp only looks at the top 32 bits).
//
// Preconditions checked at the boundary:
//   * tile_ranges MUST be zero-initialised before dispatch — the shader only
//     writes tiles that contain at least one key, and leaves empty tiles at
//     [0,0). We pre-zero in every test.
//   * num_elements may be 0 (empty scene) — the shader no-ops and all
//     ranges stay zero.

#include "vulkan/sort_passes.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

// Build a uint64 sort key from (tile_id, depth) layout.
constexpr uint64_t make_key(uint32_t tile, uint32_t depth) {
    return (static_cast<uint64_t>(tile) << 32) | static_cast<uint64_t>(depth);
}

// Host-side oracle matching the tile_range.comp contract exactly:
//   For each i in [0, R), if i is the first element of its tile
//     then tile_ranges[tile_id*2+0] = i.
//   If i is the last element of its tile
//     then tile_ranges[tile_id*2+1] = i + 1.
// All other tiles remain [0, 0).
std::vector<uint32_t> oracle_tile_ranges(const std::vector<uint64_t>& sorted_keys,
                                         uint32_t num_tiles) {
    std::vector<uint32_t> ranges(num_tiles * 2u, 0u);
    const uint32_t R = static_cast<uint32_t>(sorted_keys.size());
    for (uint32_t i = 0; i < R; ++i) {
        const uint32_t tile = static_cast<uint32_t>(sorted_keys[i] >> 32);
        if (tile >= num_tiles) continue;  // malformed — match shader guard
        const bool is_start = (i == 0u) ||
            static_cast<uint32_t>(sorted_keys[i - 1u] >> 32) != tile;
        const bool is_end   = (i == R - 1u) ||
            static_cast<uint32_t>(sorted_keys[i + 1u] >> 32) != tile;
        if (is_start) ranges[tile * 2u + 0u] = i;
        if (is_end)   ranges[tile * 2u + 1u] = i + 1u;
    }
    return ranges;
}

std::vector<uint32_t> run_tile_range(VulkanContext& ctx,
                                     const std::vector<uint64_t>& sorted_keys,
                                     uint32_t num_tiles) {
    const uint32_t R = static_cast<uint32_t>(sorted_keys.size());

    // Upload keys (R may be 0 — allocate a one-byte dummy buffer then).
    VulkanBuffer keys_buf(ctx,
        std::max<VkDeviceSize>(R * sizeof(uint64_t), 4u));
    if (R > 0) {
        keys_buf.upload(sorted_keys.data(), R * sizeof(uint64_t));
    }

    // tile_ranges pre-zeroed (precondition).
    VulkanBuffer ranges_buf(ctx, num_tiles * 2u * sizeof(uint32_t));
    std::vector<uint32_t> zeros(num_tiles * 2u, 0u);
    ranges_buf.upload(zeros.data(), zeros.size() * sizeof(uint32_t));

    TileRangePass pass(ctx);
    pass.bind_buffers(keys_buf.handle(), ranges_buf.handle());
    pass.dispatch_sync(R, num_tiles);

    std::vector<uint32_t> got(num_tiles * 2u, 0u);
    ranges_buf.download(got.data(), got.size() * sizeof(uint32_t));
    return got;
}

}  // namespace

// -----------------------------------------------------------------------------
// Test 1: dense tiles with one empty tile in the middle.
//
// R=5 sorted keys spread across 3 used tiles (0, 1, 3); tile 2 has no keys
// and must remain [0, 0) after dispatch.
// -----------------------------------------------------------------------------
TEST(TileRangePassVk, DenseTilesWithGap) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    const std::vector<uint64_t> keys = {
        make_key(0, 1),  // tile 0, depth 1
        make_key(0, 2),  // tile 0, depth 2
        make_key(1, 1),  // tile 1, depth 1
        make_key(3, 1),  // tile 3, depth 1
        make_key(3, 5),  // tile 3, depth 5
    };
    const uint32_t num_tiles = 4;

    auto got      = run_tile_range(ctx, keys, num_tiles);
    auto expected = oracle_tile_ranges(keys, num_tiles);

    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(got[i], expected[i])
            << "mismatch at i=" << i
            << " (tile=" << (i / 2) << " " << (i % 2 ? "end" : "start") << ")";
    }

    // Spot-check concrete values so this test is self-explanatory:
    //   tile 0: [0, 2)
    //   tile 1: [2, 3)
    //   tile 2: [0, 0)  empty
    //   tile 3: [3, 5)
    EXPECT_EQ(got[0 * 2 + 0], 0u); EXPECT_EQ(got[0 * 2 + 1], 2u);
    EXPECT_EQ(got[1 * 2 + 0], 2u); EXPECT_EQ(got[1 * 2 + 1], 3u);
    EXPECT_EQ(got[2 * 2 + 0], 0u); EXPECT_EQ(got[2 * 2 + 1], 0u);
    EXPECT_EQ(got[3 * 2 + 0], 3u); EXPECT_EQ(got[3 * 2 + 1], 5u);
}

// -----------------------------------------------------------------------------
// Test 2: empty scene (R == 0). All ranges must remain zero.
// -----------------------------------------------------------------------------
TEST(TileRangePassVk, EmptyRange) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    const std::vector<uint64_t> keys{};
    const uint32_t num_tiles = 8;

    auto got = run_tile_range(ctx, keys, num_tiles);
    ASSERT_EQ(got.size(), static_cast<size_t>(num_tiles) * 2u);
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i], 0u) << "non-zero at i=" << i;
    }
}

// -----------------------------------------------------------------------------
// Test 3: single-tile fill. R = num_elements all in tile 0.
// Exercises the "i == R-1" end-boundary branch and "i == 0" start-boundary
// branch on the same dispatch (both run for a single-tile batch).
// -----------------------------------------------------------------------------
TEST(TileRangePassVk, SingleTileAllKeys) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    std::vector<uint64_t> keys(50);
    for (uint32_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key(0, i);
    }
    const uint32_t num_tiles = 3;

    auto got = run_tile_range(ctx, keys, num_tiles);
    // tile 0: [0, 50)
    EXPECT_EQ(got[0 * 2 + 0], 0u);
    EXPECT_EQ(got[0 * 2 + 1], 50u);
    // tile 1, 2: empty
    EXPECT_EQ(got[1 * 2 + 0], 0u); EXPECT_EQ(got[1 * 2 + 1], 0u);
    EXPECT_EQ(got[2 * 2 + 0], 0u); EXPECT_EQ(got[2 * 2 + 1], 0u);
}

// -----------------------------------------------------------------------------
// Test 4: cross-boundary sweep — multiple tiles of varying sizes, validates
// against the host oracle. 100 keys across 7 tiles.
// -----------------------------------------------------------------------------
TEST(TileRangePassVk, MultiTileOracleCompare) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    // tile -> count distribution: {3, 0, 10, 25, 0, 50, 12}  (total 100)
    // Build sorted keys deterministically.
    const std::vector<uint32_t> counts = {3, 0, 10, 25, 0, 50, 12};
    const uint32_t num_tiles = static_cast<uint32_t>(counts.size());

    std::vector<uint64_t> keys;
    keys.reserve(100);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        for (uint32_t d = 0; d < counts[t]; ++d) {
            keys.push_back(make_key(t, d));
        }
    }
    ASSERT_EQ(keys.size(), 100u);

    auto got      = run_tile_range(ctx, keys, num_tiles);
    auto expected = oracle_tile_ranges(keys, num_tiles);

    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(got[i], expected[i]) << "mismatch at i=" << i;
    }
}
