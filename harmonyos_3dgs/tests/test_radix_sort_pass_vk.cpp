// test_radix_sort_pass_vk.cpp -- SP-2 T14 TDD gate for RadixSortPass.
//
// No CUDA/CPU-golden byte-for-byte comparison: `sort_keys_sorted.npy`/
// `sort_values_sorted.npy` are not part of the tiny fixture. Instead we
// validate correctness structurally against std::sort on the same input:
//
//   * Monotonicity:   got_keys[i] <= got_keys[i+1] for all i (LSD radix over
//                     uint64 produces ascending order).
//   * Permutation:    sorted(input_keys) == got_keys (multiset equality via
//                     side-by-side std::sort + element compare).
//   * Value mapping:  every (got_keys[i], got_values[i]) pair that started
//                     as (in_keys[k], in_values[k] == k) must satisfy
//                     in_keys[got_values[i]] == got_keys[i]. In our tests
//                     we seed values[k] = k so this test reduces to
//                     in_keys[got_values[i]] == got_keys[i].
//   * Stability:      for elements whose keys share the same high-32-bit
//                     tile_id, they must retain their relative ordering by
//                     low-32-bit depth (we build the test inputs so that
//                     stability is observable, see TestCase_TileCollisions).
//
// The Vulkan pass supports multi-workgroup SortPairs so large R stays on the
// CUDA-style key/value path instead of falling back to packed keyvals.

#include "vulkan/sort_passes.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Helper: run RadixSortPass on `keys` + `values` and return the sorted
// (keys, values). Values are returned so tests can verify the value mapping
// is preserved (stable permutation).
struct SortedResult {
    std::vector<uint64_t> keys;
    std::vector<uint32_t> values;
};

SortedResult run_sort(VulkanContext& ctx,
                      const std::vector<uint64_t>& keys,
                      const std::vector<uint32_t>& values) {
    const uint32_t R = static_cast<uint32_t>(keys.size());
    EXPECT_EQ(values.size(), static_cast<size_t>(R));

    constexpr uint32_t kLocalSize = 256u;
    constexpr uint32_t kBuckets = 16u;
    const uint32_t sort_wgs = (R + kLocalSize - 1u) / kLocalSize;
    const uint32_t hist_entries = sort_wgs * kBuckets;
    const uint32_t scan_wgs = (hist_entries + kLocalSize - 1u) / kLocalSize;
    const uint32_t scan_wgs2 = (scan_wgs + kLocalSize - 1u) / kLocalSize;

    VulkanBuffer keys_a(ctx, R * sizeof(uint64_t));
    VulkanBuffer vals_a(ctx, R * sizeof(uint32_t));
    VulkanBuffer keys_b(ctx, R * sizeof(uint64_t));
    VulkanBuffer vals_b(ctx, R * sizeof(uint32_t));
    VulkanBuffer hist_count(ctx, hist_entries * sizeof(uint32_t));
    VulkanBuffer hist_scan (ctx, hist_entries * sizeof(uint32_t));
    VulkanBuffer wg_sums   (ctx, std::max(scan_wgs, 1u) * sizeof(uint32_t));
    VulkanBuffer wg_sums2  (ctx, std::max(scan_wgs2, 1u) * sizeof(uint32_t));

    keys_a.upload(keys.data(),   R * sizeof(uint64_t));
    vals_a.upload(values.data(), R * sizeof(uint32_t));

    RadixSortPass sort(ctx);
    sort.sort_sync(keys_a.handle(),  vals_a.handle(),
                   keys_b.handle(),  vals_b.handle(),
                   hist_count.handle(), hist_scan.handle(),
                   wg_sums.handle(), wg_sums2.handle(),
                   R);

    SortedResult got;
    got.keys.resize(R, 0);
    got.values.resize(R, 0);
    keys_a.download(got.keys.data(),   R * sizeof(uint64_t));
    vals_a.download(got.values.data(), R * sizeof(uint32_t));
    return got;
}

// Check: keys are monotonically non-decreasing.
void expect_monotonic(const std::vector<uint64_t>& keys) {
    for (size_t i = 1; i < keys.size(); ++i) {
        EXPECT_LE(keys[i - 1], keys[i])
            << "non-monotonic at i=" << i
            << " keys[i-1]=" << keys[i - 1]
            << " keys[i]=" << keys[i];
    }
}

// Check: got_keys is a permutation of in_keys (multiset equality).
void expect_permutation(const std::vector<uint64_t>& in_keys,
                        const std::vector<uint64_t>& got_keys) {
    ASSERT_EQ(in_keys.size(), got_keys.size());
    std::vector<uint64_t> expected = in_keys;
    std::sort(expected.begin(), expected.end());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], got_keys[i])
            << "permutation mismatch at i=" << i;
    }
}

// Check: for each i, the value-index got_values[i] points back to a slot in
// the original keys whose key equals got_keys[i]. Since values[k] == k on
// input, this reduces to `in_keys[got_values[i]] == got_keys[i]`.
void expect_value_mapping(const std::vector<uint64_t>& in_keys,
                          const SortedResult& got) {
    ASSERT_EQ(got.keys.size(), got.values.size());
    for (size_t i = 0; i < got.keys.size(); ++i) {
        const uint32_t v = got.values[i];
        ASSERT_LT(v, in_keys.size()) << "value index OOB at i=" << i;
        EXPECT_EQ(in_keys[v], got.keys[i])
            << "value mapping broken at i=" << i
            << ": in_keys[" << v << "]=" << in_keys[v]
            << " got_keys[i]=" << got.keys[i];
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// Test 1: single element (degenerate).
// -----------------------------------------------------------------------------
TEST(RadixSortPassVk, SingleElement) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    std::vector<uint64_t> keys   = {0xABCDEF0123456789ull};
    std::vector<uint32_t> values = {0u};

    auto got = run_sort(ctx, keys, values);
    ASSERT_EQ(got.keys.size(), 1u);
    EXPECT_EQ(got.keys[0], keys[0]);
    EXPECT_EQ(got.values[0], 0u);
}

// -----------------------------------------------------------------------------
// Test 2: R=64 random uint64 keys. Deterministic seed for reproducibility.
// -----------------------------------------------------------------------------
TEST(RadixSortPassVk, Random64) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    constexpr uint32_t R = 64;
    std::mt19937_64 rng(0x12345678u);
    std::vector<uint64_t> keys(R);
    std::vector<uint32_t> values(R);
    for (uint32_t i = 0; i < R; ++i) {
        keys[i]   = rng();
        values[i] = i;  // identity mapping — we verify it below
    }

    auto got = run_sort(ctx, keys, values);
    expect_monotonic(got.keys);
    expect_permutation(keys, got.keys);
    expect_value_mapping(keys, got);
}

// -----------------------------------------------------------------------------
// Test 3: R=103 (tiny fixture size — realistic SP-2 Phase-1 upper bound).
// -----------------------------------------------------------------------------
TEST(RadixSortPassVk, TinyFixtureSize103) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    constexpr uint32_t R = 103;
    std::mt19937_64 rng(0xDEADBEEFull);
    std::vector<uint64_t> keys(R);
    std::vector<uint32_t> values(R);
    for (uint32_t i = 0; i < R; ++i) {
        keys[i]   = rng();
        values[i] = i;
    }

    auto got = run_sort(ctx, keys, values);
    expect_monotonic(got.keys);
    expect_permutation(keys, got.keys);
    expect_value_mapping(keys, got);
}

// -----------------------------------------------------------------------------
// Test 4: tile collisions — stable ordering by depth within same tile.
//
// Build keys where the high 32 bits encode a tile_id and the low 32 bits
// encode a depth. The "sort by tile, then by depth" property of LSD radix
// on a uint64 cat-key is exactly what the SP-2 forward pipeline relies on.
// Input is constructed with multiple tile_ids that each have multiple
// Gaussians, interleaved, so the stability test is non-trivial.
// -----------------------------------------------------------------------------
TEST(RadixSortPassVk, StabilityTileCollisions) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    // Build 8 pairs across 3 tiles: tile 2 has 3 gaussians, tile 0 has 3,
    // tile 5 has 2. Interleave insertion order so correctness requires
    // inter-bucket swap across passes.
    auto make_key = [](uint32_t tile, uint32_t depth) -> uint64_t {
        return (static_cast<uint64_t>(tile) << 32) | static_cast<uint64_t>(depth);
    };

    std::vector<uint64_t> keys = {
        make_key(2, 100),
        make_key(5, 200),
        make_key(0,  50),
        make_key(2,  30),
        make_key(0, 999),
        make_key(5,  10),
        make_key(2, 500),
        make_key(0,  75),
    };
    std::vector<uint32_t> values(keys.size());
    for (uint32_t i = 0; i < keys.size(); ++i) values[i] = i;

    auto got = run_sort(ctx, keys, values);

    // Expected sort-by-(tile, depth):
    //   tile 0: depths 50, 75, 999   (orig indices 2, 7, 4)
    //   tile 2: depths 30, 100, 500  (orig indices 3, 0, 6)
    //   tile 5: depths 10, 200       (orig indices 5, 1)
    const std::vector<uint64_t> expected_keys = {
        make_key(0,  50), make_key(0,  75), make_key(0, 999),
        make_key(2,  30), make_key(2, 100), make_key(2, 500),
        make_key(5,  10), make_key(5, 200),
    };
    const std::vector<uint32_t> expected_vals = {2, 7, 4, 3, 0, 6, 5, 1};

    ASSERT_EQ(got.keys.size(),   expected_keys.size());
    ASSERT_EQ(got.values.size(), expected_vals.size());
    for (size_t i = 0; i < expected_keys.size(); ++i) {
        EXPECT_EQ(got.keys[i],   expected_keys[i]) << "key mismatch at i=" << i;
        EXPECT_EQ(got.values[i], expected_vals[i]) << "val mismatch at i=" << i;
    }
    expect_monotonic(got.keys);
}

// -----------------------------------------------------------------------------
// Test 5: R=256 exact workgroup size boundary.
// -----------------------------------------------------------------------------
TEST(RadixSortPassVk, MaxCapacity256) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    constexpr uint32_t R = 256;
    std::mt19937_64 rng(42u);
    std::vector<uint64_t> keys(R);
    std::vector<uint32_t> values(R);
    for (uint32_t i = 0; i < R; ++i) {
        keys[i]   = rng();
        values[i] = i;
    }

    auto got = run_sort(ctx, keys, values);
    expect_monotonic(got.keys);
    expect_permutation(keys, got.keys);
    expect_value_mapping(keys, got);
}

TEST(RadixSortPassVk, MultiWorkgroup1025PreservesFullValuesAndLowDepthBits) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    auto make_key = [](uint32_t tile, uint32_t depth) -> uint64_t {
        return (static_cast<uint64_t>(tile) << 32u) | depth;
    };

    constexpr uint32_t R = 1025;
    std::vector<uint64_t> keys(R);
    std::vector<uint32_t> values(R);
    for (uint32_t i = 0; i < R; ++i) {
        const uint32_t tile = (R - 1u - i) % 17u;
        const uint32_t depth = ((i * 37u) & 0xFFFFFFF0u) | (i & 0xFu);
        keys[i] = make_key(tile, depth);
        values[i] = (1u << 20) + i;
    }

    auto got = run_sort(ctx, keys, values);
    std::vector<uint32_t> order(R);
    for (uint32_t i = 0; i < R; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return keys[a] < keys[b];
    });

    for (uint32_t i = 0; i < R; ++i) {
        EXPECT_EQ(got.keys[i], keys[order[i]]) << "key mismatch at i=" << i;
        EXPECT_EQ(got.values[i], values[order[i]]) << "value mismatch at i=" << i;
    }
    expect_monotonic(got.keys);
}
