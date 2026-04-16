#include <gtest/gtest.h>
#include "types.h"
#include "cpu/sorter_cpu.h"
#include <cstring>

TEST(SorterCPU, BasicSort) {
    FrameAllocator alloc(1024 * 1024);
    BinningOutput bin{};
    bin.total_pairs = 3;
    bin.num_tiles = 2;

    // Keys: tile1|depth3, tile0|depth1, tile0|depth2
    uint32_t d1 = 1, d2 = 2, d3 = 3; // Using ints as fake depth bits
    uint64_t keys[3] = {
        (1ULL << 32) | d3,  // tile 1
        (0ULL << 32) | d1,  // tile 0
        (0ULL << 32) | d2   // tile 0
    };
    uint32_t vals[3] = {0, 1, 2};
    bin.keys_unsorted = keys;
    bin.values_unsorted = vals;
    bin.keys_sorted = alloc.allocate_array<uint64_t>(3);
    bin.values_sorted = alloc.allocate_array<uint32_t>(3);
    bin.tile_ranges = alloc.allocate_array<uint32_t>(4); // 2 tiles * 2

    SorterCPU sorter;
    sorter.sort(bin, alloc);

    // Sorted: tile0|d1, tile0|d2, tile1|d3
    EXPECT_EQ(bin.keys_sorted[0] >> 32, 0u);
    EXPECT_EQ(bin.keys_sorted[1] >> 32, 0u);
    EXPECT_EQ(bin.keys_sorted[2] >> 32, 1u);
    EXPECT_EQ(bin.values_sorted[0], 1u); // was index 1
    EXPECT_EQ(bin.values_sorted[1], 2u); // was index 2
    EXPECT_EQ(bin.values_sorted[2], 0u); // was index 0

    // Tile ranges
    EXPECT_EQ(bin.tile_ranges[0], 0u);  // tile 0 start
    EXPECT_EQ(bin.tile_ranges[1], 2u);  // tile 0 end
    EXPECT_EQ(bin.tile_ranges[2], 2u);  // tile 1 start
    EXPECT_EQ(bin.tile_ranges[3], 3u);  // tile 1 end
}

TEST(SorterCPU, EmptyInput) {
    FrameAllocator alloc(1024 * 1024);
    BinningOutput bin{};
    bin.total_pairs = 0;
    bin.num_tiles = 1;
    bin.tile_ranges = alloc.allocate_array<uint32_t>(2);

    SorterCPU sorter;
    sorter.sort(bin, alloc);
    EXPECT_EQ(bin.tile_ranges[0], 0u);
    EXPECT_EQ(bin.tile_ranges[1], 0u);
}

TEST(SorterCPU, SingleEntry) {
    FrameAllocator alloc(1024 * 1024);
    BinningOutput bin{};
    bin.total_pairs = 1;
    bin.num_tiles = 1;
    uint64_t keys[1] = {(0ULL << 32) | 42};
    uint32_t vals[1] = {7};
    bin.keys_unsorted = keys;
    bin.values_unsorted = vals;
    bin.keys_sorted = alloc.allocate_array<uint64_t>(1);
    bin.values_sorted = alloc.allocate_array<uint32_t>(1);
    bin.tile_ranges = alloc.allocate_array<uint32_t>(2);

    SorterCPU sorter;
    sorter.sort(bin, alloc);
    EXPECT_EQ(bin.keys_sorted[0], keys[0]);
    EXPECT_EQ(bin.values_sorted[0], 7u);
    EXPECT_EQ(bin.tile_ranges[0], 0u);
    EXPECT_EQ(bin.tile_ranges[1], 1u);
}

TEST(SorterCPU, EmptyTileInMiddle) {
    FrameAllocator alloc(1024 * 1024);
    BinningOutput bin{};
    bin.total_pairs = 2;
    bin.num_tiles = 3; // tiles 0, 1, 2 — tile 1 is empty
    uint64_t keys[2] = {
        (0ULL << 32) | 1,
        (2ULL << 32) | 2
    };
    uint32_t vals[2] = {0, 1};
    bin.keys_unsorted = keys;
    bin.values_unsorted = vals;
    bin.keys_sorted = alloc.allocate_array<uint64_t>(2);
    bin.values_sorted = alloc.allocate_array<uint32_t>(2);
    bin.tile_ranges = alloc.allocate_array<uint32_t>(6); // 3 tiles * 2

    SorterCPU sorter;
    sorter.sort(bin, alloc);
    EXPECT_EQ(bin.tile_ranges[0], 0u);  // tile 0 start
    EXPECT_EQ(bin.tile_ranges[1], 1u);  // tile 0 end
    EXPECT_EQ(bin.tile_ranges[2], 0u);  // tile 1 start (empty)
    EXPECT_EQ(bin.tile_ranges[3], 0u);  // tile 1 end (empty)
    EXPECT_EQ(bin.tile_ranges[4], 1u);  // tile 2 start
    EXPECT_EQ(bin.tile_ranges[5], 2u);  // tile 2 end
}
