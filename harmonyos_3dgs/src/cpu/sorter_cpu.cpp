#include "cpu/sorter_cpu.h"
#include <algorithm>
#include <numeric>
#include <cstring>

void SorterCPU::sort(BinningOutput& bin, FrameAllocator& alloc) {
    // Initialize tile_ranges to zeros
    memset(bin.tile_ranges, 0, bin.num_tiles * 2 * sizeof(uint32_t));

    if (bin.total_pairs == 0)
        return;

    // Create index array from FrameAllocator (avoid heap alloc)
    uint32_t* indices = alloc.allocate_array<uint32_t>(bin.total_pairs);
    std::iota(indices, indices + bin.total_pairs, 0u);

    // Sort indices by key
    std::stable_sort(indices, indices + bin.total_pairs, [&](uint32_t a, uint32_t b) {
        return bin.keys_unsorted[a] < bin.keys_unsorted[b];
    });

    // Copy sorted results
    for (int i = 0; i < bin.total_pairs; i++) {
        bin.keys_sorted[i] = bin.keys_unsorted[indices[i]];
        bin.values_sorted[i] = bin.values_unsorted[indices[i]];
    }

    // Identify tile ranges (from CUDA identifyTileRanges kernel)
    for (int i = 0; i < bin.total_pairs; i++) {
        uint32_t cur_tile = (uint32_t)(bin.keys_sorted[i] >> 32);
        if (i == 0) {
            bin.tile_ranges[cur_tile * 2] = 0;
        } else {
            uint32_t prev_tile = (uint32_t)(bin.keys_sorted[i - 1] >> 32);
            if (cur_tile != prev_tile) {
                bin.tile_ranges[prev_tile * 2 + 1] = (uint32_t)i;
                bin.tile_ranges[cur_tile * 2] = (uint32_t)i;
            }
        }
        if (i == bin.total_pairs - 1) {
            bin.tile_ranges[cur_tile * 2 + 1] = (uint32_t)bin.total_pairs;
        }
    }
}
