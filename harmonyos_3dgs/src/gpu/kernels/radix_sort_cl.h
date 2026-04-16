// Auto-generated from radix_sort.cl -- do not edit
static const char radix_sort_cl_src[] = R"CL(
// radix_sort.cl -- tile range identification kernel
//
// Only identify_tile_ranges remains. Legacy radix_histogram, radix_scatter,
// and sort_within_tiles have been removed (replaced by counting_sort.cl
// and tile_sort.cl).

// ---------------------------------------------------------------------------
// Identify tile ranges from sorted keys
// ---------------------------------------------------------------------------
// After sorting, each key is (tile_id << 32 | depth_bits).
// This kernel finds the start/end index for each tile in the sorted array.
// tile_ranges layout: [tile_id * 2] = start, [tile_id * 2 + 1] = end

__kernel void identify_tile_ranges(
    __global const ulong* sorted_keys,
    __global uint*  tile_ranges,   // [num_tiles * 2], pre-zeroed
    const int total_pairs,
    const int num_tiles
)
{
    int idx = get_global_id(0);
    if (idx >= total_pairs) return;

    uint tile_id = (uint)(sorted_keys[idx] >> 32);

    // Guard against out-of-range tile IDs (from sentinel keys)
    if (tile_id >= (uint)num_tiles) return;

    // Check if this is the start of a new tile
    if (idx == 0) {
        tile_ranges[tile_id * 2] = 0;  // start
    } else {
        uint prev_tile = (uint)(sorted_keys[idx - 1] >> 32);
        if (tile_id != prev_tile) {
            tile_ranges[tile_id * 2] = (uint)idx;      // start of current tile
            if (prev_tile < (uint)num_tiles) {
                tile_ranges[prev_tile * 2 + 1] = (uint)idx;  // end of previous tile
            }
        }
    }

    // Check if this is the last element
    if (idx == total_pairs - 1) {
        tile_ranges[tile_id * 2 + 1] = (uint)total_pairs;  // end
    }
}
)CL";
