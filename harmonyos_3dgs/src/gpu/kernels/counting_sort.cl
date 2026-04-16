// counting_sort.cl -- GPU counting sort by tile_id (upper 32 bits of key)
//
// Three-phase approach:
// 1. count_per_tile: count elements per tile using atomics
// 2. prefix_sum_small: single-thread exclusive prefix sum (~2700 tiles)
// 3. scatter_by_tile: scatter elements to sorted positions using atomic counters

// Phase 1: Count elements per tile
__kernel void count_per_tile(
    __global const ulong* keys,    // [total_pairs] — (tile_id << 32 | depth)
    __global int* tile_counts,     // [num_tiles] — output
    const int total_pairs,
    const int num_tiles
) {
    int idx = get_global_id(0);
    if (idx >= total_pairs) return;
    uint tile_id = (uint)(keys[idx] >> 32);
    if (tile_id < (uint)num_tiles)
        atomic_add(&tile_counts[tile_id], 1);
}

// Phase 2: Exclusive prefix sum on tile_counts (single work-item, ~2700 elements)
__kernel void prefix_sum_tiles(
    __global const int* counts,    // [N] input counts
    __global int* offsets,         // [N] output exclusive prefix sum
    const int N
) {
    if (get_global_id(0) != 0) return;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        offsets[i] = sum;
        sum += counts[i];
    }
}

// Phase 3: Scatter by tile_id using atomic per-tile counters
__kernel void scatter_by_tile(
    __global const ulong* keys_in,
    __global const uint* values_in,
    __global ulong* keys_out,
    __global uint* values_out,
    __global const int* tile_offsets,   // [num_tiles] exclusive prefix sum
    __global int* tile_counters,        // [num_tiles] atomic counters (init to 0)
    const int total_pairs,
    const int num_tiles
) {
    int idx = get_global_id(0);
    if (idx >= total_pairs) return;
    ulong key = keys_in[idx];
    uint tile_id = (uint)(key >> 32);
    if (tile_id < (uint)num_tiles) {
        int pos = tile_offsets[tile_id] + atomic_add(&tile_counters[tile_id], 1);
        keys_out[pos] = key;
        values_out[pos] = values_in[idx];
    }
}
