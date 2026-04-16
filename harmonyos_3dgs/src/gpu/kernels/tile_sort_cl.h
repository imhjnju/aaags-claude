// Auto-generated from tile_sort.cl -- do not edit
static const char tile_sort_cl_src[] = R"CL(
// tile_sort.cl -- In-tile depth sort kernels
//
// Kernel 1: bitonic_sort_tiles
//   One work-group per tile. Sorts tiles <= MAX_TILE_SIZE using local memory
//   bitonic sort. Tiles > MAX_TILE_SIZE are sorted in chunks of MAX_TILE_SIZE.
//
// Kernel 2: merge_sorted_runs
//   Merges sorted runs within each tile. Called repeatedly with doubling
//   run_size until all tiles are fully sorted. Uses two-pointer merge.
//   One work-group per tile, single thread does sequential merge.

#ifndef MAX_TILE_SIZE
#define MAX_TILE_SIZE 4096
#endif

__kernel void bitonic_sort_tiles(
    __global const ulong* keys_in,       // [total_pairs] input
    __global const uint* values_in,      // [total_pairs] input
    __global ulong* keys_out,            // [total_pairs] output
    __global uint* values_out,           // [total_pairs] output
    __global const uint* tile_ranges,    // [num_tiles * 2] (start, end)
    const int num_tiles
) {
    int tile_id = get_group_id(0);
    if (tile_id >= num_tiles) return;

    uint start = tile_ranges[tile_id * 2];
    uint end   = tile_ranges[tile_id * 2 + 1];
    uint count = end - start;
    if (count <= 1) {
        int lid = get_local_id(0);
        if (lid == 0 && count == 1) {
            keys_out[start] = keys_in[start];
            values_out[start] = values_in[start];
        }
        return;
    }

    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    __local uint local_depth[MAX_TILE_SIZE];
    __local ushort local_idx[MAX_TILE_SIZE];

    ulong tile_prefix = ((ulong)tile_id) << 32;

    // Sort in chunks of MAX_TILE_SIZE
    uint num_chunks = (count + MAX_TILE_SIZE - 1) / MAX_TILE_SIZE;

    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        uint chunk_start = chunk * MAX_TILE_SIZE;
        uint chunk_count = min((uint)MAX_TILE_SIZE, count - chunk_start);

        uint padded = 1;
        while (padded < chunk_count) padded <<= 1;

        // Load chunk
        for (uint i = lid; i < chunk_count; i += wg_size) {
            local_depth[i] = (uint)(keys_in[start + chunk_start + i] & 0xFFFFFFFFUL);
            local_idx[i] = (ushort)i;
        }
        for (uint i = chunk_count + lid; i < padded; i += wg_size) {
            local_depth[i] = 0xFFFFFFFF;
            local_idx[i] = (ushort)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Bitonic sort
        for (uint k = 2; k <= padded; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                for (uint i = lid; i < padded / 2; i += wg_size) {
                    uint l = (i / j) * (j * 2) + (i % j);
                    uint r = l + j;
                    bool ascending = ((l & k) == 0);
                    if ((ascending && local_depth[l] > local_depth[r]) ||
                        (!ascending && local_depth[l] < local_depth[r])) {
                        uint td = local_depth[l];
                        local_depth[l] = local_depth[r];
                        local_depth[r] = td;
                        ushort ti = local_idx[l];
                        local_idx[l] = local_idx[r];
                        local_idx[r] = ti;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        // Write sorted chunk
        for (uint i = lid; i < chunk_count; i += wg_size) {
            uint orig = (uint)local_idx[i];
            keys_out[start + chunk_start + i] = tile_prefix | (ulong)local_depth[i];
            values_out[start + chunk_start + i] = values_in[start + chunk_start + orig];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Merge sorted runs of `run_size` elements within each tile.
// Reads from (keys_in, values_in), writes to (keys_out, values_out).
// After each call, run_size doubles. Repeat until run_size >= max_tile_size.
// Single thread per work-group does sequential two-pointer merge (simple, correct).
__kernel void merge_sorted_runs(
    __global const ulong* keys_in,
    __global const uint* values_in,
    __global ulong* keys_out,
    __global uint* values_out,
    __global const uint* tile_ranges,
    const int num_tiles,
    const int run_size          // current sorted run size
) {
    int tile_id = get_global_id(0);
    if (tile_id >= num_tiles) return;

    uint start = tile_ranges[tile_id * 2];
    uint end   = tile_ranges[tile_id * 2 + 1];
    uint count = end - start;

    // If tile fits in a single run, just copy through
    if (count <= (uint)run_size) {
        // Data is already sorted from previous pass; copy in->out
        for (uint i = 0; i < count; i++) {
            keys_out[start + i] = keys_in[start + i];
            values_out[start + i] = values_in[start + i];
        }
        return;
    }

    // Merge pairs of adjacent runs of `run_size`
    uint merge_size = (uint)run_size * 2;
    uint out_pos = 0;

    for (uint base = 0; base < count; base += merge_size) {
        // Left run: [base, mid)
        // Right run: [mid, right_end)
        uint mid = min(base + (uint)run_size, count);
        uint right_end = min(base + merge_size, count);

        uint i = base;
        uint j = mid;

        while (i < mid && j < right_end) {
            uint depth_i = (uint)(keys_in[start + i] & 0xFFFFFFFFUL);
            uint depth_j = (uint)(keys_in[start + j] & 0xFFFFFFFFUL);
            if (depth_i <= depth_j) {
                keys_out[start + out_pos] = keys_in[start + i];
                values_out[start + out_pos] = values_in[start + i];
                i++;
            } else {
                keys_out[start + out_pos] = keys_in[start + j];
                values_out[start + out_pos] = values_in[start + j];
                j++;
            }
            out_pos++;
        }
        while (i < mid) {
            keys_out[start + out_pos] = keys_in[start + i];
            values_out[start + out_pos] = values_in[start + i];
            i++;
            out_pos++;
        }
        while (j < right_end) {
            keys_out[start + out_pos] = keys_in[start + j];
            values_out[start + out_pos] = values_in[start + j];
            j++;
            out_pos++;
        }
    }
}
)CL";
