// prefix_sum.cl -- Work-efficient parallel exclusive prefix sum (Blelloch scan)
//
// Two-pass approach for arrays larger than one work-group:
// 1. scan_blocks: Each work-group computes an exclusive prefix sum over its
//    block using Blelloch up-sweep/down-sweep in local memory. Block totals
//    are written to a separate array.
// 2. add_block_sums: After scanning block totals (recursively or with a
//    single work-group), add the cumulative block sum to each element.
//
// For the 3DGS prefix sum over tiles_touched (N ~ 200K-500K), two levels
// suffice: level-0 blocks of 512 elements, level-1 has ~1000 block sums
// which fits in a single work-group scan.

#ifndef BLOCK_SIZE_SCAN
#define BLOCK_SIZE_SCAN 256
#endif

// ---------------------------------------------------------------------------
// scan_blocks -- Blelloch exclusive prefix sum within work-groups
// ---------------------------------------------------------------------------
// Each work-group processes 2 * BLOCK_SIZE_SCAN elements (each thread loads 2).
// Output: scanned values in output[], block totals in block_sums[].
// The block_sums array has one entry per work-group.

__kernel void scan_blocks(
    __global const int* input,       // [N]
    __global int*       output,      // [N] exclusive prefix sum
    __global int*       block_sums,  // [num_blocks]
    const int N
)
{
    __local int temp[BLOCK_SIZE_SCAN * 2];

    int lid = get_local_id(0);
    int block_id = get_group_id(0);
    int block_offset = block_id * (BLOCK_SIZE_SCAN * 2);

    // Each thread loads two elements
    int ai = lid;
    int bi = lid + BLOCK_SIZE_SCAN;

    int ga = block_offset + ai;
    int gb = block_offset + bi;

    temp[ai] = (ga < N) ? input[ga] : 0;
    temp[bi] = (gb < N) ? input[gb] : 0;

    // Up-sweep (reduction) phase
    int offset = 1;
    for (int d = BLOCK_SIZE_SCAN; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            int ai2 = offset * (2 * lid + 1) - 1;
            int bi2 = offset * (2 * lid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }

    // Store block total and clear last element for down-sweep
    if (lid == 0) {
        if (block_sums)
            block_sums[block_id] = temp[BLOCK_SIZE_SCAN * 2 - 1];
        temp[BLOCK_SIZE_SCAN * 2 - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < BLOCK_SIZE_SCAN * 2; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            int ai2 = offset * (2 * lid + 1) - 1;
            int bi2 = offset * (2 * lid + 2) - 1;
            int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write results
    if (ga < N) output[ga] = temp[ai];
    if (gb < N) output[gb] = temp[bi];
}

// ---------------------------------------------------------------------------
// add_block_sums -- Add cumulative block sum to each element
// ---------------------------------------------------------------------------
// After scanning block_sums[], add block_sums[block_id] to each element
// in output[] that belongs to that block.

__kernel void add_block_sums(
    __global int*       output,       // [N]
    __global const int* block_sums,   // [num_blocks] -- already scanned
    const int N
)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    int block_id = get_group_id(0);
    // Each work-group in this kernel processes BLOCK_SIZE_SCAN*2 elements
    // but we launch with a different layout: one thread per element.
    // So we compute the original block_id from the element index.
    int orig_block = gid / (BLOCK_SIZE_SCAN * 2);

    output[gid] += block_sums[orig_block];
}
