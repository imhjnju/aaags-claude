// Adapted from RadixSort/radix_sort.cl for 3DGS integration
// 4-bit radix sort with subgroup shuffle_up, ulong keys + uint values
// 16 passes for full 64-bit key sort
static const char radix_sort_kv_cl_src[] = R"CL(
// =============================================================================
// GPU Radix Sort - 4-bit radix, subgroup shuffle_up
// Adapted for 3DGS: ulong keys (tile_id << 32 | depth) + uint values (indices)
// 16 passes for full 64-bit sort
// =============================================================================
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable

#define RADIX_BITS    4
#define RADIX_BUCKETS 16
#define RADIX_MASK    0xFU
#define RS_WG_SIZE    256
#define RS_SG_SIZE    32
#define RS_NUM_SGS    (RS_WG_SIZE / RS_SG_SIZE)
#define RS_ELEMS_PER_WG  32768  // 128 elements/thread

// ---------------------------------------------------------------------------
// Kernel 1: Per-tile histograms (keys are ulong)
// ---------------------------------------------------------------------------
__kernel void rs_histogram(
    __global const ulong* restrict keys,
    __global uint*        restrict tile_hists,
    const uint n,
    const uint shift,
    const uint num_tiles
) {
    const uint lid = get_local_id(0);
    const uint tile_id = get_group_id(0);
    const uint base = tile_id * RS_ELEMS_PER_WG;

    __local uint local_hist[RADIX_BUCKETS];
    if (lid < RADIX_BUCKETS) local_hist[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint end = min(base + RS_ELEMS_PER_WG, n);
    for (uint idx = base + lid; idx < end; idx += RS_WG_SIZE) {
        uint digit = (uint)((keys[idx] >> shift) & RADIX_MASK);
        atomic_add(&local_hist[digit], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BUCKETS)
        tile_hists[lid * num_tiles + tile_id] = local_hist[lid];
}

// ---------------------------------------------------------------------------
// Kernel 2: Compute per-tile offsets (global prefix sum)
// ---------------------------------------------------------------------------
__kernel void rs_compute_offsets(
    __global const uint* restrict tile_hists,
    __global uint*       restrict tile_offsets,
    const uint num_tiles
) {
    const uint lid = get_local_id(0);

    __local uint global_prefix[RADIX_BUCKETS];
    __local uint global_count[RADIX_BUCKETS];

    if (lid < RADIX_BUCKETS) {
        __global const uint* my_hists = tile_hists + lid * num_tiles;
        uint my_total = 0;
        for (uint t = 0; t < num_tiles; t++)
            my_total += my_hists[t];
        global_count[lid] = my_total;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint sum = 0;
        for (uint d = 0; d < RADIX_BUCKETS; d++) {
            global_prefix[d] = sum;
            sum += global_count[d];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BUCKETS) {
        __global const uint* my_hists = tile_hists + lid * num_tiles;
        uint running = global_prefix[lid];
        for (uint t = 0; t < num_tiles; t++) {
            tile_offsets[t * RADIX_BUCKETS + lid] = running;
            running += my_hists[t];
        }
    }
}

// ---------------------------------------------------------------------------
// Subgroup inclusive prefix sum via shuffle_up
// ---------------------------------------------------------------------------
inline uint sg_inclusive_scan_kv(uint val) {
    uint sg_lid = get_sub_group_local_id();
    for (uint s = 1; s < RS_SG_SIZE; s <<= 1) {
        uint tmp = sub_group_shuffle_up(val, s);
        if (sg_lid >= s) val += tmp;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Kernel 3: Scatter with key-value pairs
// Uses subgroup shuffle_up for stable ranking
// ---------------------------------------------------------------------------
__kernel void rs_scatter_kv(
    __global const ulong* restrict keys_in,
    __global ulong*       restrict keys_out,
    __global const uint*  restrict values_in,
    __global uint*        restrict values_out,
    __global const uint*  restrict tile_offsets,
    const uint n,
    const uint shift
) {
    const uint lid = get_local_id(0);
    const uint tile_id = get_group_id(0);
    const uint base = tile_id * RS_ELEMS_PER_WG;

    const uint sg_id  = lid / RS_SG_SIZE;
    const uint sg_lid = lid % RS_SG_SIZE;

    __local uint offsets[RADIX_BUCKETS];
    __local uint sg_totals[RADIX_BUCKETS * RS_NUM_SGS];
    __local uint sub_count[RADIX_BUCKETS];

    if (lid < RADIX_BUCKETS)
        offsets[lid] = tile_offsets[tile_id * RADIX_BUCKETS + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tile_end = min(base + RS_ELEMS_PER_WG, n);

    for (uint sub_base = base; sub_base < tile_end; sub_base += RS_WG_SIZE) {
        uint idx = sub_base + lid;
        bool valid = (idx < tile_end);

        ulong key = valid ? keys_in[idx] : 0;
        uint  val = valid ? values_in[idx] : 0;
        uint digit = valid ? (uint)((key >> shift) & RADIX_MASK) : 0xFFu;

        // Phase 1: Intra-subgroup rank via shuffle
        uint my_intra_rank = 0;

        for (uint d = 0; d < RADIX_BUCKETS; d++) {
            uint flag = (valid && digit == d) ? 1u : 0u;
            uint scan = sg_inclusive_scan_kv(flag);
            uint sg_total = sub_group_shuffle_down(scan, RS_SG_SIZE - 1 - sg_lid);

            if (sg_lid == 0)
                sg_totals[d * RS_NUM_SGS + sg_id] = sg_total;

            if (digit == d)
                my_intra_rank = scan - flag;
        }

        // Phase 2: Inter-subgroup offset
        barrier(CLK_LOCAL_MEM_FENCE);

        uint inter_offset = 0;
        if (valid) {
            for (uint s = 0; s < sg_id; s++)
                inter_offset += sg_totals[digit * RS_NUM_SGS + s];
        }

        // Phase 3: Write output (key + value)
        if (valid) {
            uint dst = offsets[digit] + inter_offset + my_intra_rank;
            keys_out[dst] = key;
            values_out[dst] = val;
        }

        // Phase 4: Advance offsets
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < RADIX_BUCKETS) sub_count[lid] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (valid) atomic_add(&sub_count[digit], 1);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < RADIX_BUCKETS) offsets[lid] += sub_count[lid];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)CL";
