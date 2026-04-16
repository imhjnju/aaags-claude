// sorter_gpu.h -- OpenCL GPU sorter for 3DGS
// Implements the Sorter interface using optimized 4-bit radix sort:
// - 16 passes of 4-bit radix sort on 64-bit keys (subgroup shuffle_up)
// - Replaces counting sort + bitonic with single global radix sort
// - identify_tile_ranges kernel extracts per-tile start/end
// No CPU readback in the sort path (only tile_ranges at end).

#pragma once

#include "sorter.h"
#include "gpu/opencl_context.h"
#include <vector>

class SorterGPU : public Sorter {
public:
    explicit SorterGPU(OpenCLContext& ctx);
    ~SorterGPU() override;

    // Not copyable/movable
    SorterGPU(const SorterGPU&) = delete;
    SorterGPU& operator=(const SorterGPU&) = delete;

    void sort(BinningOutput& binning, FrameAllocator& allocator) override;

private:
    OpenCLContext& ctx_;

    // --- 4-bit radix sort kernels (subgroup shuffle_up) ---
    cl_kernel rs_histogram_kernel_       = nullptr;
    cl_kernel rs_compute_offsets_kernel_ = nullptr;
    cl_kernel rs_scatter_kv_kernel_      = nullptr;

    // --- Tile range identification (reused from radix_sort.cl) ---
    cl_kernel tile_ranges_kernel_        = nullptr;

    // --- Radix sort buffers ---
    cl_mem d_keys_ping_   = nullptr;   // [capacity] ulong
    cl_mem d_keys_pong_   = nullptr;
    cl_mem d_values_ping_ = nullptr;   // [capacity] uint
    cl_mem d_values_pong_ = nullptr;

    // Per-tile histograms and offsets for 4-bit radix sort
    cl_mem d_rs_tile_hists_   = nullptr;  // [num_rs_tiles * 16] uint
    cl_mem d_rs_tile_offsets_ = nullptr;  // [num_rs_tiles * 16] uint

    int last_capacity_     = 0;
    int last_rs_num_tiles_ = 0;

    // Constants for 4-bit radix sort
    static constexpr int RS_RADIX_BITS    = 4;
    static constexpr int RS_RADIX_BUCKETS = 16;
    static constexpr int RS_NUM_PASSES    = 16;   // 64 / 4
    static constexpr int RS_WG_SIZE       = 256;
    static constexpr int RS_ELEMS_PER_WG  = 32768; // 128 elements/thread

    void allocBuffers(int total_pairs);
    void releaseBuffers();
};
