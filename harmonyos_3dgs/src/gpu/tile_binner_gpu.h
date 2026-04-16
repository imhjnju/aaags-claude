// tile_binner_gpu.h -- OpenCL GPU tile binner for 3DGS
// Implements the TileBinner interface using GPU prefix sum + scatter kernel.
// No CPU readback of keys/values -- they stay on GPU for the sorter.

#pragma once

#include "tile_binner.h"
#include "gpu/opencl_context.h"

/// Device-side buffers for binning output.
/// Passed via BinningOutput::device_data so downstream GPU stages
/// can consume them without readback.
struct BinningDeviceBuffers {
    cl_mem keys_unsorted;    // [total_pairs] uint64
    cl_mem values_unsorted;  // [total_pairs] uint32
    cl_mem keys_sorted;      // [total_pairs] uint64
    cl_mem values_sorted;    // [total_pairs] uint32
    cl_mem tile_ranges;      // [num_tiles * 2] uint32
    cl_mem point_offsets;    // [N] int
};

class TileBinnerGPU : public TileBinner {
public:
    explicit TileBinnerGPU(OpenCLContext& ctx);
    ~TileBinnerGPU() override;

    // Not copyable/movable (owns CL resources)
    TileBinnerGPU(const TileBinnerGPU&) = delete;
    TileBinnerGPU& operator=(const TileBinnerGPU&) = delete;

    BinningOutput bin(
        const PreprocessOutput& preprocess,
        int num_gaussians,
        const Camera& camera,
        const RenderConfig& config,
        FrameAllocator& allocator) override;

private:
    OpenCLContext& ctx_;
    cl_kernel scatter_kernel_ = nullptr;

    // Prefix sum kernels
    cl_kernel scan_blocks_kernel_ = nullptr;
    cl_kernel add_block_sums_kernel_ = nullptr;

    // Persistent device buffers (reallocated if sizes change)
    BinningDeviceBuffers dev_bufs_{};
    int last_total_pairs_ = 0;
    int last_num_tiles_ = 0;
    int last_N_ = 0;

    // Prefix sum intermediate buffers
    cl_mem d_block_sums_ = nullptr;
    cl_mem d_block_sums_scanned_ = nullptr;
    cl_mem d_block_sums_l2_ = nullptr;
    int last_num_blocks_ = 0;

    void allocBuffers(int total_pairs, int num_tiles, int N);
    void releaseBuffers();
    void allocPrefixSumBuffers(int num_blocks);
    void releasePrefixSumBuffers();

    /// Run GPU prefix sum on d_input -> d_output, return total sum
    int gpuPrefixSum(cl_mem d_input, cl_mem d_output, int N);
};
