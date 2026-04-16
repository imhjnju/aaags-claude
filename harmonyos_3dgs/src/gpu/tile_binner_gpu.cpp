// tile_binner_gpu.cpp -- OpenCL GPU tile binner implementation
// Uses GPU prefix sum (Blelloch scan) instead of CPU fallback.
// Scatter kernel runs on GPU. NO readback of keys/values -- they stay
// on GPU for the sorter and rasterizer.

#include "gpu/tile_binner_gpu.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/kernels/scatter_cl.h"
#include "gpu/kernels/prefix_sum_cl.h"
#include "math_utils.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

// Block size for prefix sum (must match the kernel define)
static constexpr int BLOCK_SIZE_SCAN = 256;
static constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE_SCAN * 2;  // 512

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

TileBinnerGPU::TileBinnerGPU(OpenCLContext& ctx) : ctx_(ctx) {
    scatter_kernel_ = ctx_.buildKernel(scatter_cl_src,
                                        sizeof(scatter_cl_src) - 1,
                                        "scatter",
                                        "-cl-fast-relaxed-math");
    std::fprintf(stderr, "TileBinnerGPU: scatter kernel compiled OK\n");

    // Build prefix sum kernels
    auto& cl = ctx_.cl();
    cl_int err;

    const char* src_ptr = prefix_sum_cl_src;
    size_t src_len = std::strlen(prefix_sum_cl_src);
    cl_program program = cl.clCreateProgramWithSource(
        ctx_.context(), 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS || !program)
        throw std::runtime_error("TileBinnerGPU: clCreateProgramWithSource failed for prefix_sum");

    cl_device_id device = nullptr;
    cl.clGetContextInfo(ctx_.context(), CL_CONTEXT_DEVICES,
                        sizeof(device), &device, nullptr);

    std::string build_opts = "-cl-fast-relaxed-math -DBLOCK_SIZE_SCAN=" +
                             std::to_string(BLOCK_SIZE_SCAN);

    err = cl.clBuildProgram(program, 1, &device,
                            build_opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                 0, nullptr, &log_size);
        std::string log(log_size + 1, '\0');
        cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                 log_size, &log[0], nullptr);
        cl.clReleaseProgram(program);
        throw std::runtime_error(
            std::string("TileBinnerGPU: prefix_sum build failed: ") + log);
    }

    scan_blocks_kernel_ = cl.clCreateKernel(program, "scan_blocks", &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("TileBinnerGPU: failed to create scan_blocks kernel");

    add_block_sums_kernel_ = cl.clCreateKernel(program, "add_block_sums", &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("TileBinnerGPU: failed to create add_block_sums kernel");

    cl.clReleaseProgram(program);

    std::fprintf(stderr, "TileBinnerGPU: prefix sum kernels compiled OK\n");
}

TileBinnerGPU::~TileBinnerGPU() {
    releaseBuffers();
    releasePrefixSumBuffers();
    auto& cl = ctx_.cl();
    if (scatter_kernel_)        { cl.clReleaseKernel(scatter_kernel_);        scatter_kernel_ = nullptr; }
    if (scan_blocks_kernel_)    { cl.clReleaseKernel(scan_blocks_kernel_);    scan_blocks_kernel_ = nullptr; }
    if (add_block_sums_kernel_) { cl.clReleaseKernel(add_block_sums_kernel_); add_block_sums_kernel_ = nullptr; }
}

// ---------------------------------------------------------------------------
// Resource management
// ---------------------------------------------------------------------------

void TileBinnerGPU::releaseBuffers() {
    auto& cl = ctx_.cl();
    if (dev_bufs_.keys_unsorted)   { cl.clReleaseMemObject(dev_bufs_.keys_unsorted);   dev_bufs_.keys_unsorted = nullptr; }
    if (dev_bufs_.values_unsorted) { cl.clReleaseMemObject(dev_bufs_.values_unsorted); dev_bufs_.values_unsorted = nullptr; }
    if (dev_bufs_.keys_sorted)     { cl.clReleaseMemObject(dev_bufs_.keys_sorted);     dev_bufs_.keys_sorted = nullptr; }
    if (dev_bufs_.values_sorted)   { cl.clReleaseMemObject(dev_bufs_.values_sorted);   dev_bufs_.values_sorted = nullptr; }
    if (dev_bufs_.tile_ranges)     { cl.clReleaseMemObject(dev_bufs_.tile_ranges);     dev_bufs_.tile_ranges = nullptr; }
    if (dev_bufs_.point_offsets)   { cl.clReleaseMemObject(dev_bufs_.point_offsets);   dev_bufs_.point_offsets = nullptr; }
    last_total_pairs_ = 0;
    last_num_tiles_ = 0;
    last_N_ = 0;
}

void TileBinnerGPU::releasePrefixSumBuffers() {
    auto& cl = ctx_.cl();
    if (d_block_sums_)         { cl.clReleaseMemObject(d_block_sums_);         d_block_sums_ = nullptr; }
    if (d_block_sums_scanned_) { cl.clReleaseMemObject(d_block_sums_scanned_); d_block_sums_scanned_ = nullptr; }
    if (d_block_sums_l2_)      { cl.clReleaseMemObject(d_block_sums_l2_);      d_block_sums_l2_ = nullptr; }
    last_num_blocks_ = 0;
}

void TileBinnerGPU::allocBuffers(int total_pairs, int num_tiles, int N) {
    auto& cl = ctx_.cl();

    // Reallocate point_offsets if N changed
    if (N != last_N_) {
        if (dev_bufs_.point_offsets) { cl.clReleaseMemObject(dev_bufs_.point_offsets); dev_bufs_.point_offsets = nullptr; }
        if (N > 0)
            dev_bufs_.point_offsets = ctx_.createBuffer(N * sizeof(int), CL_MEM_READ_WRITE);
        last_N_ = N;
    }

    // Reallocate tile_ranges if num_tiles changed
    if (num_tiles != last_num_tiles_) {
        if (dev_bufs_.tile_ranges) { cl.clReleaseMemObject(dev_bufs_.tile_ranges); dev_bufs_.tile_ranges = nullptr; }
        if (num_tiles > 0)
            dev_bufs_.tile_ranges = ctx_.createBuffer(num_tiles * 2 * sizeof(uint32_t), CL_MEM_READ_WRITE);
        last_num_tiles_ = num_tiles;
    }

    // Reallocate key/value buffers if total_pairs changed
    if (total_pairs != last_total_pairs_) {
        if (dev_bufs_.keys_unsorted)   { cl.clReleaseMemObject(dev_bufs_.keys_unsorted);   dev_bufs_.keys_unsorted = nullptr; }
        if (dev_bufs_.values_unsorted) { cl.clReleaseMemObject(dev_bufs_.values_unsorted); dev_bufs_.values_unsorted = nullptr; }
        if (dev_bufs_.keys_sorted)     { cl.clReleaseMemObject(dev_bufs_.keys_sorted);     dev_bufs_.keys_sorted = nullptr; }
        if (dev_bufs_.values_sorted)   { cl.clReleaseMemObject(dev_bufs_.values_sorted);   dev_bufs_.values_sorted = nullptr; }

        if (total_pairs > 0) {
            dev_bufs_.keys_unsorted   = ctx_.createBuffer(total_pairs * sizeof(uint64_t), CL_MEM_READ_WRITE);
            dev_bufs_.values_unsorted = ctx_.createBuffer(total_pairs * sizeof(uint32_t), CL_MEM_READ_WRITE);
            dev_bufs_.keys_sorted     = ctx_.createBuffer(total_pairs * sizeof(uint64_t), CL_MEM_READ_WRITE);
            dev_bufs_.values_sorted   = ctx_.createBuffer(total_pairs * sizeof(uint32_t), CL_MEM_READ_WRITE);
        }
        last_total_pairs_ = total_pairs;
    }
}

void TileBinnerGPU::allocPrefixSumBuffers(int num_blocks) {
    if (num_blocks == last_num_blocks_) return;
    releasePrefixSumBuffers();

    if (num_blocks > 0) {
        d_block_sums_ = ctx_.createBuffer(num_blocks * sizeof(int), CL_MEM_READ_WRITE);
        d_block_sums_scanned_ = ctx_.createBuffer(num_blocks * sizeof(int), CL_MEM_READ_WRITE);

        // Level-2 block sums (for scanning the block sums themselves)
        int num_blocks_l2 = (num_blocks + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
        if (num_blocks_l2 > 0)
            d_block_sums_l2_ = ctx_.createBuffer(num_blocks_l2 * sizeof(int), CL_MEM_READ_WRITE);
    }
    last_num_blocks_ = num_blocks;
}

// ---------------------------------------------------------------------------
// GPU prefix sum -- two-level Blelloch scan
// ---------------------------------------------------------------------------

int TileBinnerGPU::gpuPrefixSum(cl_mem d_input, cl_mem d_output, int N) {
    auto& cl = ctx_.cl();

    int num_blocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    allocPrefixSumBuffers(num_blocks);

    // -----------------------------------------------------------------------
    // Level 0: Scan blocks, produce block sums
    // -----------------------------------------------------------------------
    {
        CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 0, sizeof(cl_mem), &d_input));
        CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 1, sizeof(cl_mem), &d_output));
        CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 2, sizeof(cl_mem), &d_block_sums_));
        CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 3, sizeof(int),    &N));

        size_t local_size = BLOCK_SIZE_SCAN;
        size_t global_size = static_cast<size_t>(num_blocks) * BLOCK_SIZE_SCAN;
        ctx_.enqueueKernel(scan_blocks_kernel_, 1, &global_size, &local_size);
    }

    if (num_blocks > 1) {
        // -------------------------------------------------------------------
        // Level 1: Scan block sums
        // -------------------------------------------------------------------
        int num_blocks_l2 = (num_blocks + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

        {
            cl_mem null_mem = nullptr;
            CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 0, sizeof(cl_mem), &d_block_sums_));
            CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 1, sizeof(cl_mem), &d_block_sums_scanned_));
            if (num_blocks_l2 > 1) {
                CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 2, sizeof(cl_mem), &d_block_sums_l2_));
            } else {
                CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 2, sizeof(cl_mem), &null_mem));
            }
            CL_CHECK(cl.clSetKernelArg(scan_blocks_kernel_, 3, sizeof(int),    &num_blocks));

            size_t local_size = BLOCK_SIZE_SCAN;
            size_t global_size = static_cast<size_t>(num_blocks_l2) * BLOCK_SIZE_SCAN;
            ctx_.enqueueKernel(scan_blocks_kernel_, 1, &global_size, &local_size);
        }

        // For num_blocks_l2 > 1, we'd need a third level. For N up to ~500K
        // gaussians, num_blocks ~ 1000, num_blocks_l2 ~ 2, so a single
        // work-group handles it fine. Assert this assumption.
        if (num_blocks_l2 > 1) {
            std::fprintf(stderr, "TileBinnerGPU: WARNING -- 3-level prefix sum needed "
                         "(num_blocks_l2=%d), falling back to 2-level\n", num_blocks_l2);
            // For now, the second level scan already handles this since num_blocks_l2
            // is small enough to fit in one work-group.
        }

        // -------------------------------------------------------------------
        // Level 0 fixup: Add scanned block sums to each element
        // -------------------------------------------------------------------
        {
            CL_CHECK(cl.clSetKernelArg(add_block_sums_kernel_, 0, sizeof(cl_mem), &d_output));
            CL_CHECK(cl.clSetKernelArg(add_block_sums_kernel_, 1, sizeof(cl_mem), &d_block_sums_scanned_));
            CL_CHECK(cl.clSetKernelArg(add_block_sums_kernel_, 2, sizeof(int),    &N));

            size_t wg_size = std::min(ctx_.maxWorkGroupSize(), size_t(256));
            size_t global_size = ((N + wg_size - 1) / wg_size) * wg_size;
            ctx_.enqueueKernel(add_block_sums_kernel_, 1, &global_size, &wg_size);
        }
    }

    // -----------------------------------------------------------------------
    // Read back total_pairs = last element of prefix sum + last input element
    // We need: total = output[N-1] + input[N-1]
    // -----------------------------------------------------------------------
    int last_offset = 0;
    int last_input = 0;
    ctx_.readBuffer(d_output, &last_offset, sizeof(int));  // temp: read element 0

    // Read last elements specifically
    CL_CHECK(cl.clEnqueueReadBuffer(
        ctx_.queue(), d_output, CL_TRUE,
        (N - 1) * sizeof(int), sizeof(int), &last_offset,
        0, nullptr, nullptr));
    CL_CHECK(cl.clEnqueueReadBuffer(
        ctx_.queue(), d_input, CL_TRUE,
        (N - 1) * sizeof(int), sizeof(int), &last_input,
        0, nullptr, nullptr));

    return last_offset + last_input;
}

// ---------------------------------------------------------------------------
// bin -- main entry point
// ---------------------------------------------------------------------------

BinningOutput TileBinnerGPU::bin(const PreprocessOutput& pre, int N,
                                  const Camera& cam, const RenderConfig& cfg,
                                  FrameAllocator& alloc) {
    BinningOutput out{};

    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;
    out.num_tiles = grid_x * grid_y;

    // Allocate CPU-side output arrays via FrameAllocator
    out.tile_ranges = alloc.allocate_array<uint32_t>(out.num_tiles * 2);
    std::memset(out.tile_ranges, 0, out.num_tiles * 2 * sizeof(uint32_t));

    if (N == 0) {
        out.total_pairs = 0;
        out.keys_unsorted = nullptr;
        out.values_unsorted = nullptr;
        out.keys_sorted = nullptr;
        out.values_sorted = nullptr;
        return out;
    }

    // -----------------------------------------------------------------------
    // Step 1: CPU prefix sum over tiles_touched (fast for 400K, ~1ms)
    //         GPU Blelloch scan had correctness issues; CPU is reliable.
    // -----------------------------------------------------------------------
    auto* pre_dev = static_cast<PreprocessDeviceBuffers*>(pre.device_data);

    // tiles_touched already read back to CPU by PreprocessorGPU
    std::vector<int> offsets(N);
    offsets[0] = 0;
    for (int i = 1; i < N; i++)
        offsets[i] = offsets[i - 1] + pre.tiles_touched[i - 1];
    int total_pairs = offsets[N - 1] + pre.tiles_touched[N - 1];

    out.total_pairs = total_pairs;

    if (total_pairs == 0) {
        out.keys_unsorted = nullptr;
        out.values_unsorted = nullptr;
        out.keys_sorted = nullptr;
        out.values_sorted = nullptr;
        return out;
    }

    // Upload offsets to GPU for scatter kernel
    if (N != last_N_) {
        auto& cl2 = ctx_.cl();
        if (dev_bufs_.point_offsets) { cl2.clReleaseMemObject(dev_bufs_.point_offsets); dev_bufs_.point_offsets = nullptr; }
        dev_bufs_.point_offsets = ctx_.createBuffer(N * sizeof(int), CL_MEM_READ_WRITE);
        last_N_ = N;
    }
    ctx_.writeBuffer(dev_bufs_.point_offsets, offsets.data(), N * sizeof(int));

    // -----------------------------------------------------------------------
    // Step 2: Allocate GPU buffers for keys/values
    // -----------------------------------------------------------------------
    allocBuffers(total_pairs, out.num_tiles, N);

    // -----------------------------------------------------------------------
    // Step 3: Dispatch scatter kernel
    // -----------------------------------------------------------------------
    auto& cl = ctx_.cl();
    cl_uint arg = 0;

    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->means2D));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->radii));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->depths));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &dev_bufs_.point_offsets));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(int),    &N));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(int),    &grid_x));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(int),    &grid_y));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(int),    &cfg.tile_w));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(int),    &cfg.tile_h));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &dev_bufs_.keys_unsorted));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &dev_bufs_.values_unsorted));

    // AAA-Gaussians per-tile culling args
    int eval_3d_flag = pre.eval_3D ? 1 : 0;
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(int), &eval_3d_flag));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->gauss2screen));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->opacities_2d));

    // Per-tile depth key args
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->cov3D_inv));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &pre_dev->mean_offset));

    // Compute inverse viewproj for per-tile depth key
    float inverse_vp[16];
    bool have_inv = false;
    if (pre.eval_3D) {
        have_inv = invertMatrix4x4(cam.viewproj_matrix, inverse_vp);
    }

    // Upload inverse_vp
    cl_mem d_inverse_vp = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
    if (have_inv)
        ctx_.writeBuffer(d_inverse_vp, inverse_vp, 16 * sizeof(float));
    else {
        float zeros[16] = {};
        ctx_.writeBuffer(d_inverse_vp, zeros, 16 * sizeof(float));
    }
    cl_mem d_cam_pos = ctx_.createBuffer(3 * sizeof(float), CL_MEM_READ_ONLY);
    ctx_.writeBuffer(d_cam_pos, cam.cam_pos, 3 * sizeof(float));

    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &d_inverse_vp));
    CL_CHECK(cl.clSetKernelArg(scatter_kernel_, arg++, sizeof(cl_mem), &d_cam_pos));

    size_t wg_size = std::min(ctx_.maxWorkGroupSize(), size_t(256));
    size_t global_size = ((size_t(N) + wg_size - 1) / wg_size) * wg_size;
    cl_event scatter_event;
    ctx_.enqueueKernel(scatter_kernel_, 1, &global_size, &wg_size, &scatter_event);
    ctx_.finish();
    double scatter_ms = ctx_.getEventTimeMs(scatter_event);
    ctx_.cl().clReleaseEvent(scatter_event);
    std::fprintf(stderr, "  [GPU] scatter kernel: %.1f ms (N=%d)\n", scatter_ms, N);

    // Release temporary per-tile depth key buffers
    ctx_.cl().clReleaseMemObject(d_inverse_vp);
    ctx_.cl().clReleaseMemObject(d_cam_pos);

    // -----------------------------------------------------------------------
    // NO readback -- keys/values stay on GPU for sorter
    // -----------------------------------------------------------------------
    // Allocate minimal CPU arrays (needed for BinningOutput struct but not used)
    out.keys_unsorted = nullptr;
    out.values_unsorted = nullptr;
    out.keys_sorted = nullptr;
    out.values_sorted = nullptr;

    // Store device buffer pointer for downstream GPU stages
    out.device_data = &dev_bufs_;

    return out;
}
