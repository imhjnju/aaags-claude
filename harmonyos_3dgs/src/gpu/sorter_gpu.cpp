// sorter_gpu.cpp -- 4-bit radix sort (subgroup shuffle_up) for 3DGS
//
// Pipeline (zero CPU readback during sort):
// 1. 16 passes of 4-bit radix sort on 64-bit keys:
//    histogram → compute_offsets → scatter_kv (with values carried along)
// 2. identify_tile_ranges kernel — find start/end per tile
// 3. Read back tile_ranges (~43KB) to CPU
//
// Replaces previous counting sort + bitonic sort pipeline.
// The 4-bit radix sort uses subgroup shuffle_up for stable ranking,
// avoiding local memory atomics in the scatter phase.

#include "gpu/sorter_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/kernels/radix_sort_kv_cl.h"
#include "gpu/kernels/radix_sort_cl.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <string>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

SorterGPU::SorterGPU(OpenCLContext& ctx) : ctx_(ctx) {
    auto& cl = ctx_.cl();
    cl_int err;

    // --- Build 4-bit radix sort program (subgroup shuffle_up) ---
    {
        const char* src_ptr = radix_sort_kv_cl_src;
        size_t src_len = std::strlen(radix_sort_kv_cl_src);
        cl_program program = cl.clCreateProgramWithSource(
            ctx_.context(), 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS || !program)
            throw std::runtime_error("SorterGPU: radix_sort_kv clCreateProgramWithSource failed");

        cl_device_id device = nullptr;
        cl.clGetContextInfo(ctx_.context(), CL_CONTEXT_DEVICES,
                            sizeof(device), &device, nullptr);

        err = cl.clBuildProgram(program, 1, &device,
                                "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math",
                                nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                     0, nullptr, &log_size);
            std::string log(log_size + 1, '\0');
            cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                     log_size, &log[0], nullptr);
            cl.clReleaseProgram(program);
            throw std::runtime_error(
                std::string("SorterGPU: radix_sort_kv build failed: ") + log);
        }

        rs_histogram_kernel_ = cl.clCreateKernel(program, "rs_histogram", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("SorterGPU: failed to create rs_histogram kernel");

        rs_compute_offsets_kernel_ = cl.clCreateKernel(program, "rs_compute_offsets", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("SorterGPU: failed to create rs_compute_offsets kernel");

        rs_scatter_kv_kernel_ = cl.clCreateKernel(program, "rs_scatter_kv", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("SorterGPU: failed to create rs_scatter_kv kernel");

        cl.clReleaseProgram(program);
    }

    // --- Build radix sort program (for identify_tile_ranges) ---
    {
        std::string build_opts = "-cl-fast-relaxed-math"
                                 " -DRADIX_BITS=8"
                                 " -DBLOCK_SIZE_SORT=256"
                                 " -DELEMS_PER_WG=1024";

        const char* src_ptr = radix_sort_cl_src;
        size_t src_len = std::strlen(radix_sort_cl_src);
        cl_program program = cl.clCreateProgramWithSource(
            ctx_.context(), 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS || !program)
            throw std::runtime_error("SorterGPU: radix_sort clCreateProgramWithSource failed");

        cl_device_id device = nullptr;
        cl.clGetContextInfo(ctx_.context(), CL_CONTEXT_DEVICES,
                            sizeof(device), &device, nullptr);

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
                std::string("SorterGPU: radix_sort build failed: ") + log);
        }

        tile_ranges_kernel_ = cl.clCreateKernel(program, "identify_tile_ranges", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("SorterGPU: failed to create identify_tile_ranges kernel");

        cl.clReleaseProgram(program);
    }

    std::fprintf(stderr, "SorterGPU: 4-bit radix sort (subgroup shuffle_up) compiled OK\n");
}

SorterGPU::~SorterGPU() {
    releaseBuffers();
    auto& cl = ctx_.cl();
    if (rs_histogram_kernel_)       { cl.clReleaseKernel(rs_histogram_kernel_);       rs_histogram_kernel_ = nullptr; }
    if (rs_compute_offsets_kernel_)  { cl.clReleaseKernel(rs_compute_offsets_kernel_);  rs_compute_offsets_kernel_ = nullptr; }
    if (rs_scatter_kv_kernel_)       { cl.clReleaseKernel(rs_scatter_kv_kernel_);       rs_scatter_kv_kernel_ = nullptr; }
    if (tile_ranges_kernel_)         { cl.clReleaseKernel(tile_ranges_kernel_);         tile_ranges_kernel_ = nullptr; }
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

void SorterGPU::releaseBuffers() {
    auto& cl = ctx_.cl();
    if (d_keys_ping_)       { cl.clReleaseMemObject(d_keys_ping_);       d_keys_ping_ = nullptr; }
    if (d_keys_pong_)       { cl.clReleaseMemObject(d_keys_pong_);       d_keys_pong_ = nullptr; }
    if (d_values_ping_)     { cl.clReleaseMemObject(d_values_ping_);     d_values_ping_ = nullptr; }
    if (d_values_pong_)     { cl.clReleaseMemObject(d_values_pong_);     d_values_pong_ = nullptr; }
    if (d_rs_tile_hists_)   { cl.clReleaseMemObject(d_rs_tile_hists_);   d_rs_tile_hists_ = nullptr; }
    if (d_rs_tile_offsets_) { cl.clReleaseMemObject(d_rs_tile_offsets_); d_rs_tile_offsets_ = nullptr; }
    last_capacity_ = 0;
    last_rs_num_tiles_ = 0;
}

void SorterGPU::allocBuffers(int total_pairs) {
    int rs_num_tiles = (total_pairs + RS_ELEMS_PER_WG - 1) / RS_ELEMS_PER_WG;

    if (total_pairs > last_capacity_) {
        auto& cl = ctx_.cl();
        if (d_keys_ping_)   { cl.clReleaseMemObject(d_keys_ping_);   d_keys_ping_ = nullptr; }
        if (d_keys_pong_)   { cl.clReleaseMemObject(d_keys_pong_);   d_keys_pong_ = nullptr; }
        if (d_values_ping_) { cl.clReleaseMemObject(d_values_ping_); d_values_ping_ = nullptr; }
        if (d_values_pong_) { cl.clReleaseMemObject(d_values_pong_); d_values_pong_ = nullptr; }

        d_keys_ping_   = ctx_.createBuffer(total_pairs * sizeof(uint64_t), CL_MEM_READ_WRITE);
        d_keys_pong_   = ctx_.createBuffer(total_pairs * sizeof(uint64_t), CL_MEM_READ_WRITE);
        d_values_ping_ = ctx_.createBuffer(total_pairs * sizeof(uint32_t), CL_MEM_READ_WRITE);
        d_values_pong_ = ctx_.createBuffer(total_pairs * sizeof(uint32_t), CL_MEM_READ_WRITE);
        last_capacity_ = total_pairs;
    }

    if (rs_num_tiles > last_rs_num_tiles_) {
        auto& cl = ctx_.cl();
        if (d_rs_tile_hists_)   { cl.clReleaseMemObject(d_rs_tile_hists_);   d_rs_tile_hists_ = nullptr; }
        if (d_rs_tile_offsets_) { cl.clReleaseMemObject(d_rs_tile_offsets_); d_rs_tile_offsets_ = nullptr; }

        size_t hist_size = rs_num_tiles * RS_RADIX_BUCKETS * sizeof(uint32_t);
        d_rs_tile_hists_   = ctx_.createBuffer(hist_size, CL_MEM_READ_WRITE);
        d_rs_tile_offsets_ = ctx_.createBuffer(hist_size, CL_MEM_READ_WRITE);
        last_rs_num_tiles_ = rs_num_tiles;
    }
}

// ---------------------------------------------------------------------------
// sort -- 4-bit radix sort (16 passes) + identify_tile_ranges
// ---------------------------------------------------------------------------

void SorterGPU::sort(BinningOutput& bin, FrameAllocator& alloc) {
    // Zero host-side tile_ranges (will be overwritten by GPU)
    std::memset(bin.tile_ranges, 0, bin.num_tiles * 2 * sizeof(uint32_t));

    if (bin.total_pairs == 0)
        return;

    auto* dev = static_cast<BinningDeviceBuffers*>(bin.device_data);
    if (!dev) {
        std::fprintf(stderr, "SorterGPU: ERROR -- no device buffers\n");
        return;
    }

    auto& cl = ctx_.cl();
    uint32_t total = static_cast<uint32_t>(bin.total_pairs);
    int num_tiles = bin.num_tiles;

    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    auto t_start = now();

    // Allocate radix sort buffers
    allocBuffers(bin.total_pairs);

    // Number of radix sort work-group tiles
    uint32_t rs_num_tiles = (total + RS_ELEMS_PER_WG - 1) / RS_ELEMS_PER_WG;

    // Copy unsorted data into ping buffers
    CL_CHECK(cl.clEnqueueCopyBuffer(
        ctx_.queue(), dev->keys_unsorted, d_keys_ping_,
        0, 0, total * sizeof(uint64_t), 0, nullptr, nullptr));
    CL_CHECK(cl.clEnqueueCopyBuffer(
        ctx_.queue(), dev->values_unsorted, d_values_ping_,
        0, 0, total * sizeof(uint32_t), 0, nullptr, nullptr));

    // -----------------------------------------------------------------------
    // 4-bit radix sort: 16 passes over 64-bit keys
    // -----------------------------------------------------------------------
    cl_mem cur_keys_in  = d_keys_ping_;
    cl_mem cur_keys_out = d_keys_pong_;
    cl_mem cur_vals_in  = d_values_ping_;
    cl_mem cur_vals_out = d_values_pong_;

    size_t hist_global = static_cast<size_t>(rs_num_tiles) * RS_WG_SIZE;
    size_t hist_local  = RS_WG_SIZE;
    size_t offset_global = RS_WG_SIZE;
    size_t offset_local  = RS_WG_SIZE;
    size_t scatter_global = static_cast<size_t>(rs_num_tiles) * RS_WG_SIZE;
    size_t scatter_local  = RS_WG_SIZE;

    for (int pass = 0; pass < RS_NUM_PASSES; pass++) {
        uint32_t shift = pass * RS_RADIX_BITS;

        // Step 1: Histogram
        CL_CHECK(cl.clSetKernelArg(rs_histogram_kernel_, 0, sizeof(cl_mem),  &cur_keys_in));
        CL_CHECK(cl.clSetKernelArg(rs_histogram_kernel_, 1, sizeof(cl_mem),  &d_rs_tile_hists_));
        CL_CHECK(cl.clSetKernelArg(rs_histogram_kernel_, 2, sizeof(uint32_t), &total));
        CL_CHECK(cl.clSetKernelArg(rs_histogram_kernel_, 3, sizeof(uint32_t), &shift));
        CL_CHECK(cl.clSetKernelArg(rs_histogram_kernel_, 4, sizeof(uint32_t), &rs_num_tiles));

        ctx_.enqueueKernel(rs_histogram_kernel_, 1, &hist_global, &hist_local);

        // Step 2: Compute offsets
        CL_CHECK(cl.clSetKernelArg(rs_compute_offsets_kernel_, 0, sizeof(cl_mem),  &d_rs_tile_hists_));
        CL_CHECK(cl.clSetKernelArg(rs_compute_offsets_kernel_, 1, sizeof(cl_mem),  &d_rs_tile_offsets_));
        CL_CHECK(cl.clSetKernelArg(rs_compute_offsets_kernel_, 2, sizeof(uint32_t), &rs_num_tiles));

        ctx_.enqueueKernel(rs_compute_offsets_kernel_, 1, &offset_global, &offset_local);

        // Step 3: Scatter (keys + values)
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 0, sizeof(cl_mem),  &cur_keys_in));
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 1, sizeof(cl_mem),  &cur_keys_out));
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 2, sizeof(cl_mem),  &cur_vals_in));
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 3, sizeof(cl_mem),  &cur_vals_out));
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 4, sizeof(cl_mem),  &d_rs_tile_offsets_));
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 5, sizeof(uint32_t), &total));
        CL_CHECK(cl.clSetKernelArg(rs_scatter_kv_kernel_, 6, sizeof(uint32_t), &shift));

        ctx_.enqueueKernel(rs_scatter_kv_kernel_, 1, &scatter_global, &scatter_local);

        // Ping-pong swap
        std::swap(cur_keys_in, cur_keys_out);
        std::swap(cur_vals_in, cur_vals_out);
    }

    ctx_.finish();
    auto t_radix = now();
    std::fprintf(stderr, "  [Sort] Radix sort (16 passes): %.1f ms\n", ms(t_start, t_radix));

    // After 16 passes (even number), sorted data is back in ping buffers
    // (cur_keys_in == d_keys_ping_, cur_vals_in == d_values_ping_)
    // Copy to keys_sorted / values_sorted for downstream consumption
    CL_CHECK(cl.clEnqueueCopyBuffer(
        ctx_.queue(), cur_keys_in, dev->keys_sorted,
        0, 0, total * sizeof(uint64_t), 0, nullptr, nullptr));
    CL_CHECK(cl.clEnqueueCopyBuffer(
        ctx_.queue(), cur_vals_in, dev->values_sorted,
        0, 0, total * sizeof(uint32_t), 0, nullptr, nullptr));

    // -----------------------------------------------------------------------
    // Identify tile ranges from sorted keys
    // -----------------------------------------------------------------------
    {
        uint32_t zero = 0;
        CL_CHECK(cl.clEnqueueFillBuffer(
            ctx_.queue(), dev->tile_ranges, &zero, sizeof(zero),
            0, num_tiles * 2 * sizeof(uint32_t), 0, nullptr, nullptr));

        CL_CHECK(cl.clSetKernelArg(tile_ranges_kernel_, 0, sizeof(cl_mem), &dev->keys_sorted));
        CL_CHECK(cl.clSetKernelArg(tile_ranges_kernel_, 1, sizeof(cl_mem), &dev->tile_ranges));
        int total_int = static_cast<int>(total);
        CL_CHECK(cl.clSetKernelArg(tile_ranges_kernel_, 2, sizeof(int),    &total_int));
        CL_CHECK(cl.clSetKernelArg(tile_ranges_kernel_, 3, sizeof(int),    &num_tiles));

        size_t wg = std::min(ctx_.maxWorkGroupSize(), size_t(256));
        size_t gs = ((total + wg - 1) / wg) * wg;
        ctx_.enqueueKernel(tile_ranges_kernel_, 1, &gs, &wg);
    }

    // Finish all GPU work
    ctx_.finish();
    auto t_end = now();
    std::fprintf(stderr, "  [Sort] Tile ranges: %.1f ms\n", ms(t_radix, t_end));
    std::fprintf(stderr, "  [Sort] Total GPU sort: %.1f ms\n", ms(t_start, t_end));

    // Read back tile_ranges to CPU (needed by rasterizer dispatch)
    ctx_.readBuffer(dev->tile_ranges, bin.tile_ranges,
                    num_tiles * 2 * sizeof(uint32_t));
}
