// rasterizer_backward_gpu.cpp -- OpenCL GPU rasterizer backward implementation.
// Uploads d_image and forward cache to GPU, dispatches backward kernel,
// reads back per-Gaussian gradient arrays to CPU.

#include "gpu/rasterizer_backward_gpu.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/kernels/rasterize_backward_cl.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

static constexpr int TILE_W = 16;
static constexpr int TILE_H = 16;
static constexpr int BLOCK_SIZE = TILE_W * TILE_H;

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RasterizerBackwardGPU::RasterizerBackwardGPU(OpenCLContext& ctx) : ctx_(ctx) {
    kernel_ = ctx_.buildKernel(rasterize_backward_cl_src,
                                sizeof(rasterize_backward_cl_src) - 1,
                                "rasterize_backward",
                                "-cl-fast-relaxed-math");
    std::fprintf(stderr, "RasterizerBackwardGPU: kernel compiled OK\n");
}

RasterizerBackwardGPU::~RasterizerBackwardGPU() {
    releaseGradBuffers();
    releasePixelBuffers();
    if (kernel_) {
        ctx_.cl().clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Resource management
// ---------------------------------------------------------------------------

void RasterizerBackwardGPU::releaseGradBuffers() {
    auto& cl = ctx_.cl();
    if (d_d_means2D_)      { cl.clReleaseMemObject(d_d_means2D_);      d_d_means2D_ = nullptr; }
    if (d_d_conics_)       { cl.clReleaseMemObject(d_d_conics_);       d_d_conics_ = nullptr; }
    if (d_d_rgb_)          { cl.clReleaseMemObject(d_d_rgb_);          d_d_rgb_ = nullptr; }
    if (d_d_opacities_2d_) { cl.clReleaseMemObject(d_d_opacities_2d_); d_d_opacities_2d_ = nullptr; }
    last_N_ = 0;
}

void RasterizerBackwardGPU::releasePixelBuffers() {
    auto& cl = ctx_.cl();
    if (d_T_final_)   { cl.clReleaseMemObject(d_T_final_);   d_T_final_ = nullptr; }
    if (d_n_contrib_) { cl.clReleaseMemObject(d_n_contrib_); d_n_contrib_ = nullptr; }
    if (d_d_image_)   { cl.clReleaseMemObject(d_d_image_);   d_d_image_ = nullptr; }
    last_num_pixels_ = 0;
}

void RasterizerBackwardGPU::allocGradBuffers(int N) {
    if (N == last_N_) return;
    releaseGradBuffers();
    d_d_means2D_      = ctx_.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_WRITE);
    d_d_conics_       = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_WRITE);
    d_d_rgb_          = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_WRITE);
    d_d_opacities_2d_ = ctx_.createBuffer(N * sizeof(float),     CL_MEM_READ_WRITE);
    last_N_ = N;
}

void RasterizerBackwardGPU::allocPixelBuffers(int num_pixels) {
    if (num_pixels == last_num_pixels_) return;
    releasePixelBuffers();
    d_T_final_   = ctx_.createBuffer(num_pixels * sizeof(float),     CL_MEM_READ_ONLY);
    d_n_contrib_ = ctx_.createBuffer(num_pixels * sizeof(int),       CL_MEM_READ_ONLY);
    d_d_image_   = ctx_.createBuffer(num_pixels * 3 * sizeof(float), CL_MEM_READ_ONLY);
    last_num_pixels_ = num_pixels;
}

// ---------------------------------------------------------------------------
// backward -- main entry point
// ---------------------------------------------------------------------------

void RasterizerBackwardGPU::backward(const PreprocessOutput& pre, const BinningOutput& bin,
                                      const Camera& cam, const RenderConfig& cfg,
                                      const ForwardCache& cache, const float* d_image,
                                      int N, RasterGradOutput& rgrad) {
    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;
    int num_tiles = grid_x * grid_y;
    int num_pixels = cam.width * cam.height;

    // Allocate GPU buffers
    allocGradBuffers(N);
    allocPixelBuffers(num_pixels);

    // Zero gradient buffers on GPU
    {
        std::vector<float> zeros_f;
        // Zero d_means2D
        zeros_f.assign(N * 2, 0.0f);
        ctx_.writeBuffer(d_d_means2D_, zeros_f.data(), N * 2 * sizeof(float));
        // Zero d_conics
        zeros_f.assign(N * 3, 0.0f);
        ctx_.writeBuffer(d_d_conics_, zeros_f.data(), N * 3 * sizeof(float));
        // Zero d_rgb
        ctx_.writeBuffer(d_d_rgb_, zeros_f.data(), N * 3 * sizeof(float));
        // Zero d_opacities_2d
        zeros_f.assign(N, 0.0f);
        ctx_.writeBuffer(d_d_opacities_2d_, zeros_f.data(), N * sizeof(float));
    }

    // Upload forward cache and d_image
    ctx_.writeBuffer(d_T_final_,   cache.T_final,  num_pixels * sizeof(float));
    ctx_.writeBuffer(d_n_contrib_, cache.n_contrib, num_pixels * sizeof(int));
    ctx_.writeBuffer(d_d_image_,   d_image,         num_pixels * 3 * sizeof(float));

    // Get device buffers from upstream stages
    auto* pre_dev = static_cast<PreprocessDeviceBuffers*>(pre.device_data);
    auto* bin_dev = static_cast<BinningDeviceBuffers*>(bin.device_data);

    if (!pre_dev || !bin_dev) {
        std::fprintf(stderr, "RasterizerBackwardGPU: missing device buffers\n");
        return;
    }

    // Set kernel arguments
    auto& cl = ctx_.cl();
    cl_uint arg = 0;

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->means2D));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->conics));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->rgb));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->opacities_2d));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &bin_dev->values_sorted));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &bin_dev->tile_ranges));

    int width = cam.width;
    int height = cam.height;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &width));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &height));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.bg_color[0]));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.bg_color[1]));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.bg_color[2]));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &grid_x));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_image_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_T_final_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_n_contrib_));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_means2D_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_conics_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_rgb_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_opacities_2d_));

    // Dispatch kernel
    size_t global_size = static_cast<size_t>(num_tiles) * BLOCK_SIZE;
    size_t local_size = BLOCK_SIZE;
    cl_event ev;
    ctx_.enqueueKernel(kernel_, 1, &global_size, &local_size, &ev);
    ctx_.finish();
    double kernel_ms = ctx_.getEventTimeMs(ev);
    ctx_.cl().clReleaseEvent(ev);
    std::fprintf(stderr, "  [GPU] rasterize_backward kernel: %.1f ms (%d tiles)\n",
                 kernel_ms, num_tiles);

    // Read back gradient arrays to CPU
    ctx_.readBuffer(d_d_means2D_,      rgrad.d_means2D,      N * 2 * sizeof(float));
    ctx_.readBuffer(d_d_conics_,       rgrad.d_conics,       N * 3 * sizeof(float));
    ctx_.readBuffer(d_d_rgb_,          rgrad.d_rgb,          N * 3 * sizeof(float));
    ctx_.readBuffer(d_d_opacities_2d_, rgrad.d_opacities_2d, N * sizeof(float));
}
