// rasterizer_gpu.cpp -- OpenCL GPU rasterizer implementation
// Dispatches the tile-based alpha blending kernel on GPU.
// Gets sorted data from BinningOutput::device_data and preprocess data
// from PreprocessOutput::device_data. Reads back the output image to CPU.

#include "gpu/rasterizer_gpu.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/kernels/rasterize_cl.h"
#include "math_utils.h"

#include <cstdio>
#include <algorithm>

// Tile dimensions must match the kernel defines
static constexpr int TILE_W = 16;
static constexpr int TILE_H = 16;
static constexpr int BLOCK_SIZE = TILE_W * TILE_H;  // 256

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RasterizerGPU::RasterizerGPU(OpenCLContext& ctx) : ctx_(ctx) {
    kernel_ = ctx_.buildKernel(rasterize_cl_src,
                                sizeof(rasterize_cl_src) - 1,
                                "rasterize",
                                "-cl-fast-relaxed-math");
    std::fprintf(stderr, "RasterizerGPU: rasterize kernel compiled OK\n");
}

RasterizerGPU::~RasterizerGPU() {
    releaseOutputBuffer();
    if (kernel_) {
        ctx_.cl().clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Resource management
// ---------------------------------------------------------------------------

void RasterizerGPU::releaseOutputBuffer() {
    if (d_out_image_) {
        ctx_.cl().clReleaseMemObject(d_out_image_);
        d_out_image_ = nullptr;
    }
    last_width_ = 0;
    last_height_ = 0;
}

void RasterizerGPU::allocOutputBuffer(int width, int height) {
    if (width == last_width_ && height == last_height_)
        return;

    releaseOutputBuffer();
    d_out_image_ = ctx_.createBuffer(width * height * 3 * sizeof(float), CL_MEM_WRITE_ONLY);
    last_width_ = width;
    last_height_ = height;
}

// ---------------------------------------------------------------------------
// rasterize -- main entry point
// ---------------------------------------------------------------------------

void RasterizerGPU::rasterize(const PreprocessOutput& pre, const BinningOutput& bin,
                               const Camera& cam, const RenderConfig& cfg,
                               float* out_img, float* out_depth,
                               ForwardCache* /*cache*/, FrameAllocator* /*allocator*/) {
    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;
    int num_tiles = grid_x * grid_y;

    // Allocate output image buffer on GPU
    allocOutputBuffer(cam.width, cam.height);

    // Get device buffers from upstream stages
    auto* pre_dev = static_cast<PreprocessDeviceBuffers*>(pre.device_data);
    auto* bin_dev = static_cast<BinningDeviceBuffers*>(bin.device_data);

    if (!pre_dev || !bin_dev) {
        std::fprintf(stderr, "RasterizerGPU: missing device buffers, cannot rasterize\n");
        return;
    }

    // -----------------------------------------------------------------------
    // Set kernel arguments
    // -----------------------------------------------------------------------
    auto& cl = ctx_.cl();
    cl_uint arg = 0;

    // Preprocess data from GPU
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->means2D));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->conics));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->rgb));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->opacities_2d));

    // Sorted binning data from GPU
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &bin_dev->values_sorted));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &bin_dev->tile_ranges));

    // Scalar uniforms
    int width = cam.width;
    int height = cam.height;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &width));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &height));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.bg_color[0]));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.bg_color[1]));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.bg_color[2]));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &grid_x));

    // Output image buffer
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_out_image_));

    // AAA-Gaussians: eval_3D flag and gauss2screen buffer
    int eval_3d_flag = pre.eval_3D ? 1 : 0;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int), &eval_3d_flag));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->gauss2screen));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->depths));

    // StopThePop per-pixel depth parameters
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->cov3D_inv));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->mean_offset));

    // Upload inverse_vp and cam_pos for per-pixel ray computation
    float inverse_vp[16];
    bool have_inv = false;
    if (pre.eval_3D) {
        have_inv = invertMatrix4x4(cam.viewproj_matrix, inverse_vp);
    }
    cl_mem d_inv_vp = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
    if (have_inv)
        ctx_.writeBuffer(d_inv_vp, inverse_vp, 16 * sizeof(float));
    else {
        float zeros[16] = {};
        ctx_.writeBuffer(d_inv_vp, zeros, 16 * sizeof(float));
    }
    cl_mem d_cam_pos = ctx_.createBuffer(3 * sizeof(float), CL_MEM_READ_ONLY);
    ctx_.writeBuffer(d_cam_pos, cam.cam_pos, 3 * sizeof(float));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_inv_vp));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_cam_pos));

    // -----------------------------------------------------------------------
    // Dispatch kernel: one work-group per tile, BLOCK_SIZE threads per group
    // -----------------------------------------------------------------------
    size_t global_size = static_cast<size_t>(num_tiles) * BLOCK_SIZE;
    size_t local_size = BLOCK_SIZE;
    cl_event raster_event;
    ctx_.enqueueKernel(kernel_, 1, &global_size, &local_size, &raster_event);
    ctx_.finish();
    double kernel_ms = ctx_.getEventTimeMs(raster_event);
    ctx_.cl().clReleaseEvent(raster_event);
    std::fprintf(stderr, "  [GPU] rasterize kernel: %.1f ms (%d tiles, %zu threads)\n",
                 kernel_ms, num_tiles, global_size);

    // Release temporary StopThePop buffers
    ctx_.cl().clReleaseMemObject(d_inv_vp);
    ctx_.cl().clReleaseMemObject(d_cam_pos);

    // -----------------------------------------------------------------------
    // Read back output image to CPU
    // -----------------------------------------------------------------------
    ctx_.readBuffer(d_out_image_, out_img, cam.width * cam.height * 3 * sizeof(float));

    // Note: depth output is not supported in the GPU path yet
    if (out_depth) {
        std::fprintf(stderr, "RasterizerGPU: depth output not yet supported on GPU\n");
    }
}
