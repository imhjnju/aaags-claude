// preprocessor_backward_gpu.cpp -- OpenCL GPU preprocessor backward implementation.
// Uploads all needed data, dispatches the backward kernel, reads back gradients.

#include "gpu/preprocessor_backward_gpu.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/kernels/preprocess_backward_cl.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PreprocessorBackwardGPU::PreprocessorBackwardGPU(OpenCLContext& ctx) : ctx_(ctx) {
    kernel_ = ctx_.buildKernel(preprocess_backward_cl_src,
                                sizeof(preprocess_backward_cl_src) - 1,
                                "preprocess_backward",
                                "-cl-fast-relaxed-math");
    std::fprintf(stderr, "PreprocessorBackwardGPU: kernel compiled OK\n");
}

PreprocessorBackwardGPU::~PreprocessorBackwardGPU() {
    releaseBuffers();
    if (kernel_) {
        ctx_.cl().clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Resource management
// ---------------------------------------------------------------------------

void PreprocessorBackwardGPU::releaseBuffers() {
    auto& cl = ctx_.cl();
    auto rel = [&](cl_mem& m) { if (m) { cl.clReleaseMemObject(m); m = nullptr; } };
    rel(d_positions_); rel(d_sh_coeffs_); rel(d_scales_);
    rel(d_rotations_); rel(d_opacities_);
    rel(d_view_matrix_); rel(d_viewproj_); rel(d_cam_pos_);
    rel(d_cov2D_); rel(d_cov2D_det_); rel(d_p_view_);
    rel(d_p_hom_w_); rel(d_means2D_fwd_); rel(d_radii_);
    rel(d_d_means2D_); rel(d_d_conics_); rel(d_d_rgb_); rel(d_d_opacities_2d_);
    rel(d_raw_rotations_);
    rel(d_d_raw_positions_); rel(d_d_raw_scales_); rel(d_d_raw_rotations_);
    rel(d_d_raw_sh_coeffs_); rel(d_d_raw_opacities_);
    last_N_ = 0;
    last_mc3_ = 0;
}

void PreprocessorBackwardGPU::allocBuffers(int N, int max_coeffs) {
    int mc3 = max_coeffs * 3;
    if (N == last_N_ && mc3 == last_mc3_) return;
    releaseBuffers();

    // Model data
    d_positions_  = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    d_sh_coeffs_  = ctx_.createBuffer(N * mc3 * sizeof(float), CL_MEM_READ_ONLY);
    d_scales_     = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    d_rotations_  = ctx_.createBuffer(N * 4 * sizeof(float), CL_MEM_READ_ONLY);
    d_opacities_  = ctx_.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);

    // Camera
    d_view_matrix_ = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
    d_viewproj_    = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
    d_cam_pos_     = ctx_.createBuffer(3 * sizeof(float), CL_MEM_READ_ONLY);

    // Forward cache
    d_cov2D_       = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    d_cov2D_det_   = ctx_.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);
    d_p_view_      = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    d_p_hom_w_     = ctx_.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);
    d_means2D_fwd_ = ctx_.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    d_radii_       = ctx_.createBuffer(N * sizeof(int), CL_MEM_READ_ONLY);

    // Rasterizer gradient inputs
    d_d_means2D_      = ctx_.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    d_d_conics_       = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    d_d_rgb_          = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    d_d_opacities_2d_ = ctx_.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);

    // Raw rotations
    d_raw_rotations_ = ctx_.createBuffer(N * 4 * sizeof(float), CL_MEM_READ_ONLY);

    // Output gradients
    d_d_raw_positions_  = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_WRITE_ONLY);
    d_d_raw_scales_     = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_WRITE_ONLY);
    d_d_raw_rotations_  = ctx_.createBuffer(N * 4 * sizeof(float), CL_MEM_WRITE_ONLY);
    d_d_raw_sh_coeffs_  = ctx_.createBuffer(N * mc3 * sizeof(float), CL_MEM_WRITE_ONLY);
    d_d_raw_opacities_  = ctx_.createBuffer(N * sizeof(float), CL_MEM_WRITE_ONLY);

    last_N_ = N;
    last_mc3_ = mc3;
}

// ---------------------------------------------------------------------------
// backward -- main entry point
// ---------------------------------------------------------------------------

void PreprocessorBackwardGPU::backward(const GaussianData& g, const Camera& cam,
                                        const RenderConfig& cfg, const ForwardCache& cache,
                                        const RasterGradOutput& rgrad, const RawGaussianParams& raw,
                                        GradientOutput& grads) {
    int N = g.count;
    int max_coeffs = g.max_coeffs;
    int mc3 = max_coeffs * 3;

    allocBuffers(N, max_coeffs);

    // Upload model data
    ctx_.writeBuffer(d_positions_,  g.positions,  N * 3 * sizeof(float));
    ctx_.writeBuffer(d_sh_coeffs_,  g.sh_coeffs,  N * mc3 * sizeof(float));
    ctx_.writeBuffer(d_scales_,     g.scales,     N * 3 * sizeof(float));
    ctx_.writeBuffer(d_rotations_,  g.rotations,  N * 4 * sizeof(float));
    ctx_.writeBuffer(d_opacities_,  g.opacities,  N * sizeof(float));

    // Upload camera
    ctx_.writeBuffer(d_view_matrix_, cam.view_matrix,    16 * sizeof(float));
    ctx_.writeBuffer(d_viewproj_,    cam.viewproj_matrix, 16 * sizeof(float));
    ctx_.writeBuffer(d_cam_pos_,     cam.cam_pos,         3 * sizeof(float));

    // Upload forward cache
    ctx_.writeBuffer(d_cov2D_,       cache.cov2D,     N * 3 * sizeof(float));
    ctx_.writeBuffer(d_cov2D_det_,   cache.cov2D_det, N * sizeof(float));
    ctx_.writeBuffer(d_p_view_,      cache.p_view,    N * 3 * sizeof(float));
    ctx_.writeBuffer(d_p_hom_w_,     cache.p_hom_w,   N * sizeof(float));
    ctx_.writeBuffer(d_means2D_fwd_, cache.pre->means2D, N * 2 * sizeof(float));
    ctx_.writeBuffer(d_radii_,       cache.pre->radii,   N * sizeof(int));

    // Upload rasterizer gradients
    ctx_.writeBuffer(d_d_means2D_,      rgrad.d_means2D,      N * 2 * sizeof(float));
    ctx_.writeBuffer(d_d_conics_,       rgrad.d_conics,       N * 3 * sizeof(float));
    ctx_.writeBuffer(d_d_rgb_,          rgrad.d_rgb,          N * 3 * sizeof(float));
    ctx_.writeBuffer(d_d_opacities_2d_, rgrad.d_opacities_2d, N * sizeof(float));

    // Upload raw rotations
    ctx_.writeBuffer(d_raw_rotations_, raw.raw_rotations, N * 4 * sizeof(float));

    // Set kernel arguments
    auto& cl = ctx_.cl();
    cl_uint arg = 0;

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_positions_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_sh_coeffs_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_scales_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_rotations_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_opacities_));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_view_matrix_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_viewproj_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_cam_pos_));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &N));
    int width = cam.width;
    int height = cam.height;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &width));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &height));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cam.tan_fovx));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cam.tan_fovy));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.scale_modifier));
    int sh_degree = std::min(cfg.sh_degree, g.sh_degree);
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &sh_degree));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &max_coeffs));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_cov2D_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_cov2D_det_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_p_view_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_p_hom_w_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_means2D_fwd_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_radii_));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_means2D_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_conics_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_rgb_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_opacities_2d_));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_raw_rotations_));

    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_raw_positions_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_raw_scales_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_raw_rotations_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_raw_sh_coeffs_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_d_raw_opacities_));

    // Dispatch kernel
    size_t wg_size = std::min(ctx_.maxWorkGroupSize(), size_t(256));
    size_t global_size = ((size_t(N) + wg_size - 1) / wg_size) * wg_size;
    cl_event ev;
    ctx_.enqueueKernel(kernel_, 1, &global_size, &wg_size, &ev);
    ctx_.finish();
    double kernel_ms = ctx_.getEventTimeMs(ev);
    ctx_.cl().clReleaseEvent(ev);
    std::fprintf(stderr, "  [GPU] preprocess_backward kernel: %.1f ms (N=%d)\n",
                 kernel_ms, N);

    // Read back gradient arrays
    ctx_.readBuffer(d_d_raw_positions_,  grads.d_raw_positions,  N * 3 * sizeof(float));
    ctx_.readBuffer(d_d_raw_scales_,     grads.d_raw_scales,     N * 3 * sizeof(float));
    ctx_.readBuffer(d_d_raw_rotations_,  grads.d_raw_rotations,  N * 4 * sizeof(float));
    ctx_.readBuffer(d_d_raw_sh_coeffs_,  grads.d_raw_sh_coeffs,  N * mc3 * sizeof(float));
    ctx_.readBuffer(d_d_raw_opacities_,  grads.d_raw_opacities,  N * sizeof(float));
}
