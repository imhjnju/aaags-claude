// preprocessor_gpu.cpp -- OpenCL GPU preprocessor implementation
// Uploads model data once, runs the preprocess kernel each frame.
// Only reads back radii + tiles_touched (for CPU prefix sum);
// other outputs stay on GPU for downstream scatter/rasterize kernels.

#include "gpu/preprocessor_gpu.h"
#include "gpu/kernels/preprocess_cl.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PreprocessorGPU::PreprocessorGPU(OpenCLContext& ctx) : ctx_(ctx) {
    // Compile the preprocess kernel from embedded source
    kernel_ = ctx_.buildKernel(preprocess_cl_src,
                               sizeof(preprocess_cl_src) - 1,
                               "preprocess",
                               "-cl-fast-relaxed-math");
    std::fprintf(stderr, "PreprocessorGPU: kernel compiled OK\n");
}

PreprocessorGPU::~PreprocessorGPU() {
    releaseOutputBuffers();
    releaseModelBuffers();
    releaseCameraBuffers();

    if (kernel_) {
        ctx_.cl().clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Resource management helpers
// ---------------------------------------------------------------------------

void PreprocessorGPU::releaseModelBuffers() {
    auto& cl = ctx_.cl();
    if (d_positions_)  { cl.clReleaseMemObject(d_positions_);  d_positions_ = nullptr; }
    if (d_sh_coeffs_)  { cl.clReleaseMemObject(d_sh_coeffs_);  d_sh_coeffs_ = nullptr; }
    if (d_scales_)     { cl.clReleaseMemObject(d_scales_);     d_scales_ = nullptr; }
    if (d_rotations_)  { cl.clReleaseMemObject(d_rotations_);  d_rotations_ = nullptr; }
    if (d_opacities_)  { cl.clReleaseMemObject(d_opacities_);  d_opacities_ = nullptr; }
    if (d_filter_3D_)  { cl.clReleaseMemObject(d_filter_3D_);  d_filter_3D_ = nullptr; }
    last_count_ = 0;
}

void PreprocessorGPU::releaseOutputBuffers() {
    auto& cl = ctx_.cl();
    if (dev_bufs_.means2D)       { cl.clReleaseMemObject(dev_bufs_.means2D);       dev_bufs_.means2D = nullptr; }
    if (dev_bufs_.depths)        { cl.clReleaseMemObject(dev_bufs_.depths);        dev_bufs_.depths = nullptr; }
    if (dev_bufs_.conics)        { cl.clReleaseMemObject(dev_bufs_.conics);        dev_bufs_.conics = nullptr; }
    if (dev_bufs_.rgb)           { cl.clReleaseMemObject(dev_bufs_.rgb);           dev_bufs_.rgb = nullptr; }
    if (dev_bufs_.opacities_2d)  { cl.clReleaseMemObject(dev_bufs_.opacities_2d);  dev_bufs_.opacities_2d = nullptr; }
    if (dev_bufs_.radii)         { cl.clReleaseMemObject(dev_bufs_.radii);         dev_bufs_.radii = nullptr; }
    if (dev_bufs_.tiles_touched) { cl.clReleaseMemObject(dev_bufs_.tiles_touched); dev_bufs_.tiles_touched = nullptr; }
    if (dev_bufs_.gauss2screen)  { cl.clReleaseMemObject(dev_bufs_.gauss2screen);  dev_bufs_.gauss2screen = nullptr; }
    if (dev_bufs_.cov3D_inv)     { cl.clReleaseMemObject(dev_bufs_.cov3D_inv);     dev_bufs_.cov3D_inv = nullptr; }
    if (dev_bufs_.mean_offset)   { cl.clReleaseMemObject(dev_bufs_.mean_offset);   dev_bufs_.mean_offset = nullptr; }
    last_output_count_ = 0;
}

void PreprocessorGPU::releaseCameraBuffers() {
    auto& cl = ctx_.cl();
    if (d_view_matrix_) { cl.clReleaseMemObject(d_view_matrix_); d_view_matrix_ = nullptr; }
    if (d_viewproj_)    { cl.clReleaseMemObject(d_viewproj_);    d_viewproj_ = nullptr; }
    if (d_cam_pos_)     { cl.clReleaseMemObject(d_cam_pos_);     d_cam_pos_ = nullptr; }
}

// ---------------------------------------------------------------------------
// uploadModel -- upload positions, SH coeffs, scales, rotations, opacities
// ---------------------------------------------------------------------------

void PreprocessorGPU::uploadModel(const GaussianData& g) {
    int N = g.count;

    // Release old buffers if count changed
    if (N != last_count_) {
        releaseModelBuffers();
    }

    // Create buffers (or reuse if same count -- we re-upload data anyway for correctness)
    if (N != last_count_) {
        d_positions_ = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
        d_sh_coeffs_ = ctx_.createBuffer(N * g.max_coeffs * 3 * sizeof(float), CL_MEM_READ_ONLY);
        d_scales_    = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
        d_rotations_ = ctx_.createBuffer(N * 4 * sizeof(float), CL_MEM_READ_ONLY);
        d_opacities_ = ctx_.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);
        if (g.filter_3D)
            d_filter_3D_ = ctx_.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);
    }

    // Upload data
    ctx_.writeBuffer(d_positions_, g.positions, N * 3 * sizeof(float));
    ctx_.writeBuffer(d_sh_coeffs_, g.sh_coeffs, N * g.max_coeffs * 3 * sizeof(float));
    ctx_.writeBuffer(d_scales_,    g.scales,    N * 3 * sizeof(float));
    ctx_.writeBuffer(d_rotations_, g.rotations, N * 4 * sizeof(float));
    ctx_.writeBuffer(d_opacities_, g.opacities, N * sizeof(float));
    if (g.filter_3D && d_filter_3D_)
        ctx_.writeBuffer(d_filter_3D_, g.filter_3D, N * sizeof(float));

    last_count_ = N;
}

// ---------------------------------------------------------------------------
// allocOutputBuffers -- create/resize per-frame output buffers
// ---------------------------------------------------------------------------

void PreprocessorGPU::allocOutputBuffers(int N) {
    if (N == last_output_count_)
        return;

    releaseOutputBuffers();

    // Use READ_WRITE because downstream kernels (scatter, rasterize) read these buffers
    dev_bufs_.means2D       = ctx_.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_WRITE);
    dev_bufs_.depths        = ctx_.createBuffer(N * sizeof(float),     CL_MEM_READ_WRITE);
    dev_bufs_.conics        = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_WRITE);
    dev_bufs_.rgb           = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_WRITE);
    dev_bufs_.opacities_2d  = ctx_.createBuffer(N * sizeof(float),     CL_MEM_READ_WRITE);
    dev_bufs_.radii         = ctx_.createBuffer(N * sizeof(int),       CL_MEM_READ_WRITE);
    dev_bufs_.tiles_touched = ctx_.createBuffer(N * sizeof(int),       CL_MEM_READ_WRITE);
    dev_bufs_.gauss2screen  = ctx_.createBuffer(N * 16 * sizeof(float), CL_MEM_READ_WRITE);
    dev_bufs_.cov3D_inv     = ctx_.createBuffer(N * 6 * sizeof(float), CL_MEM_READ_WRITE);
    dev_bufs_.mean_offset   = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_WRITE);

    last_output_count_ = N;
}

// ---------------------------------------------------------------------------
// process -- main entry point
// ---------------------------------------------------------------------------

PreprocessOutput PreprocessorGPU::process(const GaussianData& g, const Camera& cam,
                                          const RenderConfig& cfg, FrameAllocator& alloc,
                                          ForwardCache* /*cache*/) {
    int N = g.count;
    int sh_degree = std::min(cfg.sh_degree, g.sh_degree);

    // 1. Allocate CPU-side output arrays via FrameAllocator
    PreprocessOutput out;
    out.means2D       = alloc.allocate_array<float>(N * 2);
    out.depths        = alloc.allocate_array<float>(N);
    out.conics        = alloc.allocate_array<float>(N * 3);
    out.opacities_2d  = alloc.allocate_array<float>(N);
    out.rgb           = alloc.allocate_array<float>(N * 3);
    out.radii         = alloc.allocate_array<int>(N);
    out.tiles_touched = alloc.allocate_array<int>(N);
    out.eval_3D       = cfg.eval_3D;
    out.gauss2screen  = cfg.eval_3D ? alloc.allocate_array<float>(N * 16) : nullptr;
    out.cov3D_inv     = cfg.eval_3D ? alloc.allocate_array<float>(N * 6) : nullptr;
    out.mean_offset   = cfg.eval_3D ? alloc.allocate_array<float>(N * 3) : nullptr;

    // 2. Upload model data if first time or count changed
    if (N != last_count_) {
        uploadModel(g);
    }

    // 3. Allocate output device buffers
    allocOutputBuffers(N);

    // 4. Upload camera uniforms to __constant buffers
    //    Reuse existing buffers (small: 16+16+3 floats = 140 bytes total)
    if (!d_view_matrix_) {
        d_view_matrix_ = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
        d_viewproj_    = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
        d_cam_pos_     = ctx_.createBuffer(3 * sizeof(float),  CL_MEM_READ_ONLY);
    }

    ctx_.writeBuffer(d_view_matrix_, cam.view_matrix, 16 * sizeof(float));
    ctx_.writeBuffer(d_viewproj_,    cam.viewproj_matrix, 16 * sizeof(float));
    ctx_.writeBuffer(d_cam_pos_,     cam.cam_pos, 3 * sizeof(float));

    // 5. Set kernel arguments
    //    Arg indices match the kernel signature order exactly.
    auto& cl = ctx_.cl();
    cl_uint arg = 0;

    // Model data (args 0-4)
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_positions_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_sh_coeffs_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_scales_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_rotations_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_opacities_));

    // Camera uniforms (args 5-7)
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_view_matrix_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_viewproj_));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_cam_pos_));

    // Scalar uniforms (args 8-18)
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &N));
    int width = cam.width;
    int height = cam.height;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &width));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &height));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cam.tan_fovx));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cam.tan_fovy));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(float), &cfg.scale_modifier));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &sh_degree));
    int max_coeffs = g.max_coeffs;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &max_coeffs));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &cfg.tile_w));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &cfg.tile_h));
    int aa = cfg.antialiasing ? 1 : 0;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &aa));
    int training = cfg.training ? 1 : 0;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int),   &training));

    // Output buffers (args 20-26)
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.means2D));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.depths));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.conics));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.rgb));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.opacities_2d));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.radii));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.tiles_touched));

    // AAA-Gaussians: eval_3D args (26-28)
    int eval_3d_flag = cfg.eval_3D ? 1 : 0;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(int), &eval_3d_flag));
    cl_mem filter_buf = d_filter_3D_ ? d_filter_3D_ : nullptr;
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &filter_buf));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.gauss2screen));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.cov3D_inv));
    CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &dev_bufs_.mean_offset));

    // 6. Enqueue kernel with profiling event
    size_t wg_size = std::min(ctx_.maxWorkGroupSize(), size_t(256));
    size_t global_size = ((size_t(N) + wg_size - 1) / wg_size) * wg_size;
    cl_event ev;
    ctx_.enqueueKernel(kernel_, 1, &global_size, &wg_size, &ev);
    ctx_.finish();
    double kernel_ms = ctx_.getEventTimeMs(ev);
    ctx_.cl().clReleaseEvent(ev);
    std::fprintf(stderr, "  [GPU] preprocess kernel: %.1f ms (global=%zu, local=%zu)\n",
                 kernel_ms, global_size, wg_size);

    // 7. Read back ONLY radii and tiles_touched (needed for CPU prefix sum).
    //    means2D, depths, conics, rgb, opacities_2d stay on GPU --
    //    downstream kernels (scatter, rasterize) read from device_data buffers.
    ctx_.readBuffer(dev_bufs_.radii,         out.radii,         N * sizeof(int));
    ctx_.readBuffer(dev_bufs_.tiles_touched, out.tiles_touched,  N * sizeof(int));

    // 8. Store device buffer pointer for downstream GPU stages
    out.device_data = &dev_bufs_;

    // Diagnostic: count valid Gaussians (only radii/tiles_touched are on CPU)
    int valid_count = 0;
    int total_tiles = 0;
    for (int i = 0; i < N; i++) {
        if (out.radii[i] > 0) valid_count++;
        total_tiles += out.tiles_touched[i];
    }
    std::fprintf(stderr, "PreprocessorGPU: %d/%d valid, %d total tile-pairs\n",
                 valid_count, N, total_tiles);

    return out;
}
