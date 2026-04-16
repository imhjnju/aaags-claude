// preprocessor_backward_gpu.h -- OpenCL GPU backward pass for preprocessing.
// Computes gradients of raw parameters from rasterizer gradients.

#pragma once

#include "types.h"
#include "train_types.h"
#include "gpu/opencl_context.h"

class PreprocessorBackwardGPU {
public:
    explicit PreprocessorBackwardGPU(OpenCLContext& ctx);
    ~PreprocessorBackwardGPU();

    PreprocessorBackwardGPU(const PreprocessorBackwardGPU&) = delete;
    PreprocessorBackwardGPU& operator=(const PreprocessorBackwardGPU&) = delete;

    void backward(const GaussianData& g, const Camera& cam,
                  const RenderConfig& cfg, const ForwardCache& cache,
                  const RasterGradOutput& rgrad, const RawGaussianParams& raw,
                  GradientOutput& grads);

private:
    OpenCLContext& ctx_;
    cl_kernel kernel_ = nullptr;

    // Model data buffers (uploaded each call since training updates them)
    cl_mem d_positions_ = nullptr;
    cl_mem d_sh_coeffs_ = nullptr;
    cl_mem d_scales_ = nullptr;
    cl_mem d_rotations_ = nullptr;
    cl_mem d_opacities_ = nullptr;

    // Camera uniform buffers
    cl_mem d_view_matrix_ = nullptr;
    cl_mem d_viewproj_ = nullptr;
    cl_mem d_cam_pos_ = nullptr;

    // Forward cache buffers
    cl_mem d_cov2D_ = nullptr;
    cl_mem d_cov2D_det_ = nullptr;
    cl_mem d_p_view_ = nullptr;
    cl_mem d_p_hom_w_ = nullptr;
    cl_mem d_means2D_fwd_ = nullptr;
    cl_mem d_radii_ = nullptr;

    // Rasterizer gradient inputs
    cl_mem d_d_means2D_ = nullptr;
    cl_mem d_d_conics_ = nullptr;
    cl_mem d_d_rgb_ = nullptr;
    cl_mem d_d_opacities_2d_ = nullptr;

    // Raw rotations (for normalization backward)
    cl_mem d_raw_rotations_ = nullptr;

    // Output gradient buffers
    cl_mem d_d_raw_positions_ = nullptr;
    cl_mem d_d_raw_scales_ = nullptr;
    cl_mem d_d_raw_rotations_ = nullptr;
    cl_mem d_d_raw_sh_coeffs_ = nullptr;
    cl_mem d_d_raw_opacities_ = nullptr;

    int last_N_ = 0;
    int last_mc3_ = 0;

    void allocBuffers(int N, int max_coeffs);
    void releaseBuffers();
};
