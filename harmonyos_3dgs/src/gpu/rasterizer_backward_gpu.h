// rasterizer_backward_gpu.h -- OpenCL GPU backward pass for the rasterizer.
// Computes gradients of screen-space Gaussian attributes (means2D, conics,
// rgb, opacity_2d) from the image-space loss gradient.

#pragma once

#include "types.h"
#include "train_types.h"
#include "gpu/opencl_context.h"

class RasterizerBackwardGPU {
public:
    explicit RasterizerBackwardGPU(OpenCLContext& ctx);
    ~RasterizerBackwardGPU();

    RasterizerBackwardGPU(const RasterizerBackwardGPU&) = delete;
    RasterizerBackwardGPU& operator=(const RasterizerBackwardGPU&) = delete;

    /// Run the backward rasterizer kernel.
    /// d_image: [H*W*3] gradient of loss w.r.t. rendered image (on CPU, uploaded).
    /// cache: forward cache containing T_final and n_contrib.
    /// rgrad: output gradients (on CPU, downloaded after kernel).
    /// Pre/Binning device buffers are obtained from pre.device_data / bin.device_data.
    void backward(const PreprocessOutput& pre, const BinningOutput& bin,
                  const Camera& cam, const RenderConfig& cfg,
                  const ForwardCache& cache, const float* d_image,
                  int num_gaussians, RasterGradOutput& rgrad);

private:
    OpenCLContext& ctx_;
    cl_kernel kernel_ = nullptr;

    // Gradient output buffers on GPU
    cl_mem d_d_means2D_ = nullptr;       // [N*2]
    cl_mem d_d_conics_ = nullptr;        // [N*3]
    cl_mem d_d_rgb_ = nullptr;           // [N*3]
    cl_mem d_d_opacities_2d_ = nullptr;  // [N]

    // Forward cache buffers on GPU
    cl_mem d_T_final_ = nullptr;         // [H*W]
    cl_mem d_n_contrib_ = nullptr;       // [H*W]

    // Image gradient buffer on GPU
    cl_mem d_d_image_ = nullptr;         // [H*W*3]

    int last_N_ = 0;
    int last_num_pixels_ = 0;

    void allocGradBuffers(int N);
    void allocPixelBuffers(int num_pixels);
    void releaseGradBuffers();
    void releasePixelBuffers();
};
