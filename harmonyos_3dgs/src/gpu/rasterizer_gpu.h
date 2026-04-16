// rasterizer_gpu.h -- OpenCL GPU rasterizer for 3DGS
// Implements the Rasterizer interface using a tile-based alpha blending kernel.
// One work-group per tile, one work-item per pixel within the tile.
// Uses shared memory for collaborative loading of Gaussian data.

#pragma once

#include "rasterizer.h"
#include "gpu/opencl_context.h"

class RasterizerGPU : public Rasterizer {
public:
    explicit RasterizerGPU(OpenCLContext& ctx);
    ~RasterizerGPU() override;

    // Not copyable/movable (owns CL resources)
    RasterizerGPU(const RasterizerGPU&) = delete;
    RasterizerGPU& operator=(const RasterizerGPU&) = delete;

    void rasterize(
        const PreprocessOutput& preprocess,
        const BinningOutput& binning,
        const Camera& camera,
        const RenderConfig& config,
        float* output_image,
        float* output_depth = nullptr,
        ForwardCache* cache = nullptr,
        FrameAllocator* allocator = nullptr) override;

private:
    OpenCLContext& ctx_;
    cl_kernel kernel_ = nullptr;

    // Output image buffer on GPU (reallocated if resolution changes)
    cl_mem d_out_image_ = nullptr;
    int last_width_ = 0;
    int last_height_ = 0;

    void allocOutputBuffer(int width, int height);
    void releaseOutputBuffer();
};
