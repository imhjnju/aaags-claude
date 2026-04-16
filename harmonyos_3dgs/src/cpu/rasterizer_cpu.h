#pragma once
#include "rasterizer.h"

class RasterizerCPU : public Rasterizer {
public:
    void rasterize(
        const PreprocessOutput& preprocess,
        const BinningOutput& binning,
        const Camera& camera,
        const RenderConfig& config,
        float* output_image,
        float* output_depth = nullptr,
        ForwardCache* cache = nullptr,
        FrameAllocator* allocator = nullptr) override;
};
