#pragma once
#include "types.h"

class Rasterizer {
public:
    virtual ~Rasterizer() = default;
    virtual void rasterize(
        const PreprocessOutput& preprocess,
        const BinningOutput& binning,
        const Camera& camera,
        const RenderConfig& config,
        float* output_image,
        float* output_depth = nullptr,
        ForwardCache* cache = nullptr,
        FrameAllocator* allocator = nullptr) = 0;
};
