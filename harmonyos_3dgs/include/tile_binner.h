#pragma once
#include "types.h"

class TileBinner {
public:
    virtual ~TileBinner() = default;
    virtual BinningOutput bin(
        const PreprocessOutput& preprocess,
        int num_gaussians,
        const Camera& camera,
        const RenderConfig& config,
        FrameAllocator& allocator) = 0;
};
