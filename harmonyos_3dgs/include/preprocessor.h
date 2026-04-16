#pragma once
#include "types.h"

class Preprocessor {
public:
    virtual ~Preprocessor() = default;
    virtual PreprocessOutput process(
        const GaussianData& gaussians,
        const Camera& camera,
        const RenderConfig& config,
        FrameAllocator& allocator,
        ForwardCache* cache = nullptr) = 0;
};
