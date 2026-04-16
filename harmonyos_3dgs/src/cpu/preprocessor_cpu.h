#pragma once
#include "preprocessor.h"

class PreprocessorCPU : public Preprocessor {
public:
    PreprocessOutput process(
        const GaussianData& gaussians,
        const Camera& camera,
        const RenderConfig& config,
        FrameAllocator& allocator,
        ForwardCache* cache = nullptr) override;
};
