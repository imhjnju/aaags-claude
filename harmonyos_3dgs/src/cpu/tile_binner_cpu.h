#pragma once
#include "tile_binner.h"

class TileBinnerCPU : public TileBinner {
public:
    BinningOutput bin(
        const PreprocessOutput& preprocess,
        int num_gaussians,
        const Camera& camera,
        const RenderConfig& config,
        FrameAllocator& allocator) override;
};
