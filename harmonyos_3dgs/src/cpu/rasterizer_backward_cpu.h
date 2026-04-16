#pragma once
#include "types.h"
#include "train_types.h"

class RasterizerBackwardCPU {
public:
    void backward(const PreprocessOutput& pre, const BinningOutput& bin,
                  const Camera& cam, const RenderConfig& cfg,
                  const ForwardCache& cache, const float* d_image,
                  RasterGradOutput& rgrad);
};
