#pragma once
#include "types.h"
#include "train_types.h"

class PreprocessorBackwardCPU {
public:
    void backward(const GaussianData& g, const Camera& cam,
                  const RenderConfig& cfg, const ForwardCache& cache,
                  const RasterGradOutput& rgrad, const RawGaussianParams& raw,
                  GradientOutput& grads);
};

// SH backward: compute d_sh_coeffs and d_pos from d_rgb.
// d_pos may be null (for SH-only tests); if non-null, accumulates the position
// gradient from the SH view-direction dependency (required for SH degree >= 1).
void computeColorFromSH_backward(int degree, int max_coeffs,
                                   const float* sh_coeffs,
                                   const float* pos, const float* cam_pos,
                                   const float* d_rgb, float* d_sh_coeffs,
                                   bool training = true,
                                   float* d_pos = nullptr);
