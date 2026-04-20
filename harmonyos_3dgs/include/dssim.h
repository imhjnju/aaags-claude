#pragma once
#include <cstddef>

// Compute the combined L1 + lambda_dssim * DSSIM loss gradient.
// rendered, target: [W*H*3] float, pixel-major (pixel p, channel ch at p*3+ch).
// dL_dpixels: [W*H*3] float output gradient (must be pre-allocated, caller-owned).
// Returns: combined scalar loss = (1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM).
float compute_combined_loss_gradient(
    const float* rendered,
    const float* target,
    float* dL_dpixels,
    int W, int H,
    float lambda_dssim = 0.2f);
