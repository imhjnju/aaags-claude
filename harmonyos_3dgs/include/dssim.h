#pragma once
#include <cstddef>

// Compute the combined L1 + lambda_dssim * DSSIM loss gradient.
// rendered, target: [3*W*H] float, CHW channel-first (channel ch at ch*H*W + pixel p).
// dL_dpixels: [W*H*3] float output gradient (must be pre-allocated, caller-owned).
// Returns: combined scalar loss = (1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM).
float compute_combined_loss_gradient(
    const float* rendered,
    const float* target,
    float* dL_dpixels,
    int W, int H,
    float lambda_dssim = 0.2f);
