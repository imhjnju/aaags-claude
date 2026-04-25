#pragma once
// Compute L1 loss and gradient.
// rendered, gt: [3*H*W] CHW channel-first RGB
// d_image: [H*W*3] output gradient
// Returns: mean absolute error
float l1_loss(const float* rendered, const float* gt, int H, int W, float* d_image);

// Double-precision L1 loss for gradient verification (FD needs higher precision).
// Same gradient output as l1_loss, but returns loss as double.
double l1_loss_double(const float* rendered, const float* gt, int H, int W, float* d_image);
