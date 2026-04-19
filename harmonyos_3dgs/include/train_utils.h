#pragma once
#include <cmath>

// op_sigmoid: soft-step function used in Python reference noise injection.
// op_sigmoid(x, k=100, x0=0.995) = 1 / (1 + exp(-k*(x - x0)))
// For opacity=0.1: x=1-0.1=0.9 < 0.995 -> near 0 (active, no noise)
// For opacity=0.001: x=0.999 > 0.995 -> near 1 (dead, gets noise)
// Reference: train.py line 143-144
inline float op_sigmoid(float x, float k = 100.f, float x0 = 0.995f) {
    return 1.f / (1.f + std::exp(-k * (x - x0)));
}

// build_L: construct L = R @ diag(act_scale) from normalized quaternion [w,x,y,z]
// and activated (exp) scale values. L is stored row-major: L[row][col].
// Covariance Sigma = L @ L^T = R @ diag(s^2) @ R^T.
// Reference: train.py line 141 (build_scaling_rotation)
void build_L(const float rot[4], const float act_scale[3], float L[3][3]);

// spatial_lr_schedule: exponential decay scaled by spatial_lr_scale.
// pos_lr = spatial_lr_scale * lr_schedule(lr_init, lr_final, step, max_steps)
// Reference: arguments/__init__.py position_lr_init * cameras_extent
float spatial_lr_schedule(float lr_init, float lr_final,
                           float spatial_lr_scale,
                           int step, int max_steps);
