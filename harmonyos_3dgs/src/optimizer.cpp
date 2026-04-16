#include "optimizer.h"
#include <cmath>
#include <cstring>

// ============================================================
// SGD Optimizer (unchanged)
// ============================================================

void SGDOptimizer::step(RawGaussianParams& params, const GradientOutput& grads,
                         const TrainConfig& cfg, int iteration) {
    float lr_pos = lr_schedule(cfg.lr_position_init, cfg.lr_position_final,
                                iteration, cfg.max_steps);
    int N = params.count;
    int mc3 = params.max_coeffs * 3;

    for (int i = 0; i < N * 3; i++)
        params.raw_positions[i] -= lr_pos * grads.d_raw_positions[i];
    for (int i = 0; i < N * 3; i++)
        params.raw_scales[i] -= cfg.lr_scaling * grads.d_raw_scales[i];
    for (int i = 0; i < N * 4; i++)
        params.raw_rotations[i] -= cfg.lr_rotation * grads.d_raw_rotations[i];
    for (int i = 0; i < N * mc3; i++)
        params.raw_sh_coeffs[i] -= cfg.lr_feature * grads.d_raw_sh_coeffs[i];
    for (int i = 0; i < N; i++)
        params.raw_opacities[i] -= cfg.lr_opacity * grads.d_raw_opacities[i];
}

// ============================================================
// Adam Optimizer
// ============================================================

void AdamOptimizer::adam_update(float* param, const float* grad, float* m, float* v,
                                 int count, float lr, float beta1, float beta2,
                                 float eps, float bc1, float bc2) {
    for (int i = 0; i < count; i++) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] * bc1;  // bc1 = 1 / (1 - beta1^t)
        float v_hat = v[i] * bc2;  // bc2 = 1 / (1 - beta2^t)
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void AdamOptimizer::reset(int N, int max_coeffs) {
    int mc3 = max_coeffs * 3;
    auto resize_zero = [](std::vector<float>& vec, size_t n) {
        vec.assign(n, 0.0f);
    };
    resize_zero(m_positions_, N * 3);  resize_zero(v_positions_, N * 3);
    resize_zero(m_scales_, N * 3);     resize_zero(v_scales_, N * 3);
    resize_zero(m_rotations_, N * 4);  resize_zero(v_rotations_, N * 4);
    resize_zero(m_sh_coeffs_, N * mc3); resize_zero(v_sh_coeffs_, N * mc3);
    resize_zero(m_opacities_, N);      resize_zero(v_opacities_, N);
    step_count_ = 0;
    last_N_ = N;
    last_mc_ = max_coeffs;
}

void AdamOptimizer::step(RawGaussianParams& params, const GradientOutput& grads,
                          const TrainConfig& cfg, int iteration) {
    int N = params.count;
    int mc3 = params.max_coeffs * 3;

    // Auto-reset if N or max_coeffs changed
    if (N != last_N_ || params.max_coeffs != last_mc_)
        reset(N, params.max_coeffs);

    step_count_++;
    float beta1 = cfg.adam_beta1;
    float beta2 = cfg.adam_beta2;
    float eps = cfg.adam_eps;

    // Bias correction factors: 1 / (1 - beta^t)
    float bc1 = 1.0f / (1.0f - std::pow(beta1, (float)step_count_));
    float bc2 = 1.0f / (1.0f - std::pow(beta2, (float)step_count_));

    // Position LR with exponential decay schedule
    float lr_pos = lr_schedule(cfg.lr_position_init, cfg.lr_position_final,
                                iteration, cfg.max_steps);

    adam_update(params.raw_positions, grads.d_raw_positions,
               m_positions_.data(), v_positions_.data(),
               N * 3, lr_pos, beta1, beta2, eps, bc1, bc2);

    adam_update(params.raw_scales, grads.d_raw_scales,
               m_scales_.data(), v_scales_.data(),
               N * 3, cfg.lr_scaling, beta1, beta2, eps, bc1, bc2);

    adam_update(params.raw_rotations, grads.d_raw_rotations,
               m_rotations_.data(), v_rotations_.data(),
               N * 4, cfg.lr_rotation, beta1, beta2, eps, bc1, bc2);

    adam_update(params.raw_sh_coeffs, grads.d_raw_sh_coeffs,
               m_sh_coeffs_.data(), v_sh_coeffs_.data(),
               N * mc3, cfg.lr_feature, beta1, beta2, eps, bc1, bc2);

    adam_update(params.raw_opacities, grads.d_raw_opacities,
               m_opacities_.data(), v_opacities_.data(),
               N, cfg.lr_opacity, beta1, beta2, eps, bc1, bc2);
}
