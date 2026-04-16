#pragma once
#include "train_types.h"
#include <vector>

class SGDOptimizer {
public:
    void step(RawGaussianParams& params, const GradientOutput& grads,
              const TrainConfig& cfg, int iteration);
};

class AdamOptimizer {
public:
    void step(RawGaussianParams& params, const GradientOutput& grads,
              const TrainConfig& cfg, int iteration);

    // Resize and zero-initialize m/v buffers. Called when N or max_coeffs changes.
    void reset(int N, int max_coeffs);

    int step_count() const { return step_count_; }

private:
    // Per-element Adam update: param -= lr * m_hat / (sqrt(v_hat) + eps)
    static void adam_update(float* param, const float* grad, float* m, float* v,
                            int count, float lr, float beta1, float beta2,
                            float eps, float bc1, float bc2);

    std::vector<float> m_positions_, v_positions_;     // [N*3]
    std::vector<float> m_scales_, v_scales_;           // [N*3]
    std::vector<float> m_rotations_, v_rotations_;     // [N*4]
    std::vector<float> m_sh_coeffs_, v_sh_coeffs_;     // [N*max_coeffs*3]
    std::vector<float> m_opacities_, v_opacities_;     // [N]

    int step_count_ = 0;
    int last_N_ = 0;
    int last_mc_ = 0;
};
