#include "cpu_adam.h"
#include <cmath>

CpuAdam::CpuAdam(float beta1, float beta2, float eps)
    : beta1_(beta1), beta2_(beta2), eps_(eps) {}

void CpuAdam::add_group(AdamGroup g) {
    groups_.push_back(g);
}

void CpuAdam::step() {
    ++step_;
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));
    for (auto& g : groups_) {
        for (int k = 0; k < g.n; ++k) {
            float grad = g.grad[k];
            g.m[k] = beta1_ * g.m[k] + (1.0f - beta1_) * grad;
            g.v[k] = beta2_ * g.v[k] + (1.0f - beta2_) * grad * grad;
            float m_hat = g.m[k] / bc1;
            float v_hat = g.v[k] / bc2;
            g.params[k] -= g.lr * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}
