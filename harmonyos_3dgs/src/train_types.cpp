#include "train_types.h"
#include <cmath>
#include <cstring>
#include <algorithm>

void RawGaussianParams::activate(GaussianData& g) const {
    const int N = count;

    // Positions: identity
    std::memcpy(g.positions, raw_positions, N * 3 * sizeof(float));

    // Scales: exp()
    for (int i = 0; i < N * 3; i++) {
        g.scales[i] = std::exp(raw_scales[i]);
    }

    // Rotations: normalize quaternion
    for (int i = 0; i < N; i++) {
        const float* q = &raw_rotations[i * 4];
        float len = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (len < 1e-12f) len = 1e-12f;
        float inv = 1.0f / len;
        g.rotations[i * 4 + 0] = q[0] * inv;
        g.rotations[i * 4 + 1] = q[1] * inv;
        g.rotations[i * 4 + 2] = q[2] * inv;
        g.rotations[i * 4 + 3] = q[3] * inv;
    }

    // SH coeffs: identity
    std::memcpy(g.sh_coeffs, raw_sh_coeffs, N * max_coeffs * 3 * sizeof(float));

    // Opacities: sigmoid
    for (int i = 0; i < N; i++) {
        g.opacities[i] = 1.0f / (1.0f + std::exp(-raw_opacities[i]));
    }
}

void RasterGradOutput::allocate_and_zero(FrameAllocator& alloc, int N) {
    d_means2D      = alloc.allocate_array<float>(N * 2);
    d_conics       = alloc.allocate_array<float>(N * 3);
    d_rgb          = alloc.allocate_array<float>(N * 3);
    d_opacities_2d = alloc.allocate_array<float>(N);

    std::memset(d_means2D,      0, N * 2 * sizeof(float));
    std::memset(d_conics,       0, N * 3 * sizeof(float));
    std::memset(d_rgb,          0, N * 3 * sizeof(float));
    std::memset(d_opacities_2d, 0, N * sizeof(float));
}

void GradientOutput::allocate_and_zero(FrameAllocator& alloc, int N, int max_coeffs) {
    d_raw_positions  = alloc.allocate_array<float>(N * 3);
    d_raw_scales     = alloc.allocate_array<float>(N * 3);
    d_raw_rotations  = alloc.allocate_array<float>(N * 4);
    d_raw_sh_coeffs  = alloc.allocate_array<float>(N * max_coeffs * 3);
    d_raw_opacities  = alloc.allocate_array<float>(N);

    std::memset(d_raw_positions, 0, N * 3 * sizeof(float));
    std::memset(d_raw_scales,    0, N * 3 * sizeof(float));
    std::memset(d_raw_rotations, 0, N * 4 * sizeof(float));
    std::memset(d_raw_sh_coeffs, 0, N * max_coeffs * 3 * sizeof(float));
    std::memset(d_raw_opacities, 0, N * sizeof(float));
}

float lr_schedule(float lr_init, float lr_final, int step, int max_steps) {
    float t = static_cast<float>(step) / static_cast<float>(max_steps);
    t = std::max(0.0f, std::min(1.0f, t));
    return std::exp(std::log(lr_init) * (1.0f - t) + std::log(lr_final) * t);
}
