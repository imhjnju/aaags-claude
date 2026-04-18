#include "density_controller.h"
#include <cmath>
#include <algorithm>

// ============================================================
// PRNG for split sampling (Box-Muller)
// ============================================================

float DensityController::randn() {
    rng_state_ = rng_state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = (float)((rng_state_ >> 33) & 0xFFFFFF) / (float)0xFFFFFF;
    if (u1 < 1e-7f) u1 = 1e-7f;
    rng_state_ = rng_state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = (float)((rng_state_ >> 33) & 0xFFFFFF) / (float)0xFFFFFF;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

// ============================================================
// Accumulate statistics
// ============================================================

void DensityController::reset(int N) {
    current_N_ = N;
    grad_accum_.assign(N, 0.0f);
    grad_denom_.assign(N, 0.0f);
    max_radii2D_.assign(N, 0);
}

void DensityController::accumulate_stats(const float* d_means2D, const int* radii, int N) {
    if (N != current_N_) reset(N);

    for (int i = 0; i < N; i++) {
        float dx = d_means2D[i * 2];
        float dy = d_means2D[i * 2 + 1];
        float norm = std::sqrt(dx * dx + dy * dy);
        grad_accum_[i] += norm;
        grad_denom_[i] += 1.0f;
        if (radii && radii[i] > max_radii2D_[i])
            max_radii2D_[i] = radii[i];
    }
}

// ============================================================
// Clone: duplicate small Gaussians with high gradients
// ============================================================

void DensityController::densify_and_clone(OwnedRawParams& params,
                                           const std::vector<float>& avg_grads,
                                           float threshold, float extent) {
    int N = params.count();
    float size_threshold = 0.01f * extent;  // percent_dense * extent
    int mc3 = params.max_coeffs * 3;

    for (int i = 0; i < N; i++) {
        if (avg_grads[i] < threshold) continue;

        // Check scale: max(exp(raw_scale)) <= size_threshold → small Gaussian
        float max_scale = 0;
        for (int j = 0; j < 3; j++)
            max_scale = std::max(max_scale, std::exp(params.scales[i * 3 + j]));
        if (max_scale > size_threshold) continue;

        // Copy source into local stack-safe buffers BEFORE append, to prevent
        // dangling-pointer UB if push_back inside append reallocates any of
        // the OwnedRawParams vectors.
        float pos_copy[3], scale_copy[3], rot_copy[4];
        for (int j = 0; j < 3; j++) pos_copy[j]   = params.positions[i * 3 + j];
        for (int j = 0; j < 3; j++) scale_copy[j] = params.scales[i * 3 + j];
        for (int j = 0; j < 4; j++) rot_copy[j]   = params.rotations[i * 4 + j];
        std::vector<float> sh_copy(params.sh_coeffs.begin() + i * mc3,
                                   params.sh_coeffs.begin() + (i + 1) * mc3);
        float op_copy = params.opacities[i];

        params.append(pos_copy, scale_copy, rot_copy, sh_copy.data(), op_copy);
    }
}

// ============================================================
// Split: split large Gaussians into 2 smaller ones
// ============================================================

void DensityController::densify_and_split(OwnedRawParams& params,
                                           const std::vector<float>& avg_grads,
                                           float threshold, float extent) {
    int N = params.count();
    float size_threshold = 0.01f * extent;
    std::vector<bool> to_remove(N, false);

    for (int i = 0; i < N; i++) {
        if (avg_grads[i] < threshold) continue;

        float max_scale = 0;
        for (int j = 0; j < 3; j++)
            max_scale = std::max(max_scale, std::exp(params.scales[i * 3 + j]));
        if (max_scale <= size_threshold) continue;

        // Split into 2 new Gaussians
        float scale[3], rot[4];
        for (int j = 0; j < 3; j++) scale[j] = std::exp(params.scales[i * 3 + j]);
        for (int j = 0; j < 4; j++) rot[j] = params.rotations[i * 4 + j];

        // Normalize quaternion for rotation matrix
        float qlen = std::sqrt(rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2] + rot[3]*rot[3]);
        if (qlen > 1e-8f) for (int j = 0; j < 4; j++) rot[j] /= qlen;

        // Rotation matrix from quaternion
        float r = rot[0], x = rot[1], y = rot[2], z = rot[3];
        float R[9] = {
            1-2*(y*y+z*z), 2*(x*y+r*z), 2*(x*z-r*y),
            2*(x*y-r*z), 1-2*(x*x+z*z), 2*(y*z+r*x),
            2*(x*z+r*y), 2*(y*z-r*x), 1-2*(x*x+y*y)
        };

        // New scale: reduced by factor 1/(0.8 * 2) in log-space
        float new_scale_log[3];
        float scale_factor = std::log(0.8f * 2.0f);  // ~0.47
        for (int j = 0; j < 3; j++)
            new_scale_log[j] = params.scales[i * 3 + j] - scale_factor;

        int mc3 = params.max_coeffs * 3;

        // Copy source rotation and SH coefficients BEFORE appending, to prevent
        // dangling-pointer UB if push_back inside append reallocates.
        // Also snapshot position and opacity so the k-loop reads stable values.
        float rot_copy[4];
        for (int j = 0; j < 4; j++) rot_copy[j] = params.rotations[i * 4 + j];
        std::vector<float> sh_copy(params.sh_coeffs.begin() + i * mc3,
                                   params.sh_coeffs.begin() + (i + 1) * mc3);
        float op_copy = params.opacities[i];
        float pos_src[3];
        for (int j = 0; j < 3; j++) pos_src[j] = params.positions[i * 3 + j];

        for (int k = 0; k < 2; k++) {
            // Sample offset from N(0, scale) in local coords, rotate to world
            float sample[3] = {randn() * scale[0], randn() * scale[1], randn() * scale[2]};
            float offset[3] = {0, 0, 0};
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 3; col++)
                    offset[row] += R[row * 3 + col] * sample[col];

            float new_pos[3];
            for (int j = 0; j < 3; j++)
                new_pos[j] = pos_src[j] + offset[j];

            params.append(new_pos, new_scale_log, rot_copy, sh_copy.data(), op_copy);
        }

        to_remove[i] = true;
    }

    // Remove original split points
    params.compact(to_remove);
}

// ============================================================
// Prune: remove low-opacity or oversized Gaussians
// ============================================================

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int DensityController::densify_and_prune(OwnedRawParams& params,
                                          const DensifyConfig& cfg,
                                          float scene_extent) {
    int N = params.count();
    if (N == 0) return 0;

    // Compute average gradients
    std::vector<float> avg_grads(N, 0.0f);
    for (int i = 0; i < N; i++) {
        if (grad_denom_[i] > 0)
            avg_grads[i] = grad_accum_[i] / grad_denom_[i];
    }

    // Clone then split (order matches official)
    densify_and_clone(params, avg_grads, cfg.grad_threshold, scene_extent);
    densify_and_split(params, avg_grads, cfg.grad_threshold, scene_extent);

    // Prune by opacity and screen size
    int new_N = params.count();
    std::vector<bool> prune_mask(new_N, false);
    for (int i = 0; i < new_N; i++) {
        float op = sigmoid(params.opacities[i]);
        if (op < cfg.min_opacity)
            prune_mask[i] = true;
    }
    // Screen size pruning (only for original indices that have max_radii2D)
    for (int i = 0; i < std::min(N, new_N); i++) {
        if (i < (int)max_radii2D_.size() && max_radii2D_[i] > cfg.max_screen_size)
            prune_mask[i] = true;
    }
    // World-space size pruning
    for (int i = 0; i < new_N; i++) {
        float max_s = 0;
        for (int j = 0; j < 3; j++)
            max_s = std::max(max_s, std::exp(params.scales[i * 3 + j]));
        if (max_s > 0.1f * scene_extent)
            prune_mask[i] = true;
    }

    params.compact(prune_mask);

    // Reset accumulators for next interval
    reset(params.count());

    return params.count();
}

// ============================================================
// Reset opacity
// ============================================================

void DensityController::reset_opacity(OwnedRawParams& params) {
    // inverse_sigmoid(0.01) = log(0.01/0.99) ≈ -4.595
    float inv_sig_001 = std::log(0.01f / 0.99f);
    for (int i = 0; i < params.count(); i++)
        params.opacities[i] = inv_sig_001;
}
