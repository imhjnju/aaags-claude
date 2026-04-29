// densification.cpp — MCMC-style densification for AAA-Gaussians (3DGS).
//
// Implements densify_and_prune():
//   1. For each Gaussian, determine action: prune, clone, split, or keep.
//   2. Build output OwnedRawParams by iterating original set.
//   3. Apply MCMC position noise to all surviving Gaussians.
//
// References:
//   - AAA-Gaussians Python reference (gaussian_model.py: densify_and_prune)
//   - 3DGS original densification (gaussian_model.py)

#include "densification.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

// ---------------------------------------------------------------------------
// densify_and_prune
// ---------------------------------------------------------------------------

int densify_and_prune(
    OwnedRawParams& params_owned,
    const float*    grad_means2D_norm,
    int             step,
    const VkTrainingConfig& cfg,
    float           scene_extent)
{
    const int N = params_owned.count();
    if (N == 0) return 0;

    const int mc3 = params_owned.max_coeffs * 3;

    // Reserve output — worst case 3*N (keep + 2 split children per Gaussian).
    OwnedRawParams output;
    output.sh_degree  = params_owned.sh_degree;
    output.max_coeffs = params_owned.max_coeffs;

    // Pre-reserve to avoid repeated allocations.
    output.positions.reserve(static_cast<size_t>(N) * 3 * 3);
    output.scales.reserve   (static_cast<size_t>(N) * 3 * 3);
    output.rotations.reserve(static_cast<size_t>(N) * 4 * 3);
    output.sh_coeffs.reserve(static_cast<size_t>(N) * mc3 * 3);
    output.opacities.reserve(static_cast<size_t>(N) * 3);
    if (!params_owned.filter_3D.empty()) output.filter_3D.reserve(static_cast<size_t>(N) * 3);

    const float grad_thresh     = cfg.densify_grad_thresh;
    const float opacity_thresh  = cfg.opacity_thresh;
    const float percent_dense   = cfg.densify_percent_dense;
    const float size_threshold  = percent_dense * scene_extent;
    const float log_1_6         = std::log(1.6f);  // scale shrink for split children

    for (int i = 0; i < N; ++i) {
        // --- Compute per-Gaussian properties ---

        const float* pos = params_owned.positions.data() + static_cast<size_t>(i) * 3;
        const float* sc  = params_owned.scales.data()    + static_cast<size_t>(i) * 3;
        const float* rot = params_owned.rotations.data() + static_cast<size_t>(i) * 4;
        const float* sh  = params_owned.sh_coeffs.data() + static_cast<size_t>(i) * mc3;
        const float  op  = params_owned.opacities[i];
        const float* filter_3D = params_owned.filter_3D.empty() ? nullptr : &params_owned.filter_3D[static_cast<size_t>(i)];

        // Compute exp(raw_scales) and find max.
        const float s0 = std::exp(sc[0]);
        const float s1 = std::exp(sc[1]);
        const float s2 = std::exp(sc[2]);
        float max_scale = s0;
        int   max_axis  = 0;
        if (s1 > max_scale) { max_scale = s1; max_axis = 1; }
        if (s2 > max_scale) { max_scale = s2; max_axis = 2; }

        const bool should_densify = (grad_means2D_norm[i] > grad_thresh);
        const bool is_small       = (max_scale < size_threshold);
        const bool pruned         = (sigmoid(op) < opacity_thresh);

        // --- Keep original (if not pruned) ---
        if (!pruned) {
            output.append(pos, sc, rot, sh, op, filter_3D);
        }

        // --- Clone: small Gaussians with high gradient ---
        if (should_densify && is_small && !pruned) {
            output.append(pos, sc, rot, sh, op, filter_3D);
        }

        // --- Split: large Gaussians with high gradient (bypasses opacity prune) ---
        if (should_densify && !is_small) {
            // New scales: shrink all by log(1.6) in log space.
            float new_sc[3] = { sc[0] - log_1_6, sc[1] - log_1_6, sc[2] - log_1_6 };

            // New positions: offset ±0.5 * exp(max_scale_value) along world axis.
            // The world axis for axis k is the k-th standard basis vector.
            const float half_offset = 0.5f * max_scale;
            float pos_child1[3] = { pos[0], pos[1], pos[2] };
            float pos_child2[3] = { pos[0], pos[1], pos[2] };
            pos_child1[max_axis] += half_offset;
            pos_child2[max_axis] -= half_offset;

            output.append(pos_child1, new_sc, rot, sh, op, filter_3D);
            output.append(pos_child2, new_sc, rot, sh, op, filter_3D);
        }
    }

    params_owned = std::move(output);

    // --- MCMC position noise injection ---
    // After densification, inject small Gaussian noise proportional to scale,
    // matching the MCMC sampling step from AAA-Gaussians.
    const int new_N = params_owned.count();
    if (new_N > 0) {
        const float noise_lr = 5e-4f * scene_extent;

        // Seed with step so noise is unique per densification call while
        // remaining deterministic (reproducible given the same step).
        std::mt19937 rng(static_cast<uint32_t>(step));
        std::normal_distribution<float> normal(0.0f, 1.0f);

        for (int i = 0; i < new_N; ++i) {
            const float* sc = params_owned.scales.data() + static_cast<size_t>(i) * 3;
            const float s0 = std::exp(sc[0]);
            const float s1 = std::exp(sc[1]);
            const float s2 = std::exp(sc[2]);
            const float max_s = std::max({ s0, s1, s2 });

            float* pos = params_owned.positions.data() + static_cast<size_t>(i) * 3;
            for (int k = 0; k < 3; ++k) {
                pos[k] += max_s * noise_lr * normal(rng);
            }
        }
    }

    return new_N;
}
