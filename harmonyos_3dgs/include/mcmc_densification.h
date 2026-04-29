#pragma once

// mcmc_densification.h — MCMC densification (relocate dead + grow up to cap_max).
// Port of AAA-Gaussians scene/gaussian_model.py: relocate_gs / add_new_gs.
//
// Differs from densification.cpp (legacy 3DGS clone/split/prune) — this file
// implements the MCMC algorithm from "3D Gaussian Splatting as Markov Chain
// Monte Carlo" used by AAA-Gaussians train.py:130-134.

#include "train_types.h"  // OwnedRawParams

#include <cstdint>
#include <vector>

namespace mcmc {

// Replace dead Gaussians (sigmoid(opacity) <= opacity_thresh) with samples
// drawn — with replacement, weighted by activation-space opacity — from the
// alive pool. Source Gaussians' opacity and scale are also updated to the
// redistributed values per Eq. 9. Total Gaussian count is unchanged.
//
// Returns the number of dead Gaussians relocated.
//
// Determinism: results are a pure function of (params, opacity_thresh, rng_seed).
int relocate_gs(OwnedRawParams& params, float opacity_thresh, uint32_t rng_seed);

// Grow Gaussian count up to min(cap_max, int(1.05 * N)). New Gaussians are
// sampled (with replacement, weighted by activation-space opacity) from the
// entire current pool. Each source's opacity and scale are also updated per
// Eq. 9 (matching the Python reference behavior).
//
// Returns the number of Gaussians appended. Final count is N + return value.
int add_new_gs(OwnedRawParams& params, int cap_max, uint32_t rng_seed);

// Combined densify(): relocate then add. Mirrors AAA-Gaussians train.py:130-134.
// Returns the final Gaussian count.
int densify(OwnedRawParams& params,
            float opacity_thresh,
            int cap_max,
            uint32_t rng_seed);

// Result type for densify_ex() — includes final count plus Gaussian indices whose
// Adam state must be reset after relocate/add changes parameter ownership.
struct DensifyResult {
    int final_count;
    std::vector<int> modified_source_indices;       // Eq. 9 source updates
    std::vector<int> replaced_destination_indices;  // relocated dead slots
};

struct DensifySamplePlan {
    std::vector<int> relocate_sources;
    std::vector<int> add_sources;
};

// Like densify() but also returns indices needed for Adam moment zeroing.
DensifyResult densify_ex(OwnedRawParams& params,
                          float opacity_thresh,
                          int cap_max,
                          uint32_t rng_seed);

// Replay a densification pass with pre-sampled global source indices. This is
// used by CUDA/PyTorch golden tests because torch.multinomial and C++ RNG are
// intentionally not bit-exact.
DensifyResult densify_with_samples(OwnedRawParams& params,
                                   float opacity_thresh,
                                   int cap_max,
                                   const DensifySamplePlan& plan);

}  // namespace mcmc
