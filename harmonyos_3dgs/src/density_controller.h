#pragma once
#include "train_types.h"
#include <vector>

// Adaptive density control: gradient accumulation, clone, split, prune.
// Matches the official 3DGS densify_and_prune logic.
class DensityController {
public:
    // Accumulate 2D position gradient norms (called every training step).
    // d_means2D: [N*2] from RasterGradOutput
    // radii: [N] screen-space radius from PreprocessOutput (or null)
    void accumulate_stats(const float* d_means2D, const int* radii, int N);

    // Run densification + pruning (called every densify_interval steps).
    // Modifies owned_params in-place (may add/remove Gaussians).
    // Returns new Gaussian count.
    int densify_and_prune(OwnedRawParams& params, const DensifyConfig& cfg,
                          float scene_extent);

    // Reset all opacities to inverse_sigmoid(0.01) ≈ -4.595
    static void reset_opacity(OwnedRawParams& params);

    // Reset accumulators (called after densification or when N changes)
    void reset(int N);

    int current_N() const { return current_N_; }

private:
    void densify_and_clone(OwnedRawParams& params, const std::vector<float>& avg_grads,
                           float threshold, float extent);
    void densify_and_split(OwnedRawParams& params, const std::vector<float>& avg_grads,
                           float threshold, float extent);

    std::vector<float> grad_accum_;     // [N]
    std::vector<float> grad_denom_;     // [N]
    std::vector<int>   max_radii2D_;    // [N]
    int current_N_ = 0;

    // Simple PRNG for split sampling
    uint64_t rng_state_ = 12345;
    float randn();
};
