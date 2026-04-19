#pragma once
#include "train_types.h"  // RawGaussianParams, OwnedRawParams, VkTrainingConfig

// densify_and_prune: AAA-Gaussians MCMC densification.
// Modifies params_owned in-place: appends clones/splits, then compacts out pruned Gaussians.
// Returns the new Gaussian count.
//
// Algorithm:
//   For each Gaussian i:
//     - Prune if sigmoid(raw_opacity[i]) < opacity_thresh
//     - Clone if: should_densify && is_small && !pruned
//     - Split (2 children) if: should_densify && !is_small (regardless of opacity)
//   After restructuring, inject MCMC position noise proportional to max scale.
//
// scene_extent: max axis range of positions (used for is_small test and noise scale).
int densify_and_prune(
    OwnedRawParams& params_owned,
    const float* grad_means2D_norm,  // [N] accumulated |dL/dmeans2D| per Gaussian
    int step,
    const VkTrainingConfig& cfg,
    float scene_extent);             // scene scale (max axis range of positions)
