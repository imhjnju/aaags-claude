#pragma once
#include "types.h"
#include <vector>
#include <cstring>
#include <cmath>

// Raw (pre-activation) Gaussian parameters for SGD training.
// SGD updates these directly; activate() converts to GaussianData.
struct RawGaussianParams {
    int count;
    int sh_degree;
    int max_coeffs;       // (sh_degree+1)^2

    float* raw_positions;  // [N*3] identity (no activation)
    float* raw_scales;     // [N*3] log-space (activated via exp)
    float* raw_rotations;  // [N*4] unnormalized quaternion
    float* raw_sh_coeffs;  // [N*max_coeffs*3] identity (no activation)
    float* raw_opacities;  // [N] inverse-sigmoid space (activated via sigmoid)

    // Write activated values into g. Caller must pre-allocate g's arrays.
    void activate(GaussianData& g) const;
};

// Gradients from the rasterizer (screen-space quantities).
struct RasterGradOutput {
    float* d_means2D;       // [N*2]
    float* d_conics;        // [N*3]
    float* d_rgb;           // [N*3]
    float* d_opacities_2d;  // [N]

    void allocate_and_zero(FrameAllocator& alloc, int N);
};

// Gradients w.r.t. raw (pre-activation) parameters.
struct GradientOutput {
    float* d_raw_positions;  // [N*3]
    float* d_raw_scales;     // [N*3]
    float* d_raw_rotations;  // [N*4]
    float* d_raw_sh_coeffs;  // [N*max_coeffs*3]
    float* d_raw_opacities;  // [N]

    void allocate_and_zero(FrameAllocator& alloc, int N, int max_coeffs);
};

// Training hyper-parameters with 3DGS defaults.
struct TrainConfig {
    float lr_position_init  = 0.00016f;
    float lr_position_final = 0.0000016f;
    float lr_feature        = 0.0025f;
    float lr_opacity        = 0.025f;  // CPU trainer default; VulkanTrainer uses 0.05f (Python ref)
    float lr_scaling        = 0.005f;
    float lr_rotation       = 0.001f;
    int   max_steps         = 30000;

    // Adam optimizer (matches official 3DGS / PyTorch Adam)
    bool  use_adam    = false;  // default SGD for backward compat; set true explicitly
    float adam_beta1  = 0.9f;
    float adam_beta2  = 0.999f;
    float adam_eps    = 1e-8f;   // PyTorch Adam default; 1e-15 causes sign-SGD behavior
};

// Minimal SGD state (per-parameter momentum etc. can be added later).
struct SGDOptimizerState {
    int step_count = 0;
};

// Exponential learning-rate decay: exp(log(init)*(1-t) + log(final)*t)
float lr_schedule(float lr_init, float lr_final, int step, int max_steps);

// Densification hyper-parameters (matches official 3DGS train.py).
struct DensifyConfig {
    int   densify_from_iter     = 500;
    int   densify_until_iter    = 15000;
    int   densify_interval      = 100;
    float grad_threshold        = 0.0002f;
    float min_opacity           = 0.005f;
    float percent_dense         = 0.01f;
    int   opacity_reset_interval = 3000;
    int   max_screen_size       = 20;
    bool  enabled               = false;   // must opt-in
};

// Owning container for raw Gaussian parameters. Manages memory via std::vector.
// Provides resize, append, compact operations for densification.
struct OwnedRawParams {
    int sh_degree = 0;
    int max_coeffs = 1;

    std::vector<float> positions;    // [N*3]
    std::vector<float> scales;       // [N*3]
    std::vector<float> rotations;    // [N*4]
    std::vector<float> sh_coeffs;    // [N*max_coeffs*3]
    std::vector<float> opacities;    // [N]

    int count() const { return (int)opacities.size(); }

    // Resize all arrays for N Gaussians (zero-initialized for new elements)
    void resize(int N) {
        int mc3 = max_coeffs * 3;
        positions.resize(N * 3, 0.0f);
        scales.resize(N * 3, 0.0f);
        rotations.resize(N * 4, 0.0f);
        sh_coeffs.resize(N * mc3, 0.0f);
        opacities.resize(N, 0.0f);
    }

    // Append one Gaussian (copies from src arrays at index src_i)
    void append(const float* src_pos, const float* src_sc, const float* src_rot,
                const float* src_sh, float src_op) {
        int mc3 = max_coeffs * 3;
        for (int j = 0; j < 3; j++) positions.push_back(src_pos[j]);
        for (int j = 0; j < 3; j++) scales.push_back(src_sc[j]);
        for (int j = 0; j < 4; j++) rotations.push_back(src_rot[j]);
        for (int j = 0; j < mc3; j++) sh_coeffs.push_back(src_sh[j]);
        opacities.push_back(src_op);
    }

    // Remove Gaussians where mask[i] == true. Compact arrays in-place.
    void compact(const std::vector<bool>& remove_mask) {
        int N = count();
        int mc3 = max_coeffs * 3;
        int dst = 0;
        for (int src = 0; src < N; src++) {
            if (remove_mask[src]) continue;
            if (dst != src) {
                std::memcpy(&positions[dst*3], &positions[src*3], 3*sizeof(float));
                std::memcpy(&scales[dst*3], &scales[src*3], 3*sizeof(float));
                std::memcpy(&rotations[dst*4], &rotations[src*4], 4*sizeof(float));
                std::memcpy(&sh_coeffs[dst*mc3], &sh_coeffs[src*mc3], mc3*sizeof(float));
                opacities[dst] = opacities[src];
            }
            dst++;
        }
        resize(dst);
    }

    // Export to RawGaussianParams (non-owning view)
    RawGaussianParams as_raw() {
        RawGaussianParams r;
        r.count = count();
        r.sh_degree = sh_degree;
        r.max_coeffs = max_coeffs;
        r.raw_positions = positions.data();
        r.raw_scales = scales.data();
        r.raw_rotations = rotations.data();
        r.raw_sh_coeffs = sh_coeffs.data();
        r.raw_opacities = opacities.data();
        return r;
    }

    // Import from RawGaussianParams (copy data)
    void from_raw(const RawGaussianParams& r) {
        sh_degree = r.sh_degree;
        max_coeffs = r.max_coeffs;
        int N = r.count;
        int mc3 = max_coeffs * 3;
        positions.assign(r.raw_positions, r.raw_positions + N*3);
        scales.assign(r.raw_scales, r.raw_scales + N*3);
        rotations.assign(r.raw_rotations, r.raw_rotations + N*4);
        sh_coeffs.assign(r.raw_sh_coeffs, r.raw_sh_coeffs + N*mc3);
        opacities.assign(r.raw_opacities, r.raw_opacities + N);
    }
};

// Training configuration for VulkanTrainer.
// Separate from the legacy TrainConfig used by the CPU trainer.
struct VkTrainingConfig {
    int   max_steps           = 30000;
    float pos_lr_init         = 1.6e-4f;
    float pos_lr_final        = 1.6e-6f;
    int   sh_degree_max       = 3;      // final SH degree
    int   sh_degree_warmup    = 1000;   // steps between SH degree increments

    // Loss: combined L1 + DSSIM weight (0 = L1-only, 0.2 = Python reference default)
    float lambda_dssim          = 0.2f;

    // Densification hyperparameters (used in Task 5 — MCMC densification)
    float densify_grad_thresh   = 2e-4f;
    float opacity_thresh        = 0.005f;
    int   densify_from_step     = 500;   // 0 = disabled (skips densification entirely)
    int   densify_until_step    = 15000;
    int   densify_interval      = 100;
    float densify_percent_dense = 0.01f;
};
