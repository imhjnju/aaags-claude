#pragma once

#include "types.h"
#include "train_types.h"
#include "densification.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vulkan_adam.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/rasterizer_backward_vulkan.h"
#include "vulkan/preprocessor_backward_vulkan.h"

#include <vector>
#include <memory>

// VulkanTrainer wires the full forward→backward→Adam pipeline for one
// training step. The optimizer is VulkanAdam (GPU compute). All 6 parameter
// groups are updated via GPU dispatch each step; raw params are then downloaded
// back to CPU for the next forward pass activation.
class VulkanTrainer {
public:
    // init_g: initial Gaussian parameters (activated values)
    // init_raw: initial raw (pre-activation) parameters — must be same count as init_g
    VulkanTrainer(VulkanContext& ctx,
                  const GaussianData& init_g,
                  const RawGaussianParams& init_raw,
                  int sh_degree,
                  int cam_width,
                  int cam_height,
                  const VkTrainingConfig& tcfg = VkTrainingConfig{});

    // One training step. Returns L1 loss value.
    float step(const Camera& cam,
               const RenderConfig& cfg,
               const float* target_image,   // [H*W*3] pixel-major
               int target_W,
               int target_H);

    int step_count() const { return step_count_; }
    float last_loss() const { return last_loss_; }
    int active_sh_degree() const { return active_sh_degree_; }

    // Access current raw parameters (for inspection/checkpointing).
    const RawGaussianParams& raw_params() const { return raw_view_; }

    // Access the last rendered image (pixel-major [H*W*3] float in [0,1]).
    // Valid after the first call to step(). Size is cam_height * cam_width * 3.
    const float* rendered_image() const { return image_.data(); }
    int rendered_image_size() const { return static_cast<int>(image_.size()); }

private:
    void activate_params();   // raw_ → g_ (exp/sigmoid/normalize)
    // Re-allocate GPU buffers and re-initialize Adam groups after Gaussian count changes.
    void reallocate_for_n(int new_N);
    // Inject covariance-scaled Gaussian noise into positions of near-dead Gaussians.
    // Called after GPU Adam download. Matches train.py:141-148.
    void inject_position_noise(float pos_lr);

    VulkanContext&             ctx_;
    int                        N_;
    int                        max_coeffs_;

    // Raw parameters — GPU Adam updates these in-place (via GPU bufs), then
    // downloads back to CPU for activation.
    // Stored as vectors so they own the memory.
    std::vector<float> raw_positions_;    // [N*3]
    std::vector<float> raw_scales_;       // [N*3]
    std::vector<float> raw_rotations_;   // [N*4]
    std::vector<float> raw_sh_coeffs_;   // [N*max_coeffs*3]
    std::vector<float> raw_opacities_;   // [N]

    // Non-owning view into the above vectors.
    RawGaussianParams raw_view_;

    // Activated parameters (rebuilt each step from raw_).
    std::vector<float> act_positions_;   // [N*3]   (= raw, no activation)
    std::vector<float> act_scales_;      // [N*3]   exp(raw_scales)
    std::vector<float> act_rotations_;  // [N*4]   normalized quaternion
    std::vector<float> act_opacities_;  // [N]     sigmoid(raw_opacities)
    std::vector<float> act_sh_coeffs_;  // [N*max_coeffs*3] (= raw, no activation)
    GaussianData       g_;               // non-owning view into act_* buffers

    // Per-frame output buffers.
    std::vector<float> image_;           // [H*W*3]
    std::vector<float> dL_dpixels_;     // [H*W*3]

    FrameAllocator alloc_;
    float          last_loss_ = 0.0f;

    // GPU Adam optimizer (one pipeline shared across 6 groups).
    VulkanAdam vulkan_adam_;

    // Persistent GPU raw param buffers — one per Adam group.
    // Group 0: positions      [N*3]
    // Group 1: sh DC          [N*3]          (first N*3 floats of raw_sh_coeffs_)
    // Group 2: sh rest        [N*(max-1)*3]  (remaining floats of raw_sh_coeffs_)
    // Group 3: opacities      [N]
    // Group 4: scales         [N*3]
    // Group 5: rotations      [N*4]
    std::unique_ptr<VulkanBuffer> raw_param_gpu_bufs_[6];

    // Persistent GPU gradient buffers — one per param group.
    std::unique_ptr<VulkanBuffer> grad_positions_gpu_;   // [N*3]
    std::unique_ptr<VulkanBuffer> grad_sh_dc_gpu_;       // [N*3]
    std::unique_ptr<VulkanBuffer> grad_sh_rest_gpu_;     // [N*(max_coeffs-1)*3]
    std::unique_ptr<VulkanBuffer> grad_opacities_gpu_;   // [N]
    std::unique_ptr<VulkanBuffer> grad_scales_gpu_;      // [N*3]
    std::unique_ptr<VulkanBuffer> grad_rotations_gpu_;   // [N*4]

    // Per-Gaussian accumulated gradient norm of means2D (proxy: |d_raw_positions|).
    // Size N_, reset to zeros after each densification step.
    std::vector<float> grad_means2D_accum_;

    // 1-indexed step counter (incremented before each GPU Adam dispatch).
    int step_count_ = 0;

    // Tracks actual binner R (total Gaussian-tile pairs) from the previous step.
    // Used to size the FrameAllocator at the start of each step. After densification
    // N grows, so we scale by the N ratio and add a 4× safety margin.
    size_t last_bin_R_ = 0u;
    size_t last_bin_N_ = 0u;

    // Learning rates per group — stored so step() doesn't hardcode them.
    float group_lrs_[6] = {1.6e-4f, 2.5e-3f, 1.25e-4f, 0.05f, 0.005f, 0.001f};

    // Training configuration (LR schedule, SH warmup).
    VkTrainingConfig tcfg_;

    // Current active SH degree (incremented by SH warmup schedule).
    int active_sh_degree_ = 0;

    // Vulkan pipeline components.
    PreprocessorVulkan        preprocessor_;
    TileBinnerVulkan          binner_;
    SorterVulkan              sorter_;
    RasterizerVulkan          rasterizer_;
    RasterizerBackwardVulkan  rasterizer_bwd_;
    PreprocessorBackwardVulkan preprocessor_bwd_;
};
