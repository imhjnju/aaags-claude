#pragma once

#include "types.h"
#include "train_types.h"
#include "densification.h"
#include "mcmc_densification.h"
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
// training step. The optimizer is VulkanAdam (GPU compute). By default all 6
// parameter groups are updated via GPU dispatch each step; raw params are then
// downloaded back to CPU for the next forward pass activation.
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

    // One training step. Returns L1 loss value. apply_update=false is for a
    // terminal parity step: it runs forward/backward but skips Adam and all
    // post-update maintenance.
    float step(const Camera& cam,
               const RenderConfig& cfg,
               const float* target_image,   // [3*H*W] CHW channel-first
               int target_W,
               int target_H,
               bool apply_update = true);

    // Run forward (preprocess + bin + sort + rasterize) and L1+DSSIM loss
    // computation only. No backward, no Adam, no densification, no
    // position-noise injection. Used by VK-vs-CUDA parity harnesses on
    // eval_3D=true paths (where the backward pass is not yet supported and
    // would throw). Does NOT advance step_count_.
    //
    // On success, the same accessors that step() populates are valid:
    //   rendered_image(), captured_*() (when intermediate capture is
    //   enabled), and last_loss().
    //
    // Returns the scalar L1 + lambda_dssim*(1-SSIM) loss (the value step()
    // would have at its loss-computation point).
    float forward_only(const Camera& cam,
                       const RenderConfig& cfg,
                       const float* target_image,   // [3*H*W] CHW channel-first
                       int target_W,
                       int target_H);

    int step_count() const { return step_count_; }
    float last_loss() const { return last_loss_; }
    int active_sh_degree() const { return active_sh_degree_; }

    // Access current raw parameters (for inspection/checkpointing).
    const RawGaussianParams& raw_params() const { return raw_view_; }
    const std::vector<float>& filter_3D() const { return act_filter_3D_; }

#ifdef GS3D_TESTING
    mcmc::DensifyResult apply_mcmc_densification_for_test(
        float opacity_thresh,
        int cap_max,
        const mcmc::DensifySamplePlan& plan);
    const std::vector<float>& filter_3D_for_test() const { return act_filter_3D_; }
    size_t last_replay_order_count_for_test() const { return last_replay_order_count_; }
#endif

    void download_adam_moments(int group_idx,
                               std::vector<float>& m_out,
                               std::vector<float>& v_out) const {
        vulkan_adam_.download_group_moments(group_idx, m_out, v_out);
    }

    // Access the last rendered image (CHW [3*H*W] float in [0,1]).
    // Valid after the first call to step(). Size is cam_height * cam_width * 3.
    const float* rendered_image() const { return image_.data(); }
    int rendered_image_size() const { return static_cast<int>(image_.size()); }

    // Reset params and zero Adam state for oracle per-step testing.
    // Injects new_raw into all owned param vectors + GPU buffers,
    // zeros all Adam moment buffers, and resets step_count_ to 0.
    // Call before each oracle step to test algorithm correctness
    // independent of FP accumulation.
    void reset_for_oracle(const RawGaussianParams& new_raw);

    // Gradient capture — for testing only. Enable before calling step().
    // Each step() call overwrites the previously captured gradients.
    void enable_gradient_capture(bool on) { capture_grads_ = on; }
    const std::vector<float>& captured_grad_positions()  const { return captured_grad_positions_; }
    const std::vector<float>& captured_grad_scales()     const { return captured_grad_scales_; }
    const std::vector<float>& captured_grad_rotations()  const { return captured_grad_rotations_; }
    const std::vector<float>& captured_grad_sh()         const { return captured_grad_sh_; }
    const std::vector<float>& captured_grad_opacities()  const { return captured_grad_opacities_; }

    void enable_adam_capture(bool on) { capture_adam_ = on; }
    const std::vector<std::vector<float>>& captured_adam_m() const { return captured_adam_m_; }
    const std::vector<std::vector<float>>& captured_adam_v() const { return captured_adam_v_; }

    // Intermediate forward-pass capture — for VK-vs-CUDA parity tests only.
    // When enabled, step() copies per-Gaussian and per-tile forward-pass
    // intermediates into CPU-side std::vector fields after the rasterize pass
    // finishes but before the backward pass clobbers state. Zero overhead when
    // disabled (no extra buffers allocated, no GPU-side sync added).
    void enable_intermediate_capture(bool enable) { capture_intermediates_ = enable; }
    const std::vector<float>&    captured_means2D()             const { return captured_means2D_; }
    const std::vector<float>&    captured_conic_opacity()       const { return captured_conic_opacity_; }
    const std::vector<float>&    captured_rgb()                 const { return captured_rgb_; }
    const std::vector<int>&      captured_radii()               const { return captured_radii_; }
    const std::vector<int>&      captured_tiles_touched()       const { return captured_tiles_touched_; }
    const std::vector<float>&    captured_depths()              const { return captured_depths_; }
    const std::vector<float>&    captured_radius_f()            const { return captured_radius_f_; }
    const std::vector<float>&    captured_gauss2screen()        const { return captured_gauss2screen_; }
    const std::vector<int>&      captured_sorted_gaussian_ids() const { return captured_sorted_gaussian_ids_; }
    const std::vector<int>&      captured_tile_offsets()        const { return captured_tile_offsets_; }
    const std::vector<float>&    captured_T_final()             const { return captured_T_final_; }
    const std::vector<int>&      captured_n_contrib()           const { return captured_n_contrib_; }

    void enable_backward_diagnostic_capture(bool enable);
    const std::vector<float>& captured_bwd_d_means2D() const { return captured_bwd_d_means2D_; }
    const std::vector<float>& captured_bwd_d_conics()  const { return captured_bwd_d_conics_; }
    const std::vector<float>& captured_bwd_d_opacity()      const { return captured_bwd_d_opacity_; }
    const std::vector<float>& captured_bwd_d_rgb()          const { return captured_bwd_d_rgb_; }
    const std::vector<float>& captured_bwd_d_gauss2screen() const { return captured_bwd_d_gauss2screen_; }
    const std::vector<float>& captured_bwd_d_fabc()         const { return captured_bwd_d_fabc_; }
    const std::vector<float>& captured_bwd_d_cov3D()   const { return captured_bwd_d_cov3D_; }
    const std::vector<float>& captured_bwd_d_M()       const { return captured_bwd_d_M_; }
    const std::vector<float>& captured_bwd_d_scale()   const { return captured_bwd_d_scale_; }
    const std::vector<float>& captured_bwd_d_R()       const { return captured_bwd_d_R_; }
    const std::vector<float>& captured_bwd_d_qn()      const { return captured_bwd_d_qn_; }

private:
    void activate_params();   // raw_ → g_ (exp/sigmoid/normalize)
    // Re-allocate GPU buffers after Gaussian count changes.
    void reallocate_for_n(int new_N, int old_N = -1);

    // Shared forward+loss path used by both step() and forward_only().
    // Runs (alloc reset, activate, preprocess, bin, sort, rasterize,
    // optional intermediate capture, combined loss). Updates last_loss_,
    // last_bin_R_, last_bin_N_, image_, dL_dpixels_. Does NOT touch
    // backward / Adam / densification / step_count_.
    //
    // Out-params reference FrameAllocator memory and are valid until the
    // next alloc_.reset() (i.e. until the next call to step()/forward_only()).
    // The backward path inside step() reads them in-place.
    float run_forward_and_loss(const Camera& cam,
                               const RenderConfig& cfg,
                               const float* target,
                               int W,
                               int H,
                               PreprocessOutput& out_pre,
                               BinningOutput& out_bin,
                               ForwardCache& out_cache);
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
    std::vector<float> act_filter_3D_;  // [N]     copied from input model, or zeros
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

    // Gradient capture (for testing). When capture_grads_ is true, each call
    // to step() copies the computed gradients (after regularization, before
    // GPU Adam upload) into captured_grad_* vectors.
    bool                       capture_grads_ = false;
    std::vector<float>         captured_grad_positions_;   // [N*3]
    std::vector<float>         captured_grad_scales_;      // [N*3]
    std::vector<float>         captured_grad_rotations_;   // [N*4]
    std::vector<float>         captured_grad_sh_;          // [N*max_coeffs*3]
    std::vector<float>         captured_grad_opacities_;   // [N]

    bool                       capture_adam_ = false;
    std::vector<std::vector<float>> captured_adam_m_;      // six Adam groups
    std::vector<std::vector<float>> captured_adam_v_;      // six Adam groups

    // Intermediate forward-pass capture. All fields remain empty unless
    // capture_intermediates_ is enabled; vectors are populated inside step()
    // after rasterize and before backward.
    bool                       capture_intermediates_ = false;
    std::vector<float>         captured_means2D_;              // [N*2]
    std::vector<float>         captured_conic_opacity_;        // [N*4] {a,b,c,opacity}
    std::vector<float>         captured_rgb_;                  // [N*3]
    std::vector<int>           captured_radii_;                // [N]
    std::vector<int>           captured_tiles_touched_;        // [N]
    std::vector<float>         captured_depths_;               // [N]
    std::vector<float>         captured_radius_f_;             // [N*2]
    std::vector<float>         captured_gauss2screen_;         // [N*16]
    std::vector<int>           captured_sorted_gaussian_ids_;  // [R] — flat (values_sorted)
    std::vector<int>           captured_tile_offsets_;         // [num_tiles*2] (start, end) flat
    std::vector<float>         captured_T_final_;              // [H*W]
    std::vector<int>           captured_n_contrib_;            // [H*W]

    bool                       capture_backward_diagnostics_ = false;
    std::vector<float>         captured_bwd_d_means2D_;        // [N*2]
    std::vector<float>         captured_bwd_d_conics_;         // [N*3]
    std::vector<float>         captured_bwd_d_opacity_;        // [N]
    std::vector<float>         captured_bwd_d_rgb_;            // [N*3]
    std::vector<float>         captured_bwd_d_gauss2screen_;   // [N*16]
    std::vector<float>         captured_bwd_d_fabc_;           // [N*3]
    std::vector<float>         captured_bwd_d_cov3D_;          // [N*6]
    std::vector<float>         captured_bwd_d_M_;              // [N*9]
    std::vector<float>         captured_bwd_d_scale_;          // [N*3]
    std::vector<float>         captured_bwd_d_R_;              // [N*9]
    std::vector<float>         captured_bwd_d_qn_;             // [N*4]

    // 1-indexed step counter (incremented before each GPU Adam dispatch).
    int step_count_ = 0;

    size_t last_replay_order_count_ = 0u;

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
