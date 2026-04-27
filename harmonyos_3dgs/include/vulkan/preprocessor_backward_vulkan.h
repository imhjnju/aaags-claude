// preprocessor_backward_vulkan.h — SP-3 T23: high-level backward preprocessor
// adapter over PreprocessBackwardPass.
//
// PreprocessorBackwardVulkan::backward() mirrors PreprocessorBackwardCPU::backward()
// but executes preprocess_backward.comp on the GPU.
//
// Usage:
//   PreprocessorBackwardVulkan bwd(ctx);
//   bwd.backward(g, num_gaussians, cam, cfg, cache, rgrad, raw, grads, alloc);
//
// Preconditions:
//   - cache.eval_3D must be false (EVAL_3D=true path is not implemented)
//   - cache.cov3D, cache.p_view must be non-null (populated by forward preprocessor)
//   - grads.allocate_and_zero() is called internally

#pragma once

#include "types.h"
#include "train_types.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/preprocess_backward_pass.h"

#include <memory>

class PreprocessorBackwardVulkan {
public:
    /// @param proper_ewa  Must match the value passed to the forward
    /// PreprocessorVulkan / PreprocessorCPU. Gates the h_conv_scaling chain
    /// rule in the backward shader (mirrors CUDA backward.cu:215).
    /// Default true: legacy regression tests (PreprocessorBackwardVulkan.*) pair
    /// this class with PreprocessorCPU forward, which applies h_conv
    /// UNCONDITIONALLY — so the default mirrors CPU's "always-on" behavior.
    /// Callers that drive forward with proper_ewa=false (e.g. VulkanTrainer
    /// with VkTrainingConfig::proper_ewa=false) MUST pass false explicitly to
    /// gate the h_conv chain rule off.
    explicit PreprocessorBackwardVulkan(VulkanContext& ctx,
                                        bool proper_ewa = true);
    ~PreprocessorBackwardVulkan();

    PreprocessorBackwardVulkan(const PreprocessorBackwardVulkan&)            = delete;
    PreprocessorBackwardVulkan& operator=(const PreprocessorBackwardVulkan&) = delete;

    /// Run the backward preprocessor pass.
    ///
    /// \param g             Activated GaussianData (positions, sh_coeffs, scales, rotations, opacities)
    /// \param num_gaussians Total number of Gaussians (N)
    /// \param cam           Camera (view_matrix, viewproj_matrix, cam_pos, fov, size)
    /// \param cfg           Render config (sh_degree, scale_modifier, training)
    /// \param cache         Forward pass cache (cov3D, p_view; pre->radii must be valid)
    /// \param rgrad         Rasterizer backward output (d_means2D, d_conics, d_rgb, d_opacities_2d)
    /// \param raw           Raw (pre-activation) Gaussian params (for raw_rotations in normalization)
    /// \param grads         Output gradients (allocated and zeroed internally)
    /// \param alloc         Frame allocator for grads arrays
    ///
    /// \throws std::runtime_error if cache.pre->eval_3D is true (not supported)
    void backward(const GaussianData& g,
                  int num_gaussians,
                  const Camera& cam,
                  const RenderConfig& cfg,
                  const ForwardCache& cache,
                  const RasterGradOutput& rgrad,
                  const RawGaussianParams& raw,
                  GradientOutput& grads,
                  FrameAllocator& alloc);

    /// Record backward pass into cmd, reading rgrad directly from GPU buffer handles.
    /// Eliminates the CPU round-trip: rasterize_bwd GPU outputs bind directly as inputs.
    /// Caller must submit the CB, then call download_grads() to get grads on CPU.
    void backward_record_into(VkCommandBuffer cmd,
                              const GaussianData& g,
                              int num_gaussians,
                              const Camera& cam,
                              const RenderConfig& cfg,
                              const ForwardCache& cache,
                              VkBuffer d_conics_gpu,   // from rasterize_bwd dL_dconics_buf()
                              VkBuffer d_opacity_gpu,  // from rasterize_bwd dL_dopacity_buf()
                              VkBuffer d_rgb_gpu,      // from rasterize_bwd dL_dcolors_buf()
                              VkBuffer d_means2D_gpu,  // from rasterize_bwd dL_dmeans2D_buf()
                              const RawGaussianParams& raw);

    /// Download gradient outputs to CPU after backward_record_into() + submit.
    /// grads is allocated from alloc and filled from persistent GPU output buffers.
    void download_grads(int num_gaussians, int max_coeffs,
                        GradientOutput& grads, FrameAllocator& alloc);

    /// Clear all gradient output buffers to zero before backward pass.
    /// Critical for correctness: backward shaders only write to active Gaussians,
    /// so stale data from previous steps would accumulate without clearing.
    void clear_grad_buffers(VkCommandBuffer cmd);

private:
    VulkanContext& ctx_;
    std::unique_ptr<PreprocessBackwardPass> pass_;

    // Persistent GPU buffers — pre-allocated in prepare_for_n(), reused each
    // backward() call. Eliminates 23 vkDeviceWaitIdle/step from destructors.
    int buf_N_ = 0;
    int buf_K_ = 0;  // max_coeffs = (sh_degree+1)^2

    // Input buffers
    std::unique_ptr<VulkanBuffer> pos_buf_;        // positions   [N*3] f32
    std::unique_ptr<VulkanBuffer> rad_buf_;        // radii       [N] i32
    std::unique_ptr<VulkanBuffer> cv3_buf_;        // cov3D       [N*6] f32
    std::unique_ptr<VulkanBuffer> dcon_buf_;       // d_conics    [N*3] f32
    std::unique_ptr<VulkanBuffer> dopa_buf_;       // d_opacity   [N] f32
    std::unique_ptr<VulkanBuffer> sh_buf_;         // sh_coeffs   [N*K*3] f32
    std::unique_ptr<VulkanBuffer> sc_buf_;         // scales      [N*3] f32
    std::unique_ptr<VulkanBuffer> rot_buf_;        // rotations   [N*4] f32
    std::unique_ptr<VulkanBuffer> drgb_buf_;       // d_rgb       [N*3] f32
    std::unique_ptr<VulkanBuffer> dm2d_buf_;       // d_means2D   [N*2] f32
    std::unique_ptr<VulkanBuffer> opa_in_buf_;     // opacities   [N] f32
    std::unique_ptr<VulkanBuffer> raw_rot_buf_;    // raw_rotations [N*4] f32
    std::unique_ptr<VulkanBuffer> m2d_cache_buf_;  // means2D_cache [N*2] f32
    std::unique_ptr<VulkanBuffer> pview_in_buf_;   // p_view_cache [N*3] f32
    std::unique_ptr<VulkanBuffer> cov2d_in_buf_;   // cov2D_cache  [N*3] f32
    std::unique_ptr<VulkanBuffer> c2ddet_in_buf_;  // cov2D_det    [N] f32
    std::unique_ptr<VulkanBuffer> phomw_in_buf_;   // p_hom_w      [N] f32
    // Output (gradient) buffers — zeroed before each dispatch
    std::unique_ptr<VulkanBuffer> dm3d_buf_;       // d_means3D   [N*3] f32
    std::unique_ptr<VulkanBuffer> dsh_buf_;        // d_sh        [N*K*3] f32
    std::unique_ptr<VulkanBuffer> dsc_buf_;        // d_scales    [N*3] f32
    std::unique_ptr<VulkanBuffer> drot_buf_;       // d_rotations [N*4] f32
    std::unique_ptr<VulkanBuffer> d_raw_opa_buf_;  // d_raw_opacities [N] f32
    std::unique_ptr<VulkanBuffer> ubo_buf_;        // PreprocessBackwardUBO (192B)

    void prepare_for_n(int N, int max_coeffs);
};
