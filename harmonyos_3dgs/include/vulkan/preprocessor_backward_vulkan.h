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
    explicit PreprocessorBackwardVulkan(VulkanContext& ctx);
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
