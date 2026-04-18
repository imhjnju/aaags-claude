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
};
