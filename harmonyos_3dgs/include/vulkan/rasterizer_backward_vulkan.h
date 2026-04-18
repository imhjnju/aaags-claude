// rasterizer_backward_vulkan.h — SP-3 T22: high-level backward rasterizer
// adapter over RasterizeBackwardPass.
//
// RasterizerBackwardVulkan::backward() mirrors RasterizerBackwardCPU::backward()
// but executes rasterize_backward.comp on the GPU.
//
// Usage:
//   RasterizerBackwardVulkan bwd(ctx);
//   bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad, alloc);
//
// Preconditions:
//   - cache.eval_3D must be false (EVAL_3D=true path is not implemented)
//   - The output SSBOs are zeroed before dispatch (the adapter handles this)
//   - rgrad.allocate_and_zero() is called internally

#pragma once

#include "train_types.h"
#include "vulkan/vk_context.h"
#include "vulkan/rasterize_backward_pass.h"

#include <memory>

class RasterizerBackwardVulkan {
public:
    explicit RasterizerBackwardVulkan(VulkanContext& ctx);
    ~RasterizerBackwardVulkan();

    RasterizerBackwardVulkan(const RasterizerBackwardVulkan&)            = delete;
    RasterizerBackwardVulkan& operator=(const RasterizerBackwardVulkan&) = delete;

    /// Run the backward rasterizer pass.
    ///
    /// \param pre           Preprocessor outputs (means2D, conics, opacities_2d, rgb)
    /// \param bin           Binning outputs (tile_ranges, values_sorted, num_tiles)
    /// \param num_gaussians Total number of Gaussians (N) — must match the N used
    ///                      when pre/bin were constructed.  Passed explicitly to
    ///                      avoid the UB of deriving N from max(values_sorted)+1
    ///                      when some Gaussians are culled and absent from the list.
    /// \param cam           Camera (provides W, H)
    /// \param cfg           Render config (provides bg_color, tile_w/h)
    /// \param cache         Forward pass cache (T_final, n_contrib)
    /// \param dL_dpixels    Loss gradient w.r.t. output pixels, pixel-major [H*W*3]
    /// \param rgrad         Output gradients (allocated and zeroed internally)
    /// \param alloc         Frame allocator for rgrad arrays
    ///
    /// \throws std::runtime_error if cache.eval_3D is true (not supported)
    void backward(const PreprocessOutput& pre,
                  const BinningOutput& bin,
                  int num_gaussians,
                  const Camera& cam,
                  const RenderConfig& cfg,
                  const ForwardCache& cache,
                  const float* dL_dpixels,
                  RasterGradOutput& rgrad,
                  FrameAllocator& alloc);

private:
    VulkanContext& ctx_;
    std::unique_ptr<RasterizeBackwardPass> pass_;
};
