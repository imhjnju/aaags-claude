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
#include "vulkan/vk_buffer.h"
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

    /// GPU output buffer handles for rasterize_bwd (valid after backward() or backward_record_into()).
    /// Used for GPU-to-GPU chaining with PreprocessorBackwardVulkan::backward_record_into().
    VkBuffer dL_dmeans2D_buf() const { return dlm2d_buf_ ? dlm2d_buf_->handle() : VK_NULL_HANDLE; }
    VkBuffer dL_dconics_buf()  const { return dlcon_buf_ ? dlcon_buf_->handle() : VK_NULL_HANDLE; }
    VkBuffer dL_dopacity_buf() const { return dlopa_buf_ ? dlopa_buf_->handle() : VK_NULL_HANDLE; }
    VkBuffer dL_dcolors_buf()  const { return dlcol_buf_ ? dlcol_buf_->handle() : VK_NULL_HANDLE; }

    /// Record backward pass into cmd (no submit, no download).
    /// Uploads inputs to persistent GPU buffers, calls pass_->record().
    /// Caller must: insert_compute_barrier(), call preprocessor_bwd_.backward_record_into(),
    ///   submit the CB, then call preprocessor_bwd_.download_grads().
    void backward_record_into(VkCommandBuffer cmd,
                              const PreprocessOutput& pre,
                              const BinningOutput& bin,
                              int num_gaussians,
                              const Camera& cam,
                              const RenderConfig& cfg,
                              const ForwardCache& cache,
                              const float* dL_dpixels);

private:
    VulkanContext& ctx_;
    std::unique_ptr<RasterizeBackwardPass> pass_;

    // Persistent GPU buffers — pre-allocated in prepare_for_n(), reused each
    // backward() call. Eliminates 13 vkDeviceWaitIdle/step from destructors.
    int buf_N_         = 0;
    int buf_R_         = 0;
    int buf_num_tiles_ = 0;
    int buf_HW_        = 0;

    std::unique_ptr<VulkanBuffer> tr_buf_;     // tile_ranges   [num_tiles*2] u32
    std::unique_ptr<VulkanBuffer> vs_buf_;     // values_sorted [R] u32
    std::unique_ptr<VulkanBuffer> m2d_buf_;    // means2D       [N*2] f32
    std::unique_ptr<VulkanBuffer> co_buf_;     // conic_opacity [N*4] f32
    std::unique_ptr<VulkanBuffer> col_buf_;    // colors        [N*3] f32
    std::unique_ptr<VulkanBuffer> tf_buf_;     // T_final       [HW] f32
    std::unique_ptr<VulkanBuffer> nc_buf_;     // n_contrib      [HW] u32
    std::unique_ptr<VulkanBuffer> dlpix_buf_;  // dL_dpixels    [HW*3] f32
    std::unique_ptr<VulkanBuffer> dlm2d_buf_;  // dL_dmeans2D   [N*2] f32
    std::unique_ptr<VulkanBuffer> dlcon_buf_;  // dL_dconics    [N*3] f32
    std::unique_ptr<VulkanBuffer> dlopa_buf_;  // dL_dopacity   [N] f32
    std::unique_ptr<VulkanBuffer> dlcol_buf_;  // dL_dcolors    [N*3] f32
    std::unique_ptr<VulkanBuffer> ubo_buf_;    // RasterizeBackwardUBO (32B)

    void prepare_for_n(int N, int R, int num_tiles, int HW);
};
