// rasterizer_vulkan.h -- SP-2 T17/T19: Rasterizer adapter driving rasterize.comp.
//
// Supports two usage patterns:
//
//   1) Layer-1 rasterize() — sync, self-contained. Uploads CPU-side inputs
//      (PreprocessOutput/BinningOutput) to fresh SSBOs, dispatches, downloads.
//   2) Layer-2 prepare_record() + record() — chained forward pipeline (T19).
//      Inputs (values_sorted/tile_ranges/means2D/conic_opacity_packed/rgb) are
//      EXTERNAL handles from Sorter+Preprocessor; the adapter allocates the
//      output image + T_final + n_contrib + RasterizeUBO buffers, binds, and
//      records the dispatch. Outputs are downloaded via download_image()
//      (+ optional download_cache()) after the caller submits and waits.
//
// Input buffer sources:
//   values_sorted        : SorterVulkan
//   tile_ranges          : SorterVulkan
//   means2D              : PreprocessorVulkan
//   conic_opacity_packed : PreprocessorVulkan   (already packed N*4)
//   rgb                  : PreprocessorVulkan
#pragma once

#include "rasterizer.h"
#include "vulkan/vk_context.h"
#include "vulkan/rasterize_pass.h"

#include <cstdint>
#include <memory>

class VulkanBuffer;

class RasterizerVulkan : public Rasterizer {
public:
    explicit RasterizerVulkan(VulkanContext& ctx,
                              bool eval_3D = false,
                              bool disable_subtile_resort = false);
    ~RasterizerVulkan() override;

    RasterizerVulkan(const RasterizerVulkan&)            = delete;
    RasterizerVulkan& operator=(const RasterizerVulkan&) = delete;

    void rasterize(const PreprocessOutput& preprocess,
                   const BinningOutput& binning,
                   const Camera& camera,
                   const RenderConfig& config,
                   float* output_image,
                   float* output_depth = nullptr,
                   ForwardCache* cache = nullptr,
                   FrameAllocator* allocator = nullptr) override;

    // -- Layer-2 record-mode API -------------------------------------------
    //
    // Allocate output image + T_final + n_contrib + RasterizeUBO. Upload the
    // UBO payload. Bind all external inputs + internal outputs to the pass.
    //
    // N_eff is NOT needed at prepare time — it is passed to record() as a
    // push constant. Only record() needs it.
    //
    // bg_color is a 3-float array — channel layout matches RasterizeUBO.
    // Buffers persist until the next prepare_record() call or destruction.
    void prepare_record(uint32_t W, uint32_t H,
                        uint32_t num_tiles_x, uint32_t num_tiles_y,
                        const float bg_color[3],
                        VkBuffer values_sorted,
                        VkBuffer tile_ranges,
                        VkBuffer means2D,
                        VkBuffer conic_opacity_packed,
                        VkBuffer rgb,
                        VkBuffer gauss2screen,
                        VkBuffer opacities_2d,
                        VkBuffer cov3D_inv,
                        VkBuffer mean_offset,
                        const Camera& cam);

    // Record the rasterize dispatch into cmd. prepare_record() must have
    // been called. No internal barrier — caller submits after this returns.
    void record(VkCommandBuffer cmd,
                uint32_t N_eff, uint32_t W, uint32_t H,
                uint32_t num_tiles_x, uint32_t num_tiles_y);

    // Post-record download helpers. Must be called AFTER the caller has
    // submitted + waited on the command buffer that contained record().
    void download_image(float* dst, uint32_t W, uint32_t H);
    void download_cache(float* T_final, int* n_contrib, uint32_t HW);

    // Output handles valid after prepare_record().
    VkBuffer out_image_buf()     const;
    VkBuffer transmittance_buf() const;
    VkBuffer n_contrib_buf()     const;

private:
    VulkanContext& ctx_;
    bool eval_3D_ = false;
    std::unique_ptr<RasterizePass> pass_;

    // Layer-2 persistent buffers.
    std::unique_ptr<VulkanBuffer> r_img_;           // [3*H*W]   f32 CHW (channel-first)
    std::unique_ptr<VulkanBuffer> r_tfinal_;        // [H*W]     f32
    std::unique_ptr<VulkanBuffer> r_ncontrib_;      // [H*W]     u32
    std::unique_ptr<VulkanBuffer> r_ubo_;           // RasterizeUBO (16 bytes std140)
    std::unique_ptr<VulkanBuffer> r_eval3d_ubo_;    // RasterEval3DUBO (96 bytes std140)
    std::unique_ptr<VulkanBuffer> r_dummy4_;        // 4-byte dummy for unbound eval_3D SSBOs
    uint32_t r_W_   = 0;
    uint32_t r_H_   = 0;
    uint32_t r_ntx_ = 0;
    uint32_t r_nty_ = 0;
};
