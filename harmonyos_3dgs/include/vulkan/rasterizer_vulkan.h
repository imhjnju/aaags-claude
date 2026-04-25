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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class VulkanBuffer;

class RasterizerVulkan : public Rasterizer {
public:
    explicit RasterizerVulkan(VulkanContext& ctx, bool eval_3D = false);
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

    // -- Phase 4 / Milestone A: traced rasterize --------------------------
    //
    // Runs the eval_3D rasterize pipeline with spec_trace_enabled=1. Allocates
    // the 17 trace SSBOs sized per cascade_trace.h constants (K=
    // selected_tiles.size()), zero-fills them, populates `SlotLookup` with
    // -1 for unselected tiles and 0..K-1 for the selected tile IDs,
    // dispatches, then downloads every buffer into TraceDump. Caller writes
    // NPYs from that struct (the library layer does not depend on the NPY
    // writer in tests/golden/).
    //
    // The current rasterize.comp body does NOT emit trace writes yet, so all
    // downloaded TraceDump buffers are zero-filled; only shape/dtype match is
    // guaranteed. Milestones B..E will populate them level by level.
    struct TraceDump {
        uint32_t K = 0;
        uint32_t num_tiles = 0;
        std::vector<int32_t>  slot_lookup;        // [num_tiles]
        std::vector<float>    tail_depths;        // [K,512,16,64]
        std::vector<int32_t>  tail_ids;           // [K,512,16,64]
        std::vector<uint32_t> tail_wcur;          // [K]
        std::vector<float>    mid_depths;         // [K,1024,16,4,8]
        std::vector<int32_t>  mid_ids;            // [K,1024,16,4,8]
        std::vector<uint32_t> mid_wcur;           // [K]
        std::vector<float>    head_ins_depth;     // [K,256,4096]
        std::vector<float>    head_ins_alpha;     // [K,256,4096]
        std::vector<int32_t>  head_ins_gid;       // [K,256,4096]
        std::vector<uint32_t> head_ins_cursor;    // [K,256]
        std::vector<float>    head_blend_depth;   // [K,256,4096]
        std::vector<float>    head_blend_alpha;   // [K,256,4096]
        std::vector<float>    head_blend_T;       // [K,256,4096]
        std::vector<int32_t>  head_blend_gid;     // [K,256,4096]
        std::vector<uint32_t> head_blend_cursor;  // [K,256]
    };

    TraceDump rasterize_traced(const PreprocessOutput& preprocess,
                               const BinningOutput& binning,
                               const Camera& camera,
                               const RenderConfig& config,
                               const std::vector<uint32_t>& selected_tiles,
                               float* output_image,
                               ForwardCache* cache = nullptr);

private:
    VulkanContext& ctx_;
    bool eval_3D_ = false;
    std::unique_ptr<RasterizePass> pass_;
    // Lazily-constructed trace-enabled pass (spec_trace_enabled=1, eval_3D=1).
    // Only built when rasterize_traced() is first called.
    std::unique_ptr<RasterizePass> traced_pass_;

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
