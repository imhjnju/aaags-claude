// rasterizer_vulkan.h -- SP-2 T17: Rasterizer adapter driving rasterize.comp.
//
// Implements the Rasterizer interface by (1) repacking per-Gaussian inputs
// into the shader's expected layouts, (2) uploading them plus the sorted
// binning pair stream + tile ranges to host-visible SSBOs, (3) dispatching
// rasterize.comp via RasterizePass, and (4) downloading the CHW output image
// (plus optional T_final / n_contrib cache entries).
//
// Input from PreprocessOutput:
//   means2D               — interleaved [x,y] per Gaussian
//   conics + opacities_2d — repacked inline into conic_opacity_packed[N*4]
//   rgb                   — [N*3]
// Input from BinningOutput:
//   values_sorted[R], tile_ranges[num_tiles*2]
// Input from Camera:
//   width, height
// Input from RenderConfig:
//   bg_color[3], tile_w, tile_h
// Output to `output_image`:   float[3*H*W] (CHW layout — written by shader)
// Output to `cache` (opt):    T_final[H*W], n_contrib[H*W]
//
// Phase 1 limitations: fresh VulkanBuffers per frame, no persistent pool,
// empty-scene fast path writes background into output_image on the CPU.
#pragma once

#include "rasterizer.h"
#include "vulkan/vk_context.h"
#include "vulkan/rasterize_pass.h"

#include <memory>

class RasterizerVulkan : public Rasterizer {
public:
    explicit RasterizerVulkan(VulkanContext& ctx);
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

private:
    VulkanContext& ctx_;
    std::unique_ptr<RasterizePass> pass_;
};
