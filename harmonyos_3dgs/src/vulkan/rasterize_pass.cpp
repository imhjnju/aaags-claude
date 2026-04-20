// SP-2 T17: RasterizePass — wraps rasterize.comp (spec §4.8.7).
//
// Bindings (mirroring rasterize_bind::):
//   0..4  read-only SSBOs: values_sorted, tile_ranges, means2D,
//                          conic_opacity_packed, rgb
//   5..7  write-only SSBOs: out_image (CHW), transmittance, n_contrib
//   8     uniform buffer:   RasterizeUBO (bg_r, bg_g, bg_b, _pad)
//
// Dispatch: one 16x16 workgroup per tile. The shader itself is documented at
// src/vulkan/shaders/rasterize.comp; this adapter is a pure dispatcher.

#include "vulkan/rasterize_pass.h"

// xxd-embedded SPIR-V (CMake build dir produces rasterize_spv.h).
#include "rasterize_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

RasterizePass::RasterizePass(VulkanContext& ctx)
    : ctx_(ctx) {
    // --- 1. Load SPIR-V from embedded bytes --------------------------------
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_spv),
        static_cast<std::size_t>(rasterize_spv_len));

    // --- 2. Descriptor layout: 8 SSBOs + 1 UBO (binding 8) -----------------
    std::vector<VkDescriptorType> binding_types(9,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[rasterize_bind::RASTER_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/sizeof(RasterizePushConstants),
        /*max_descriptor_sets=*/4);

    // --- 3. Allocate the descriptor set once -------------------------------
    // Same reasoning as PreprocessPass: the pool is sized for 4 sets and does
    // not support free-descriptor-set. Allocate up front and update in place.
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void RasterizePass::bind_buffers(const Buffers& b) {
    // Bindings 0..7 are SSBOs.
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::VALUES_SORTED,        b.values_sorted);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::TILE_RANGES,          b.tile_ranges);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::MEANS2D,              b.means2D);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::CONIC_OPACITY_PACKED, b.conic_opacity_packed);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::RGB,                  b.rgb);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::OUT_IMAGE,            b.out_image);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::TRANSMITTANCE,        b.transmittance);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::N_CONTRIB,            b.n_contrib);
    // Binding 8 is the UBO (RasterizeUBO, 16 bytes).
    pipeline_->update_ubo(descriptor_set_,
                          rasterize_bind::RASTER_UBO,
                          b.raster_ubo,
                          sizeof(RasterizeUBO));
}

void RasterizePass::dispatch_sync(uint32_t num_gaussians,
                                  uint32_t image_width, uint32_t image_height,
                                  uint32_t num_tiles_x, uint32_t num_tiles_y) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizePass::dispatch_sync called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;

    RasterizePushConstants pc{};
    pc.num_gaussians = num_gaussians;
    pc.image_width   = image_width;
    pc.image_height  = image_height;
    pc.num_tiles_x   = num_tiles_x;
    pc.num_tiles_y   = num_tiles_y;

    // One 16x16 workgroup per tile — spec §4.8.7 / rasterize.comp header.
    pipeline_->dispatch_sync(descriptor_set_,
                             num_tiles_x, num_tiles_y, 1u,
                             &pc, sizeof(pc));
}

void RasterizePass::record(VkCommandBuffer cmd,
                           uint32_t num_gaussians,
                           uint32_t image_width, uint32_t image_height,
                           uint32_t num_tiles_x, uint32_t num_tiles_y) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "RasterizePass::record called before bind_buffers()");
    if (num_tiles_x == 0u || num_tiles_y == 0u) return;

    RasterizePushConstants pc{};
    pc.num_gaussians = num_gaussians;
    pc.image_width   = image_width;
    pc.image_height  = image_height;
    pc.num_tiles_x   = num_tiles_x;
    pc.num_tiles_y   = num_tiles_y;

    pipeline_->record(cmd, descriptor_set_,
                      num_tiles_x, num_tiles_y, 1u,
                      &pc, sizeof(pc));
}
