// SP-2 T17: RasterizePass — wraps rasterize.comp (spec §4.8.7).
//
// Bindings (mirroring rasterize_bind::):
//   0..4  read-only SSBOs: values_sorted, tile_ranges, means2D,
//                          conic_opacity_packed, rgb
//   5..7  write-only SSBOs: out_image (CHW channel-first), transmittance, n_contrib
//   8     uniform buffer:   RasterizeUBO (bg_r, bg_g, bg_b, _pad)
//
// Dispatch: one 16x16 workgroup per tile. The shader itself is documented at
// src/vulkan/shaders/rasterize.comp; this adapter is a pure dispatcher.

#include "vulkan/rasterize_pass.h"

// xxd-embedded SPIR-V (CMake build dir produces rasterize_spv.h).
#include "rasterize_spv.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

RasterizePass::RasterizePass(VulkanContext& ctx,
                             uint32_t spec_eval_3D,
                             uint32_t spec_disable_subtile_resort)
    : ctx_(ctx) {
    // --- 1. Load SPIR-V from embedded bytes --------------------------------
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(rasterize_spv),
        static_cast<std::size_t>(rasterize_spv_len));

    // --- 2. Specialization constants:
    //        constant_id 0 = spec_eval_3D
    //        constant_id 1 = spec_disable_subtile_resort (parity_mode)
    // ---------------------------------------------------------------------
    struct SpecData {
        uint32_t eval_3D;
        uint32_t disable_subtile_resort;
    };
    SpecData spec_data{spec_eval_3D, spec_disable_subtile_resort};
    std::array<VkSpecializationMapEntry, 2> spec_entries{{
        {rasterize_spec::EVAL_3D,
         static_cast<uint32_t>(offsetof(SpecData, eval_3D)),
         sizeof(uint32_t)},
        {rasterize_spec::DISABLE_SUBTILE_RESORT,
         static_cast<uint32_t>(offsetof(SpecData, disable_subtile_resort)),
         sizeof(uint32_t)},
    }};
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries   = spec_entries.data();
    spec_info.dataSize      = sizeof(SpecData);
    spec_info.pData         = &spec_data;

    // --- 3. Descriptor layout: 12 SSBOs + 2 UBOs (bindings 8, 13) ----------
    std::vector<VkDescriptorType> binding_types(14,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[rasterize_bind::RASTER_UBO]        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding_types[rasterize_bind::RASTER_EVAL3D_UBO] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/sizeof(RasterizePushConstants),
        /*max_descriptor_sets=*/4,
        /*spec_info=*/&spec_info);

    // --- 4. Allocate the descriptor set once -------------------------------
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
    // Binding 8: RasterizeUBO (background colour, 16 bytes).
    pipeline_->update_ubo(descriptor_set_,
                          rasterize_bind::RASTER_UBO,
                          b.raster_ubo,
                          sizeof(RasterizeUBO));
    // eval_3D bindings 9..13.
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::GAUSS2SCREEN,  b.gauss2screen);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::OPACITIES_2D,  b.opacities_2d);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::COV3D_INV,     b.cov3D_inv);
    pipeline_->update_ssbo(descriptor_set_, rasterize_bind::MEAN_OFFSET,   b.mean_offset);
    pipeline_->update_ubo(descriptor_set_,
                          rasterize_bind::RASTER_EVAL3D_UBO,
                          b.raster_eval3d_ubo,
                          sizeof(RasterEval3DUBO));
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
