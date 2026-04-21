// SP-2 T10: ScatterPass — wraps scatter.comp (spec §4.8.3).
//
// One thread per Gaussian; each writes tiles_touched[i] (key, value) pairs
// starting at point_offsets[i]. Key layout and iteration/rect contract are
// documented in detail at the top of src/vulkan/shaders/scatter.comp — this
// adapter is purely a dispatcher.

#include "vulkan/tile_binner_passes.h"

// xxd-embedded SPIR-V (CMake build dir produces scatter_spv.h).
#include "scatter_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kScatterLocalSize = 256;  // scatter.comp: local_size_x=256
}  // namespace

ScatterPass::ScatterPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(scatter_spv),
        static_cast<std::size_t>(scatter_spv_len));

    // 12 bindings: 11 SSBOs + 1 UBO (binding 10 = ScatterUBO).
    std::vector<VkDescriptorType> binding_types(12,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[scatter_bind::SCATTER_UBO] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/sizeof(ScatterPushConstants),
        /*max_descriptor_sets=*/4);

    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void ScatterPass::bind_buffers(const Buffers& b) {
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::MEANS2D,         b.means2D);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::DEPTHS,          b.depths);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::RADII,           b.radii);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::POINT_OFFSETS,   b.point_offsets);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::TILES_TOUCHED,   b.tiles_touched);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::KEYS_UNSORTED,   b.keys_unsorted);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::VALUES_UNSORTED, b.values_unsorted);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::RADIUS_F,        b.radius_f);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::COV3D_INV,      b.cov3D_inv);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::MEAN_OFFSET,    b.mean_offset);
    pipeline_->update_ssbo(descriptor_set_, scatter_bind::GAUSS2SCREEN,  b.gauss2screen);
    pipeline_->update_ubo(descriptor_set_,  scatter_bind::SCATTER_UBO,
                          b.scatter_ubo, sizeof(ScatterUBO));
}

void ScatterPass::dispatch_sync(uint32_t num_gaussians,
                                uint32_t num_tiles_x,
                                uint32_t num_tiles_y,
                                bool eval_3D) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "ScatterPass::dispatch_sync called before bind_buffers()");
    if (num_gaussians == 0u) return;

    ScatterPushConstants pc{};
    pc.num_gaussians = num_gaussians;
    pc.num_tiles_x   = num_tiles_x;
    pc.num_tiles_y   = num_tiles_y;
    pc.tile_w        = 16u;  // SP-2 tile size is hard-coded per spec §4.4.
    pc.tile_h        = 16u;
    pc.eval_3D       = eval_3D ? 1u : 0u;

    const uint32_t groups =
        (num_gaussians + kScatterLocalSize - 1u) / kScatterLocalSize;
    pipeline_->dispatch_sync(descriptor_set_,
                             groups, 1u, 1u,
                             &pc, sizeof(pc));
}

void ScatterPass::record(VkCommandBuffer cmd,
                         uint32_t num_gaussians,
                         uint32_t num_tiles_x,
                         uint32_t num_tiles_y,
                         bool eval_3D) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "ScatterPass::record called before bind_buffers()");
    if (num_gaussians == 0u) return;

    ScatterPushConstants pc{};
    pc.num_gaussians = num_gaussians;
    pc.num_tiles_x   = num_tiles_x;
    pc.num_tiles_y   = num_tiles_y;
    pc.tile_w        = 16u;
    pc.tile_h        = 16u;
    pc.eval_3D       = eval_3D ? 1u : 0u;

    const uint32_t groups =
        (num_gaussians + kScatterLocalSize - 1u) / kScatterLocalSize;
    pipeline_->record(cmd, descriptor_set_,
                      groups, 1u, 1u,
                      &pc, sizeof(pc));
}
