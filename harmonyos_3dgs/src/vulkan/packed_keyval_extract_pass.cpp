#include "vulkan/sort_passes.h"

#include "packed_keyval_extract_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
}

PackedKeyvalExtractPass::PackedKeyvalExtractPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(packed_keyval_extract_spv),
        static_cast<std::size_t>(packed_keyval_extract_spv_len));

    std::vector<VkDescriptorType> binding_types(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types,
        /*push_constant_bytes=*/sizeof(PackedKeyvalExtractPushConstants),
        /*max_descriptor_sets=*/4);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void PackedKeyvalExtractPass::bind_buffers(VkBuffer keyvals_sorted,
                                           VkBuffer values_sorted,
                                           VkBuffer tile_ranges) {
    pipeline_->update_ssbo(descriptor_set_, packed_keyval_extract_bind::KEYVALS_SORTED, keyvals_sorted);
    pipeline_->update_ssbo(descriptor_set_, packed_keyval_extract_bind::VALUES_SORTED, values_sorted);
    pipeline_->update_ssbo(descriptor_set_, packed_keyval_extract_bind::TILE_RANGES, tile_ranges);
}

void PackedKeyvalExtractPass::dispatch_record(VkCommandBuffer cmd,
                                              uint32_t num_elements,
                                              uint32_t num_tiles) {
    if (descriptor_set_ == VK_NULL_HANDLE) {
        throw std::runtime_error("PackedKeyvalExtractPass::dispatch_record called before bind_buffers");
    }
    if (num_elements == 0u) return;
    if (num_tiles == 0u) {
        throw std::runtime_error("PackedKeyvalExtractPass: num_tiles must be > 0 when num_elements > 0");
    }

    PackedKeyvalExtractPushConstants pc{};
    pc.num_elements = num_elements;
    pc.num_tiles = num_tiles;

    const uint32_t groups = (num_elements + kLocalSize - 1u) / kLocalSize;
    pipeline_->record(cmd, descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}
