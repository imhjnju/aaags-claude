#include "vulkan/l1_loss_pass.h"

#include "l1_loss_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
}

L1LossPass::L1LossPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(l1_loss_spv),
        static_cast<std::size_t>(l1_loss_spv_len));

    std::vector<VkDescriptorType> binding_types(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types,
        /*push_constant_bytes=*/sizeof(L1LossPushConstants),
        /*max_descriptor_sets=*/4);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void L1LossPass::bind_buffers(VkBuffer rendered,
                              VkBuffer target,
                              VkBuffer dL_dpixels,
                              VkBuffer partials) {
    pipeline_->update_ssbo(descriptor_set_, l1_loss_bind::RENDERED, rendered);
    pipeline_->update_ssbo(descriptor_set_, l1_loss_bind::TARGET, target);
    pipeline_->update_ssbo(descriptor_set_, l1_loss_bind::DLDPIX, dL_dpixels);
    pipeline_->update_ssbo(descriptor_set_, l1_loss_bind::PARTIALS, partials);
}

void L1LossPass::dispatch_sync(uint32_t num_elements) {
    if (descriptor_set_ == VK_NULL_HANDLE) {
        throw std::runtime_error("L1LossPass::dispatch_sync called before bind_buffers");
    }
    if (num_elements == 0u) return;

    L1LossPushConstants pc{};
    pc.num_elements = num_elements;
    pc.scale = 1.0f / static_cast<float>(num_elements);

    const uint32_t groups = (num_elements + kLocalSize - 1u) / kLocalSize;
    pipeline_->dispatch_sync(descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}
