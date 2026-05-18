#include "vulkan/regularization_pass.h"

#include "regularization_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
}

RegularizationPass::RegularizationPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(regularization_spv),
        static_cast<std::size_t>(regularization_spv_len));

    std::vector<VkDescriptorType> binding_types(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types,
        sizeof(RegularizationPushConstants),
        4);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void RegularizationPass::bind_buffers(VkBuffer active_opacities,
                                      VkBuffer active_scales,
                                      VkBuffer d_raw_opacities,
                                      VkBuffer d_raw_scales) {
    pipeline_->update_ssbo(descriptor_set_, regularization_bind::ACTIVE_OPACITIES, active_opacities);
    pipeline_->update_ssbo(descriptor_set_, regularization_bind::ACTIVE_SCALES, active_scales);
    pipeline_->update_ssbo(descriptor_set_, regularization_bind::D_RAW_OPACITIES, d_raw_opacities);
    pipeline_->update_ssbo(descriptor_set_, regularization_bind::D_RAW_SCALES, d_raw_scales);
}

void RegularizationPass::record(VkCommandBuffer cmd,
                                uint32_t N,
                                float opacity_reg,
                                float scale_reg) {
    if (descriptor_set_ == VK_NULL_HANDLE) {
        throw std::runtime_error("RegularizationPass::record called before bind_buffers");
    }
    if (N == 0u || (opacity_reg == 0.0f && scale_reg == 0.0f)) return;

    RegularizationPushConstants pc{};
    pc.N = N;
    pc.opacity_reg = opacity_reg;
    pc.scale_reg = scale_reg;
    pc.inv_N = 1.0f / static_cast<float>(N);

    const uint32_t groups = (N + kLocalSize - 1u) / kLocalSize;
    pipeline_->record(cmd, descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}
