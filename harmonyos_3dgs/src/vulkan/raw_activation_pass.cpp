#include "vulkan/raw_activation_pass.h"

#include "raw_activation_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
}

RawActivationPass::RawActivationPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(raw_activation_spv),
        static_cast<std::size_t>(raw_activation_spv_len));

    std::vector<VkDescriptorType> binding_types(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types,
        sizeof(RawActivationPushConstants),
        4);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void RawActivationPass::bind_buffers(VkBuffer raw_scales,
                                     VkBuffer raw_rotations,
                                     VkBuffer raw_opacities,
                                     VkBuffer raw_sh_dc,
                                     VkBuffer raw_sh_rest,
                                     VkBuffer active_scales,
                                     VkBuffer active_rotations,
                                     VkBuffer active_opacities,
                                     VkBuffer active_sh) {
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::RAW_SCALES, raw_scales);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::RAW_ROTATIONS, raw_rotations);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::RAW_OPACITIES, raw_opacities);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::RAW_SH_DC, raw_sh_dc);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::RAW_SH_REST, raw_sh_rest);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::ACTIVE_SCALES, active_scales);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::ACTIVE_ROTATIONS, active_rotations);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::ACTIVE_OPACITIES, active_opacities);
    pipeline_->update_ssbo(descriptor_set_, raw_activation_bind::ACTIVE_SH, active_sh);
}

void RawActivationPass::record(VkCommandBuffer cmd, uint32_t N, uint32_t K) {
    if (descriptor_set_ == VK_NULL_HANDLE) {
        throw std::runtime_error("RawActivationPass::record called before bind_buffers");
    }
    if (N == 0u || K == 0u) return;

    RawActivationPushConstants pc{};
    pc.N = N;
    pc.K = K;

    const uint32_t groups = (N + kLocalSize - 1u) / kLocalSize;
    pipeline_->record(cmd, descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}

void RawActivationPass::dispatch_sync(uint32_t N, uint32_t K) {
    VkCommandBuffer cmd = ctx_.allocatePrimary();
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd, N, K);
    vkEndCommandBuffer(cmd);
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);
}
