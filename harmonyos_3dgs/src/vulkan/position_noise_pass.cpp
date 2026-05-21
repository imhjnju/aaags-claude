#include "vulkan/position_noise_pass.h"

#include "position_noise_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
}

PositionNoisePass::PositionNoisePass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(position_noise_spv),
        static_cast<std::size_t>(position_noise_spv_len));

    std::vector<VkDescriptorType> binding_types(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types,
        sizeof(PositionNoisePushConstants),
        4);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void PositionNoisePass::bind_buffers(VkBuffer raw_positions,
                                     VkBuffer raw_scales,
                                     VkBuffer raw_rotations,
                                     VkBuffer raw_opacities) {
    pipeline_->update_ssbo(descriptor_set_, position_noise_bind::RAW_POSITIONS, raw_positions);
    pipeline_->update_ssbo(descriptor_set_, position_noise_bind::RAW_SCALES, raw_scales);
    pipeline_->update_ssbo(descriptor_set_, position_noise_bind::RAW_ROTATIONS, raw_rotations);
    pipeline_->update_ssbo(descriptor_set_, position_noise_bind::RAW_OPACITIES, raw_opacities);
    buffers_bound_ = true;
}

void PositionNoisePass::record(VkCommandBuffer cmd, uint32_t N, uint32_t step, float noise_lr, float pos_lr) {
    if (!buffers_bound_) {
        throw std::runtime_error("PositionNoisePass::record called before bind_buffers");
    }
    if (N == 0u || noise_lr == 0.0f || pos_lr == 0.0f) return;

    PositionNoisePushConstants pc{};
    pc.N = N;
    pc.step = step;
    pc.noise_lr = noise_lr;
    pc.pos_lr = pos_lr;

    const uint32_t groups = (N + kLocalSize - 1u) / kLocalSize;
    pipeline_->record(cmd, descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}

void PositionNoisePass::dispatch_sync(uint32_t N, uint32_t step, float noise_lr, float pos_lr) {
    if (!buffers_bound_) {
        throw std::runtime_error("PositionNoisePass::dispatch_sync called before bind_buffers");
    }

    VkCommandBuffer cmd = ctx_.allocatePrimary();
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd, N, step, noise_lr, pos_lr);
    vkEndCommandBuffer(cmd);
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);
}
