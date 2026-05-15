#include "vulkan/sh_grad_split_pass.h"

#include "sh_grad_split_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
}

ShGradSplitPass::ShGradSplitPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(sh_grad_split_spv),
        static_cast<std::size_t>(sh_grad_split_spv_len));

    std::vector<VkDescriptorType> binding_types(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types,
        /*push_constant_bytes=*/sizeof(ShGradSplitPushConstants),
        /*max_descriptor_sets=*/4);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void ShGradSplitPass::bind_buffers(VkBuffer interleaved, VkBuffer dc, VkBuffer rest) {
    pipeline_->update_ssbo(descriptor_set_, sh_grad_split_bind::INTERLEAVED, interleaved);
    pipeline_->update_ssbo(descriptor_set_, sh_grad_split_bind::DC, dc);
    pipeline_->update_ssbo(descriptor_set_, sh_grad_split_bind::REST, rest);
}

void ShGradSplitPass::record(VkCommandBuffer cmd, uint32_t N, uint32_t K) {
    if (descriptor_set_ == VK_NULL_HANDLE) {
        throw std::runtime_error("ShGradSplitPass::record called before bind_buffers");
    }
    if (N == 0u || K == 0u) return;

    ShGradSplitPushConstants pc{};
    pc.N = N;
    pc.K = K;

    const uint32_t groups = (N + kLocalSize - 1u) / kLocalSize;
    pipeline_->record(cmd, descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
}
