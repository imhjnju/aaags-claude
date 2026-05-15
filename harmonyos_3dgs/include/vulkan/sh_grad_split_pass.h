#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace sh_grad_split_bind {
constexpr uint32_t INTERLEAVED = 0;
constexpr uint32_t DC          = 1;
constexpr uint32_t REST        = 2;
}

struct ShGradSplitPushConstants {
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
static_assert(sizeof(ShGradSplitPushConstants) == 16, "ShGradSplitPushConstants must be 16 bytes");

class ShGradSplitPass {
public:
    explicit ShGradSplitPass(VulkanContext& ctx);
    ~ShGradSplitPass() = default;

    ShGradSplitPass(const ShGradSplitPass&) = delete;
    ShGradSplitPass& operator=(const ShGradSplitPass&) = delete;

    void bind_buffers(VkBuffer interleaved, VkBuffer dc, VkBuffer rest);
    void record(VkCommandBuffer cmd, uint32_t N, uint32_t K);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};
