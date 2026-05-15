#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace l1_loss_bind {
constexpr uint32_t RENDERED = 0;
constexpr uint32_t TARGET   = 1;
constexpr uint32_t DLDPIX   = 2;
constexpr uint32_t PARTIALS = 3;
}

struct L1LossPushConstants {
    uint32_t num_elements;
    float scale;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(L1LossPushConstants) == 16, "L1LossPushConstants must be 16 bytes");

class L1LossPass {
public:
    explicit L1LossPass(VulkanContext& ctx);
    ~L1LossPass() = default;

    L1LossPass(const L1LossPass&) = delete;
    L1LossPass& operator=(const L1LossPass&) = delete;

    void bind_buffers(VkBuffer rendered,
                      VkBuffer target,
                      VkBuffer dL_dpixels,
                      VkBuffer partials);
    void dispatch_sync(uint32_t num_elements);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};
