#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace raw_activation_bind {
constexpr uint32_t RAW_SCALES       = 0;
constexpr uint32_t RAW_ROTATIONS    = 1;
constexpr uint32_t RAW_OPACITIES    = 2;
constexpr uint32_t RAW_SH_DC        = 3;
constexpr uint32_t RAW_SH_REST      = 4;
constexpr uint32_t ACTIVE_SCALES    = 5;
constexpr uint32_t ACTIVE_ROTATIONS = 6;
constexpr uint32_t ACTIVE_OPACITIES = 7;
constexpr uint32_t ACTIVE_SH        = 8;
}

struct RawActivationPushConstants {
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
static_assert(sizeof(RawActivationPushConstants) == 16, "RawActivationPushConstants must be 16 bytes");

class RawActivationPass {
public:
    explicit RawActivationPass(VulkanContext& ctx);
    ~RawActivationPass() = default;

    RawActivationPass(const RawActivationPass&) = delete;
    RawActivationPass& operator=(const RawActivationPass&) = delete;

    void bind_buffers(VkBuffer raw_scales,
                      VkBuffer raw_rotations,
                      VkBuffer raw_opacities,
                      VkBuffer raw_sh_dc,
                      VkBuffer raw_sh_rest,
                      VkBuffer active_scales,
                      VkBuffer active_rotations,
                      VkBuffer active_opacities,
                      VkBuffer active_sh);
    void record(VkCommandBuffer cmd, uint32_t N, uint32_t K);
    void dispatch_sync(uint32_t N, uint32_t K);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};
