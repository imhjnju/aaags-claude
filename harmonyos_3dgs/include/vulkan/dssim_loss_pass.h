#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

namespace dssim_loss_bind {
constexpr uint32_t RENDERED = 0;
constexpr uint32_t TARGET   = 1;
constexpr uint32_t DLDPIX   = 2;
constexpr uint32_t PARTIALS = 3;
constexpr uint32_t ALPHA    = 4;
constexpr uint32_t BETA     = 5;
constexpr uint32_t GAMMA    = 6;
}

struct DssimLossPushConstants {
    uint32_t W;
    uint32_t H;
    uint32_t num_elements;
    float lambda_dssim;
};
static_assert(sizeof(DssimLossPushConstants) == 16, "DssimLossPushConstants must be 16 bytes");

class DssimLossPass {
public:
    explicit DssimLossPass(VulkanContext& ctx);
    ~DssimLossPass() = default;

    DssimLossPass(const DssimLossPass&) = delete;
    DssimLossPass& operator=(const DssimLossPass&) = delete;

    void bind_buffers(VkBuffer rendered,
                      VkBuffer target,
                      VkBuffer dL_dpixels,
                      VkBuffer partials,
                      VkBuffer alpha,
                      VkBuffer beta,
                      VkBuffer gamma);
    void dispatch_sync(uint32_t W, uint32_t H, float lambda_dssim);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> terms_shader_;
    std::unique_ptr<VulkanShader> grad_shader_;
    std::unique_ptr<VulkanComputePipeline> terms_pipeline_;
    std::unique_ptr<VulkanComputePipeline> grad_pipeline_;
    VkDescriptorSet terms_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet grad_descriptor_set_ = VK_NULL_HANDLE;
};
