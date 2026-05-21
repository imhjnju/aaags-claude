#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

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
    ~DssimLossPass();

    DssimLossPass(const DssimLossPass&) = delete;
    DssimLossPass& operator=(const DssimLossPass&) = delete;

    void bind_buffers(VkBuffer rendered,
                      VkBuffer target,
                      VkBuffer dL_dpixels,
                      VkBuffer partials,
                      VkBuffer alpha,
                      VkBuffer beta,
                      VkBuffer gamma,
                      VkBuffer x2,
                      VkBuffer y2,
                      VkBuffer xy,
                      VkBuffer scratch,
                      VkBuffer mu1);
    void precompute_target_terms(uint32_t W, uint32_t H, float lambda_dssim);
    void dispatch_sync(uint32_t W, uint32_t H, float lambda_dssim);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> target_blur5_h_shader_;
    std::unique_ptr<VulkanShader> target_blur5_v_shader_;
    std::unique_ptr<VulkanShader> blur5_h_shader_;
    std::unique_ptr<VulkanShader> blur5_v_terms_shader_;
    std::unique_ptr<VulkanShader> transpose3_v_shader_;
    std::unique_ptr<VulkanShader> transpose3_h_grad_shader_;
    std::unique_ptr<VulkanComputePipeline> target_blur5_h_pipeline_;
    std::unique_ptr<VulkanComputePipeline> target_blur5_v_pipeline_;
    std::unique_ptr<VulkanComputePipeline> blur5_h_pipeline_;
    std::unique_ptr<VulkanComputePipeline> blur5_v_terms_pipeline_;
    std::unique_ptr<VulkanComputePipeline> transpose3_v_pipeline_;
    std::unique_ptr<VulkanComputePipeline> transpose3_h_grad_pipeline_;
    VkDescriptorSet target_blur5_h_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet target_blur5_v_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet blur5_h_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet blur5_v_terms_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet transpose3_v_descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet transpose3_h_grad_descriptor_set_ = VK_NULL_HANDLE;
    VkBuffer bound_target_ = VK_NULL_HANDLE;
    VkBuffer bound_target_x2_ = VK_NULL_HANDLE;
    VkBuffer bound_target_y2_ = VK_NULL_HANDLE;
    VkBuffer bound_target_scratch_ = VK_NULL_HANDLE;
    VkBuffer bound_target_mu1_ = VK_NULL_HANDLE;
    bool target_terms_valid_ = false;
    uint32_t target_terms_W_ = 0;
    uint32_t target_terms_H_ = 0;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
};
