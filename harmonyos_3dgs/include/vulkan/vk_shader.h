// vk_shader.h -- SPIR-V loader + VkShaderModule wrapper.

#pragma once

#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <string>

class VulkanShader {
public:
    /// Load SPIR-V from file and create a VkShaderModule.
    VulkanShader(VulkanContext& ctx, const std::string& spv_path);
    ~VulkanShader();

    VulkanShader(const VulkanShader&)            = delete;
    VulkanShader& operator=(const VulkanShader&) = delete;

    VkShaderModule handle() const { return module_; }

private:
    VulkanContext& ctx_;
    VkShaderModule module_ = VK_NULL_HANDLE;
};
