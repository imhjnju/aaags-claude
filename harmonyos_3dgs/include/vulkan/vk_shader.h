// vk_shader.h -- SPIR-V loader + VkShaderModule wrapper.

#pragma once

#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>

class VulkanShader {
public:
    /// Load SPIR-V from file and create a VkShaderModule.
    VulkanShader(VulkanContext& ctx, const std::string& spv_path);

    /// Load SPIR-V from a raw byte stream (compatible with xxd -i output:
    /// `unsigned char <name>_spv[]` + `unsigned int <name>_spv_len`).
    /// `spirv_bytes` must point to valid SPIR-V words (multiple of 4 bytes).
    VulkanShader(VulkanContext& ctx, const uint8_t* spirv_bytes, std::size_t byte_size);

    ~VulkanShader();

    VulkanShader(const VulkanShader&)            = delete;
    VulkanShader& operator=(const VulkanShader&) = delete;

    VkShaderModule handle() const { return module_; }

private:
    VulkanContext& ctx_;
    VkShaderModule module_ = VK_NULL_HANDLE;
};
