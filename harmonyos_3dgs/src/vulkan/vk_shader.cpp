#include "vulkan/vk_shader.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("cannot open SPIR-V file: " + path);
    std::streamsize sz = f.tellg();
    if (sz <= 0 || (sz % 4) != 0)
        throw std::runtime_error("bad SPIR-V size for " + path);
    std::vector<uint32_t> words(static_cast<std::size_t>(sz) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(words.data()), sz);
    return words;
}

}  // namespace

VulkanShader::VulkanShader(VulkanContext& ctx, const std::string& spv_path)
    : ctx_(ctx) {
    auto spv = read_spv(spv_path);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    VK_CHECK(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &module_));
}

VulkanShader::VulkanShader(VulkanContext& ctx, const uint8_t* spirv_bytes,
                            std::size_t byte_size)
    : ctx_(ctx)
{
    if (byte_size == 0 || (byte_size % 4) != 0) {
        throw std::runtime_error(
            "VulkanShader: SPIR-V byte_size must be >0 and multiple of 4, got " +
            std::to_string(byte_size));
    }

    // SPIR-V is uint32_t[]; need 4-byte alignment. std::vector<uint32_t> gives it.
    std::vector<uint32_t> aligned(byte_size / 4);
    std::memcpy(aligned.data(), spirv_bytes, byte_size);

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = byte_size;
    info.pCode    = aligned.data();

    VK_CHECK(vkCreateShaderModule(ctx_.device(), &info, nullptr, &module_));
}

VulkanShader::~VulkanShader() {
    if (module_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx_.device(), module_, nullptr);
}
