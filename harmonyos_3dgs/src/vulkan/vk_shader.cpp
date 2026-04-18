#include "vulkan/vk_shader.h"

#include <fstream>
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

VulkanShader::~VulkanShader() {
    if (module_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx_.device(), module_, nullptr);
}
