#include "vulkan/vk_pipeline.h"

#include <cstdint>
#include <stdexcept>
#include <string>

VulkanComputePipeline::VulkanComputePipeline(VulkanContext& ctx,
                                             const VulkanShader& shader,
                                             uint32_t num_ssbo_bindings,
                                             uint32_t push_constant_bytes,
                                             uint32_t max_descriptor_sets)
    : ctx_(ctx), num_ssbo_bindings_(num_ssbo_bindings) {
    if (push_constant_bytes > ctx.capabilities().max_push_constants_size) {
        throw std::runtime_error(
            "VulkanComputePipeline: push_constant_bytes=" +
            std::to_string(push_constant_bytes) +
            " exceeds device limit=" +
            std::to_string(ctx.capabilities().max_push_constants_size) +
            "; use a UBO instead.");
    }

    // Descriptor set layout: N storage buffers, contiguous bindings.
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_ssbo_bindings);
    for (uint32_t i = 0; i < num_ssbo_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = num_ssbo_bindings;
    dslci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device(), &dslci, nullptr, &dsl_));

    // Pipeline layout (+ optional push constant).
    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl_;
    VkPushConstantRange pcr{};
    if (push_constant_bytes > 0) {
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = push_constant_bytes;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    VK_CHECK(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_));

    // Compute pipeline.
    VkPipelineShaderStageCreateInfo ssci{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = shader.handle();
    ssci.pName = "main";
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = layout_;
    VK_CHECK(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpci,
                                      nullptr, &pipeline_));

    // Descriptor pool sized for max_descriptor_sets, each with
    // num_ssbo_bindings storage buffers.
    if (num_ssbo_bindings > 0) {
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 num_ssbo_bindings * max_descriptor_sets};
        VkDescriptorPoolCreateInfo dpci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = max_descriptor_sets;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &dps;
        VK_CHECK(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_));
    }
}

VulkanComputePipeline::~VulkanComputePipeline() {
    VkDevice d = ctx_.device();
    if (pool_     != VK_NULL_HANDLE) vkDestroyDescriptorPool(d, pool_, nullptr);
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline_, nullptr);
    if (layout_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
    if (dsl_      != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, dsl_, nullptr);
}

VkDescriptorSet VulkanComputePipeline::allocateDescriptorSet(
    const std::vector<VkBuffer>& ssbos) {
    if (ssbos.size() != num_ssbo_bindings_)
        throw std::runtime_error(
            "allocateDescriptorSet: ssbos.size() mismatch with layout bindings");
    if (pool_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "allocateDescriptorSet: no descriptor pool (num_ssbo_bindings=0)");

    VkDescriptorSetAllocateInfo dsai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl_;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device(), &dsai, &dset));

    std::vector<VkDescriptorBufferInfo> dbis(ssbos.size());
    std::vector<VkWriteDescriptorSet> writes(ssbos.size());
    for (std::size_t i = 0; i < ssbos.size(); ++i) {
        dbis[i] = VkDescriptorBufferInfo{ssbos[i], 0, VK_WHOLE_SIZE};
        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = dset;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(ctx_.device(),
                           static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
    return dset;
}

void VulkanComputePipeline::dispatch_sync(VkDescriptorSet desc_set,
                                          uint32_t gx, uint32_t gy, uint32_t gz,
                                          const void* push_constants,
                                          uint32_t push_size)
{
    VkCommandBuffer cmd = ctx_.allocatePrimary();

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                             0, 1, &desc_set, 0, nullptr);
    if (push_size > 0 && push_constants != nullptr) {
        vkCmdPushConstants(cmd, layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_constants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);

    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);
}

void VulkanComputePipeline::record(VkCommandBuffer cmd,
                                    VkDescriptorSet desc_set,
                                    uint32_t gx, uint32_t gy, uint32_t gz,
                                    const void* push_constants,
                                    uint32_t push_size)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                             0, 1, &desc_set, 0, nullptr);
    if (push_size > 0 && push_constants != nullptr) {
        vkCmdPushConstants(cmd, layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_constants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);
}

void insert_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType          = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask  = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}
