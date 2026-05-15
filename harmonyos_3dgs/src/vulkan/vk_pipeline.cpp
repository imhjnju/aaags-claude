#include "vulkan/vk_pipeline.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Legacy SSBO-only constructor: thin wrapper delegating to the mixed form
// with binding_types = {STORAGE_BUFFER, STORAGE_BUFFER, ...}.
// ---------------------------------------------------------------------------
VulkanComputePipeline::VulkanComputePipeline(VulkanContext& ctx,
                                             const VulkanShader& shader,
                                             uint32_t num_ssbo_bindings,
                                             uint32_t push_constant_bytes,
                                             uint32_t max_descriptor_sets)
    : VulkanComputePipeline(
          ctx, shader,
          std::vector<VkDescriptorType>(
              num_ssbo_bindings, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
          push_constant_bytes, max_descriptor_sets, nullptr) {}

// ---------------------------------------------------------------------------
// Main constructor: mixed SSBO/UBO bindings + optional specialization info.
// Builds descriptor set layout from binding_types, pipeline layout (with
// optional push constants), compute pipeline (with optional spec info),
// and a descriptor pool sized per-type to fit max_descriptor_sets.
// ---------------------------------------------------------------------------
VulkanComputePipeline::VulkanComputePipeline(
    VulkanContext& ctx,
    const VulkanShader& shader,
    const std::vector<VkDescriptorType>& binding_types,
    uint32_t push_constant_bytes,
    uint32_t max_descriptor_sets,
    const VkSpecializationInfo* spec_info)
    : ctx_(ctx), binding_types_(binding_types) {
    if (push_constant_bytes > ctx.capabilities().max_push_constants_size) {
        throw std::runtime_error(
            "VulkanComputePipeline: push_constant_bytes=" +
            std::to_string(push_constant_bytes) +
            " exceeds device limit=" +
            std::to_string(ctx.capabilities().max_push_constants_size) +
            "; use a UBO instead.");
    }

    const uint32_t num_bindings = static_cast<uint32_t>(binding_types_.size());

    // Descriptor set layout: one binding per entry in binding_types_.
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = binding_types_[i];
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = num_bindings;
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

    // Compute pipeline (with optional specialization constants).
    VkPipelineShaderStageCreateInfo ssci{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = shader.handle();
    ssci.pName = "main";
    ssci.pSpecializationInfo = spec_info;  // nullable; Vulkan accepts null
    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci;
    cpci.layout = layout_;
    VK_CHECK(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpci,
                                      nullptr, &pipeline_));

    // Descriptor pool sized per-type to fit max_descriptor_sets.
    // Aggregate counts by descriptor type, then emit one VkDescriptorPoolSize
    // entry per distinct type present in binding_types_.
    if (num_bindings > 0) {
        std::unordered_map<int, uint32_t> type_counts;  // key: int(VkDescriptorType)
        for (VkDescriptorType t : binding_types_) {
            type_counts[static_cast<int>(t)] += 1;
        }
        std::vector<VkDescriptorPoolSize> sizes;
        sizes.reserve(type_counts.size());
        for (auto& kv : type_counts) {
            VkDescriptorPoolSize dps{};
            dps.type = static_cast<VkDescriptorType>(kv.first);
            dps.descriptorCount = kv.second * max_descriptor_sets;
            sizes.push_back(dps);
        }
        VkDescriptorPoolCreateInfo dpci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = max_descriptor_sets;
        dpci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        dpci.pPoolSizes = sizes.data();
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
    // Count SSBO-typed bindings in the layout.
    uint32_t ssbo_count = 0;
    for (VkDescriptorType t : binding_types_) {
        if (t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ++ssbo_count;
    }
    if (ssbos.size() != ssbo_count)
        throw std::runtime_error(
            "allocateDescriptorSet: ssbos.size()=" +
            std::to_string(ssbos.size()) +
            " does not match SSBO-binding count=" +
            std::to_string(ssbo_count));
    if (pool_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "allocateDescriptorSet: no descriptor pool (empty binding list)");

    VkDescriptorSetAllocateInfo dsai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl_;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device(), &dsai, &dset));

    // Emit one write per SSBO-typed binding. UBO-typed bindings are left
    // alone here; caller must call update_ubo() for those.
    std::vector<VkDescriptorBufferInfo> dbis;
    dbis.reserve(ssbo_count);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(ssbo_count);
    std::size_t ssbo_idx = 0;
    for (uint32_t binding = 0; binding < binding_types_.size(); ++binding) {
        if (binding_types_[binding] != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            continue;
        }
        dbis.push_back(VkDescriptorBufferInfo{ssbos[ssbo_idx], 0, VK_WHOLE_SIZE});
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // Defer pBufferInfo wiring until after the loop since dbis may still
        // grow and invalidate pointers on reallocation — reserved above so
        // push_back does not reallocate, but be defensive.
        writes.push_back(w);
        ++ssbo_idx;
    }
    // Now that dbis is finalised, patch pBufferInfo pointers.
    for (std::size_t i = 0; i < writes.size(); ++i) {
        writes[i].pBufferInfo = &dbis[i];
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(ctx_.device(),
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
    return dset;
}

VkDescriptorSet VulkanComputePipeline::allocate_empty_descriptor_set() {
    if (pool_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "allocate_empty_descriptor_set: no descriptor pool (empty binding list)");

    VkDescriptorSetAllocateInfo dsai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl_;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device(), &dsai, &dset));
    // Intentionally no vkUpdateDescriptorSets here — caller populates via
    // update_ssbo() / update_ubo().
    return dset;
}

void VulkanComputePipeline::update_ssbo(VkDescriptorSet ds,
                                        uint32_t binding,
                                        VkBuffer buffer) {
    if (binding >= binding_types_.size())
        throw std::runtime_error(
            "update_ssbo: binding=" + std::to_string(binding) +
            " out of range (layout has " +
            std::to_string(binding_types_.size()) + " bindings)");
    if (binding_types_[binding] != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        throw std::runtime_error(
            "update_ssbo: binding=" + std::to_string(binding) +
            " is not VK_DESCRIPTOR_TYPE_STORAGE_BUFFER");

    VkDescriptorBufferInfo dbi{buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
}

void VulkanComputePipeline::update_ubo(VkDescriptorSet ds,
                                       uint32_t binding,
                                       VkBuffer buffer,
                                       VkDeviceSize range) {
    if (binding >= binding_types_.size())
        throw std::runtime_error(
            "update_ubo: binding=" + std::to_string(binding) +
            " out of range (layout has " +
            std::to_string(binding_types_.size()) + " bindings)");
    if (binding_types_[binding] != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        throw std::runtime_error(
            "update_ubo: binding=" + std::to_string(binding) +
            " is not VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER");

    VkDescriptorBufferInfo dbi{buffer, 0, range};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
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
    insert_compute_barrier(cmd);
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

void VulkanComputePipeline::reset_descriptor_pool()
{
    if (pool_ != VK_NULL_HANDLE)
        VK_CHECK(vkResetDescriptorPool(ctx_.device(), pool_, /*flags=*/0));
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
