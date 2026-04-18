// vk_pipeline.h -- Compute pipeline + descriptor set layout + pipeline
// layout, plus a descriptor pool big enough for a handful of sets. Phase 1
// only supports storage-buffer bindings (all at set 0, contiguous binding
// indices 0..N-1) and a single optional push-constant range in the compute
// stage. The real preprocessor / rasterizer pipelines will extend this when
// they need uniform buffers or images.

#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class VulkanComputePipeline {
public:
    VulkanComputePipeline(VulkanContext& ctx,
                          const VulkanShader& shader,
                          uint32_t num_ssbo_bindings,
                          uint32_t push_constant_bytes = 0,
                          uint32_t max_descriptor_sets = 4);
    ~VulkanComputePipeline();

    VulkanComputePipeline(const VulkanComputePipeline&)            = delete;
    VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;

    VkPipeline            handle()              const { return pipeline_; }
    VkPipelineLayout      layout()              const { return layout_; }
    VkDescriptorSetLayout descriptorSetLayout() const { return dsl_; }

    /// Allocate a descriptor set from the internal pool and bind the given
    /// SSBOs to bindings 0..ssbos.size()-1. Size must equal num_ssbo_bindings
    /// passed to the constructor.
    VkDescriptorSet allocateDescriptorSet(const std::vector<VkBuffer>& ssbos);

private:
    VulkanContext&        ctx_;
    uint32_t              num_ssbo_bindings_;
    VkDescriptorSetLayout dsl_       = VK_NULL_HANDLE;
    VkPipelineLayout      layout_    = VK_NULL_HANDLE;
    VkPipeline            pipeline_  = VK_NULL_HANDLE;
    VkDescriptorPool      pool_      = VK_NULL_HANDLE;
};
