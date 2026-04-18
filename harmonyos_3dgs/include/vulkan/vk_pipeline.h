// vk_pipeline.h -- Compute pipeline + descriptor set layout + pipeline
// layout, plus a descriptor pool big enough for a handful of sets. Two
// constructor forms are supported:
//   1) SSBO-only (legacy, Phase 1): N contiguous storage-buffer bindings.
//   2) Mixed-binding (SP-2): caller supplies a vector of descriptor types
//      so preprocess/rasterize pipelines can mix storage buffers with a
//      uniform buffer (CameraUBO / RasterUBO) in the same descriptor set.
// Push-constant range is a single compute-stage range, still sized by the
// push_constant_bytes argument. Specialization info is optional and lets
// callers inject spec_training / spec_eval_3D style constants at pipeline
// creation without recompiling SPIR-V.

#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class VulkanComputePipeline {
public:
    /// SSBO-only constructor (legacy Phase 1 form).
    /// All bindings are VK_DESCRIPTOR_TYPE_STORAGE_BUFFER at indices 0..N-1.
    VulkanComputePipeline(VulkanContext& ctx,
                          const VulkanShader& shader,
                          uint32_t num_ssbo_bindings,
                          uint32_t push_constant_bytes = 0,
                          uint32_t max_descriptor_sets = 4);

    /// Mixed-binding constructor (SP-2).
    /// binding_types[i] specifies the descriptor type for binding i
    /// (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).
    /// Pass spec_info (non-null) to specialize the compute stage at pipeline
    /// creation; pass nullptr to skip specialization.
    VulkanComputePipeline(VulkanContext& ctx,
                          const VulkanShader& shader,
                          const std::vector<VkDescriptorType>& binding_types,
                          uint32_t push_constant_bytes = 0,
                          uint32_t max_descriptor_sets = 4,
                          const VkSpecializationInfo* spec_info = nullptr);

    ~VulkanComputePipeline();

    VulkanComputePipeline(const VulkanComputePipeline&)            = delete;
    VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;

    VkPipeline            handle()              const { return pipeline_; }
    VkPipelineLayout      layout()              const { return layout_; }
    VkDescriptorSetLayout descriptorSetLayout() const { return dsl_; }

    /// Allocate a descriptor set from the internal pool and bind the given
    /// SSBOs to bindings 0..ssbos.size()-1. For SSBO-only pipelines,
    /// ssbos.size() must equal num_ssbo_bindings passed to the constructor.
    /// For mixed-binding pipelines, ssbos.size() must equal the number of
    /// SSBO-typed bindings in binding_types, and those SSBOs are assigned
    /// in-order to the SSBO-typed binding indices. UBO bindings are left
    /// unbound; the caller must invoke update_ubo() to populate them.
    VkDescriptorSet allocateDescriptorSet(const std::vector<VkBuffer>& ssbos);

    /// Update a single UBO binding on an already-allocated descriptor set.
    /// The binding index must refer to a VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    /// binding declared in the layout.
    void update_ubo(VkDescriptorSet ds,
                    uint32_t binding,
                    VkBuffer buffer,
                    VkDeviceSize range);

    /// Layer 1 dispatch (sync, bring-up). Allocates an internal command buffer,
    /// binds this pipeline + the given descriptor set, pushes constants if any,
    /// dispatches, submits, and vkQueueWaitIdles. Use for tests and smoke runs.
    void dispatch_sync(VkDescriptorSet descriptor_set,
                       uint32_t gx, uint32_t gy, uint32_t gz,
                       const void* push_constants = nullptr,
                       uint32_t push_size = 0);

    /// Layer 2 dispatch (record to external command buffer). Caller owns cmd
    /// buffer's begin/end/submit lifecycle and any barriers between stages.
    /// Used by SP-2+ chained pipeline (preprocess → sort → rasterize).
    void record(VkCommandBuffer cmd,
                VkDescriptorSet descriptor_set,
                uint32_t gx, uint32_t gy, uint32_t gz,
                const void* push_constants = nullptr,
                uint32_t push_size = 0);

private:
    VulkanContext&        ctx_;
    // Full per-binding type list (length == binding count). For SSBO-only
    // pipelines every entry is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
    std::vector<VkDescriptorType> binding_types_;
    VkDescriptorSetLayout dsl_       = VK_NULL_HANDLE;
    VkPipelineLayout      layout_    = VK_NULL_HANDLE;
    VkPipeline            pipeline_  = VK_NULL_HANDLE;
    VkDescriptorPool      pool_      = VK_NULL_HANDLE;
};

/// Insert a compute-to-compute barrier on SSBO reads/writes. Called by the
/// caller of record() between two dispatches to ensure the second sees the
/// first's writes.
void insert_compute_barrier(VkCommandBuffer cmd);
