// SP-2: Vulkan forward pipeline — PreprocessorVulkan adapter header stub.
//
// Hard-error constraints (see dev_notes/master_plan/sp2_vulkan_forward_plan.md):
//   - No silent fallbacks: every Vulkan failure must throw or abort.
//   - No CPU shadow path: this class is Vulkan-only; do not mirror CPU work.
//   - Deterministic layout: buffers and bindings must match preprocess.comp
//     exactly — any divergence is a bug.
//   - Ownership: this adapter owns its PreprocessPass and GPU buffers; the
//     VulkanContext is held by reference and must outlive the adapter.
//
// This stub exposes the public surface only. Pipeline/buffer wiring arrives
// in later tasks (T5–T6) where the private handles below get populated.

#pragma once

#include "preprocessor.h"
#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <memory>

class PreprocessPass;

class PreprocessorVulkan : public Preprocessor {
public:
    explicit PreprocessorVulkan(VulkanContext& ctx);
    // Defined out-of-line in preprocessor_vulkan.cpp because
    // std::unique_ptr<PreprocessPass> requires PreprocessPass to be complete
    // at the destruction point.
    ~PreprocessorVulkan() override;

    PreprocessOutput process(const GaussianData& g, const Camera& cam,
                             const RenderConfig& cfg, FrameAllocator& alloc,
                             ForwardCache* cache = nullptr) override;

    // Layer 2: record into external command buffer.
    // Pre-conditions: buffers must be allocated and bound via PreprocessPass::bind_buffers()
    // before calling this. The caller is responsible for inserting a compute barrier
    // after this returns (use insert_compute_barrier). Spec §4.8.1.
    void record(VkCommandBuffer cmd, uint32_t num_gaussians, uint32_t sh_degree,
                uint32_t sh_coeffs_per_g, uint32_t num_tiles_x,
                uint32_t num_tiles_y, float scale_modifier);

    // Handles returned here are owned by *this; valid until the next process()
    // call or destruction. Do not call vkDestroyBuffer on them.
    VkBuffer means2D_buffer() const;
    VkBuffer depths_buffer() const;
    VkBuffer conic_opacity_packed_buffer() const;
    VkBuffer rgb_buffer() const;
    VkBuffer radii_buffer() const;
    VkBuffer tiles_touched_buffer() const;

private:
    VulkanContext& ctx_;
    std::unique_ptr<PreprocessPass> pass_;
    // ... private buffer handles filled in T5-T6
};
