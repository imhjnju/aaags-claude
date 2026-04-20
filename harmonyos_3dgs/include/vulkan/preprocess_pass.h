// preprocess_pass.h -- Thin owner of the preprocess.comp compute pipeline
// (spec §4.8.1). Holds the shader module, compute pipeline (with the 19
// bindings: 18 SSBO + 1 UBO mixed binding layout), and one descriptor set. Provides two
// dispatch entry points:
//   * dispatch_sync(): Layer 1 sync dispatch (used by tests/bring-up).
//   * record():        Layer 2 external cmd buffer recording (used by the
//                      chained preprocess→sort→rasterize forward pipeline).
// Caller constructs PreprocessPushConstants externally; PreprocessPass is
// intentionally stateless about those fields.
#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/preprocess_bindings.h"

#include <vulkan/vulkan.h>

#include <memory>

class PreprocessPass {
public:
    /// Input + output buffers wired to bindings 0..11, plus the CameraUBO
    /// at binding 12 and radius_f SSBO at binding 13. Names mirror
    /// preprocess_bind:: indices exactly.
    struct Buffers {
        // Inputs (SSBO, bindings 0..5)
        VkBuffer positions;
        VkBuffer scales;
        VkBuffer rotations;
        VkBuffer opacities;
        VkBuffer sh;
        VkBuffer filter_3D;
        // Outputs (SSBO, bindings 6..11)
        VkBuffer means2D;
        VkBuffer depths;
        VkBuffer conic_opacity_packed;
        VkBuffer rgb;
        VkBuffer radii;
        VkBuffer tiles_touched;
        // UBO (binding 12)
        VkBuffer camera_ubo;
        // Float radius SSBO (binding 13)
        VkBuffer radius_f;
        // Cache SSBOs for backward pass (bindings 14..18)
        VkBuffer cov3D_cache;        // binding 14, [N*6] floats
        VkBuffer p_view_cache;       // binding 15, [N*3] floats
        VkBuffer p_hom_w_cache;      // binding 16, [N]   floats
        VkBuffer cov2D_cache;        // binding 17, [N*3] floats (fa, fb, fc) dilated cov2D
        VkBuffer cov2D_det_cache;    // binding 18, [N]   floats det = fa*fc - fb*fb
    };

    /// @param spec_training 1=training (skip SH RGB clamp), 0=inference (clamp)
    /// @param spec_eval_3D  always 0 in SP-2 (2D anti-aliasing path)
    PreprocessPass(VulkanContext& ctx,
                   uint32_t spec_training,
                   uint32_t spec_eval_3D);

    ~PreprocessPass() = default;

    PreprocessPass(const PreprocessPass&)            = delete;
    PreprocessPass& operator=(const PreprocessPass&) = delete;

    /// Update buffer bindings on the pre-allocated descriptor set. Safe to
    /// call multiple times per-frame — uses vkUpdateDescriptorSets in-place
    /// rather than allocating a fresh set, so the descriptor pool is never
    /// exhausted regardless of how often buffers are rewired.
    void bind_buffers(const Buffers& b);

    /// Layer 1: allocate a cmd buffer, record, submit, wait. Convenience.
    void dispatch_sync(const PreprocessPushConstants& pc);

    /// Layer 2: record dispatch into an external cmd buffer. Caller owns
    /// begin/end/submit and any surrounding barriers.
    void record(VkCommandBuffer cmd, const PreprocessPushConstants& pc);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader> shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};
